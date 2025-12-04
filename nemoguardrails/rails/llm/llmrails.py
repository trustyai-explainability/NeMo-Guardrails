# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Rails entry point - Refactored to use modular components."""

import asyncio
import json
import logging
import threading
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from typing_extensions import Self

from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.actions.v2_x.generation import LLMGenerationActionsV2dotx
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.serialization import json_to_state
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers import register_embedding_provider
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import (
    EmbeddingSearchProvider,
    OutputRailsStreamingConfig,
    RailsConfig,
)
from nemoguardrails.rails.llm.config_loader import ConfigLoader
from nemoguardrails.rails.llm.event_translator import EventTranslator
from nemoguardrails.rails.llm.kb_builder import KnowledgeBaseBuilder
from nemoguardrails.rails.llm.model_factory import ModelFactory
from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationRailsOptions,
    GenerationResponse,
)
from nemoguardrails.rails.llm.response_assembler import ResponseAssembler
from nemoguardrails.rails.llm.runtime_orchestrator import RuntimeOrchestrator
from nemoguardrails.rails.llm.utils import get_action_details_from_flow_id
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.utils import get_or_create_event_loop

log = logging.getLogger(__name__)


class LLMRails:
    """Rails based on a given configuration.

    Refactored to use modular components:
    - ConfigEnricher: Loads and enriches configuration
    - ModelFactory: Manages LLM instantiation
    - KnowledgeBaseBuilder: Builds knowledge base
    - EventTranslator: Converts messages to/from events
    - RuntimeOrchestrator: Manages Colang runtime
    - ResponseAssembler: Assembles responses
    """

    config: RailsConfig
    llm: Optional[Union[BaseLLM, BaseChatModel]]
    runtime: Runtime

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initializes the LLMRails instance.

        Args:
            config: A rails configuration.
            llm: An optional LLM engine to use. If provided, this will be used as the main LLM
                and will take precedence over any main LLM specified in the config.
            verbose: Whether the logging should be verbose.
        """
        self.config = config
        self.verbose = verbose
        self.explain_info: Optional[ExplainInfo] = None

        if self.verbose:
            set_verbose(True, llm_calls=True)

        # We allow the user to register additional embedding search providers, so we keep
        # an index of them.
        self.embedding_search_providers = {}

        # The default embeddings model is usining FastEmbed
        self.default_embedding_model = "all-MiniLM-L6-v2"
        self.default_embedding_engine = "FastEmbed"
        self.default_embedding_params = {}

        # Load config with default flows and library content
        config_modules = ConfigLoader.load_config(config)

        # Initialize RuntimeOrchestrator
        self.runtime_orchestrator = RuntimeOrchestrator(config=config, verbose=verbose)
        self.runtime = self.runtime_orchestrator.runtime

        # Execute config.py init functions
        for config_module in config_modules:
            if hasattr(config_module, "init"):
                config_module.init(self)

        # Update embedding model if specified in config
        for model in self.config.models:
            if model.type == "embeddings":
                self.default_embedding_model = model.model
                self.default_embedding_engine = model.engine
                self.default_embedding_params = model.parameters or {}
                break

        # Initialize tracing adapters
        if config.tracing:
            from nemoguardrails.tracing import create_log_adapters

            self._log_adapters = create_log_adapters(config.tracing)
        else:
            self._log_adapters = None

        # Initialize ModelFactory
        self.model_factory = ModelFactory(config=config, injected_llm=llm)
        self.action_param_registry: Dict[str, Any] = {}

        # Initialize models and register action parameters
        self.model_factory.initialize_models(self.action_param_registry)
        for param_name, param_value in self.action_param_registry.items():
            self.runtime.register_action_param(param_name, param_value)

        # Store main LLM reference for backwards compatibility
        self.llm = self.model_factory.get_main_llm()
        self.main_llm_supports_streaming = self.model_factory.supports_streaming()

        # Expose specialized LLMs as attributes for convenient access
        for llm_type, llm_instance in self.model_factory.specialized_llms.items():
            setattr(self, f"{llm_type}_llm", llm_instance)

        # Initialize LLM Generation Actions
        llm_generation_actions_class = (
            LLMGenerationActions
            if config.colang_version == "1.0"
            else LLMGenerationActionsV2dotx
        )
        self.llm_generation_actions = llm_generation_actions_class(
            config=config,
            llm=self.llm,
            llm_task_manager=self.runtime.llm_task_manager,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
            verbose=verbose,
        )
        self.runtime.register_actions(self.llm_generation_actions, override=False)

        # Initialize KnowledgeBaseBuilder
        self.kb_builder = KnowledgeBaseBuilder(
            config=self.config,
            get_embeddings_search_provider_instance=self._get_embeddings_search_provider_instance,
        )

        # Build KB in separate thread
        loop = get_or_create_event_loop()
        if True or check_sync_call_from_async_loop():
            t = threading.Thread(target=asyncio.run, args=(self.kb_builder.build(),))
            t.start()
            t.join()
        else:
            loop.run_until_complete(self.kb_builder.build())

        # Register KB as action parameter
        self.kb = self.kb_builder.get_kb()
        self.runtime.register_action_param("kb", self.kb)

        # Initialize EventTranslator
        self.event_translator = EventTranslator(config=self.config)
        # For backwards compatibility
        self.events_history_cache = self.event_translator.events_history_cache

        # Initialize ResponseAssembler
        self.response_assembler = ResponseAssembler(config=self.config)

    def _get_embeddings_search_provider_instance(
        self, esp_config: Optional[EmbeddingSearchProvider] = None
    ) -> EmbeddingsIndex:
        """Get an embeddings search provider instance."""
        if esp_config is None:
            esp_config = EmbeddingSearchProvider()

        if esp_config.name == "default":
            from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

            return BasicEmbeddingsIndex(
                embedding_model=esp_config.parameters.get(
                    "embedding_model", self.default_embedding_model
                ),
                embedding_engine=esp_config.parameters.get(
                    "embedding_engine", self.default_embedding_engine
                ),
                embedding_params=esp_config.parameters.get(
                    "embedding_parameters", self.default_embedding_params
                ),
                cache_config=esp_config.cache,
                **{
                    k: v
                    for k, v in esp_config.parameters.items()
                    if k
                    in [
                        "use_batching",
                        "max_batch_size",
                        "matx_batch_hold",
                        "search_threshold",
                    ]
                    and v is not None
                },
            )
        else:
            if esp_config.name not in self.embedding_search_providers:
                raise Exception(f"Unknown embedding search provider: {esp_config.name}")
            else:
                kwargs = esp_config.parameters
                return self.embedding_search_providers[esp_config.name](**kwargs)

    def update_llm(self, llm: Union[BaseLLM, BaseChatModel]):
        """Replace the main LLM with the provided one."""
        self.llm = llm
        self.model_factory.update_main_llm(llm, self.action_param_registry)
        self.llm_generation_actions.llm = llm

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate a completion or next message.

        Delegates to components for actual processing.
        """
        import time

        from nemoguardrails.actions.llm.utils import get_colang_history
        from nemoguardrails.context import (
            explain_info_var,
            generation_options_var,
            llm_stats_var,
            raw_llm_request,
            streaming_handler_var,
        )
        from nemoguardrails.logging.stats import LLMStats
        from nemoguardrails.utils import extract_error_json

        # Input validation
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided.")
        if prompt is not None and messages is not None:
            raise ValueError("Only one of prompt or messages can be provided.")

        # Convert prompt to messages
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        # Deserialize state if needed
        if state is not None:
            if isinstance(state, dict) and state.get("version", "1.0") == "2.x":
                state = json_to_state(state["state"])

        # Process options
        gen_options = self._process_options(options, state)
        generation_options_var.set(gen_options)

        if streaming_handler:
            streaming_handler_var.set(streaming_handler)

        # Initialize explain info
        self.explain_info = self._ensure_explain_info()
        raw_llm_request.set(messages)

        # Inject generation options into messages
        if gen_options:
            messages = [
                {
                    "role": "context",
                    "content": {"generation_options": gen_options.model_dump()},
                }
            ] + (messages or [])

        # Handle bot message in context for non-dialog mode
        if (
            messages
            and messages[-1]["role"] == "assistant"
            and gen_options
            and gen_options.rails.dialog is False
        ):
            messages[0]["content"]["bot_message"] = messages[-1]["content"]
            messages = messages[0:-1]

        t0 = time.time()

        # Initialize LLM stats
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        # Translate messages to events
        if messages is None:
            raise ValueError("messages must be provided")
        else:
            events = self.event_translator.messages_to_events(messages, state)

        # Generate new events using runtime orchestrator
        try:
            log.info(
                f"DEBUG: Calling generate_events with state containing skip_output_rails: {state.get('skip_output_rails') if isinstance(state, dict) else 'state is not dict'}"
            )
            (
                new_events,
                output_state,
                processing_log,
            ) = await self.runtime_orchestrator.generate_events(
                events=events, state=state
            )
        except Exception as e:
            log.error("Error in generate_async: %s", e, exc_info=True)
            streaming_handler = streaming_handler_var.get()
            if streaming_handler:
                error_message = str(e)
                error_dict = extract_error_json(error_message)
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)
            raise

        # Cache events for Colang 1.0
        if self.config.colang_version == "1.0":
            responses, _, _, _ = self.response_assembler._extract_from_events(
                new_events
            )
            responses = [str(r) if not isinstance(r, str) else r for r in responses]
            new_message = {"role": "assistant", "content": "\n".join(responses)}

            if state is None:
                all_events = events + new_events
                self.event_translator.cache_events(
                    (messages or []) + [new_message], all_events
                )
            else:
                output_state = {"events": events + new_events}

        # Log conversation history
        all_events = (
            events + new_events if self.config.colang_version == "1.0" else new_events
        )
        self.explain_info.colang_history = get_colang_history(all_events)
        if self.verbose:
            log.info(
                f"Conversation history so far: \n{self.explain_info.colang_history}"
            )

        total_time = time.time() - t0
        log.info(
            "--- :: Total processing took %.2f seconds. LLM Stats: %s"
            % (total_time, llm_stats)
        )

        # Close streaming handler
        streaming_handler = streaming_handler_var.get()
        if streaming_handler:
            await streaming_handler.push_chunk(END_OF_STREAM)

        # Assemble response
        if gen_options:
            res = self.response_assembler.assemble_response(
                new_events=new_events,
                all_events=all_events,
                output_state=output_state,
                processing_log=processing_log,
                gen_options=gen_options,
                prompt=prompt,
            )

            # Handle tracing if enabled
            if self.config.tracing.enabled and messages is not None:
                await self._handle_tracing(
                    messages, res, gen_options, processing_log, all_events
                )

            return res
        else:
            simple_res = self.response_assembler.assemble_simple_response(
                new_events=new_events, prompt=prompt
            )

            # Handle tracing if enabled (even when options is None)
            if self.config.tracing.enabled and messages is not None:
                # Convert simple response to GenerationResponse for tracing
                if isinstance(simple_res, dict):
                    trace_res = GenerationResponse(response=[simple_res], log=None)
                else:
                    trace_res = GenerationResponse(response=simple_res, log=None)
                await self._handle_tracing(
                    messages, trace_res, None, processing_log, all_events
                )
                # Return GenerationResponse when tracing is enabled
                return trace_res

            return simple_res

    def _process_options(
        self,
        options: Optional[Union[dict, GenerationOptions]],
        state: Optional[Any],
    ) -> Optional[GenerationOptions]:
        """Process and normalize generation options."""
        if state is not None:
            if options is None:
                return GenerationOptions()
            elif isinstance(options, dict):
                return GenerationOptions(**options)
            else:
                return options
        else:
            if options and isinstance(options, dict):
                return GenerationOptions(**options)
            elif isinstance(options, GenerationOptions):
                return options
            elif options is None:
                return None
            else:
                raise TypeError("options must be a dict or GenerationOptions")

    @staticmethod
    def _ensure_explain_info() -> ExplainInfo:
        """Ensure ExplainInfo variable is present in context."""
        from nemoguardrails.context import explain_info_var

        explain_info = explain_info_var.get()
        if explain_info is None:
            explain_info = ExplainInfo()
            explain_info_var.set(explain_info)
        return explain_info

    async def _handle_tracing(
        self,
        messages: List[dict],
        res: GenerationResponse,
        gen_options: Optional[GenerationOptions],
        processing_log: List[dict],
        all_events: List[dict],
    ):
        """Handle tracing export."""
        from nemoguardrails.actions.llm.utils import get_colang_history
        from nemoguardrails.logging.processing_log import compute_generation_log
        from nemoguardrails.rails.llm.options import GenerationLog
        from nemoguardrails.tracing import Tracer

        span_format = getattr(self.config.tracing, "span_format", "opentelemetry")
        enable_content_capture = getattr(
            self.config.tracing, "enable_content_capture", False
        )

        # If response.log is None but tracing is enabled, create a temporary log for tracing
        # without attaching it to the response (to avoid mutating user's response)
        if res.log is None:
            # Create a log from processing_log for tracing purposes
            _log = compute_generation_log(processing_log)
            temp_log = GenerationLog()
            temp_log.stats = _log.stats
            temp_log.activated_rails = _log.activated_rails or []
            # Collect llm_calls from activated_rails
            temp_log.llm_calls = []
            for activated_rail in _log.activated_rails or []:
                for executed_action in activated_rail.executed_actions:
                    temp_log.llm_calls.extend(executed_action.llm_calls)
            # Include internal events and colang history for comprehensive tracing
            temp_log.internal_events = all_events
            temp_log.colang_history = get_colang_history(all_events)
            # Create a temporary response with the log for tracing
            temp_response = GenerationResponse(response=res.response, log=temp_log)
            tracer = Tracer(
                input=messages,
                response=temp_response,
                adapters=self._log_adapters,
                span_format=span_format,
                enable_content_capture=enable_content_capture,
            )
        else:
            tracer = Tracer(
                input=messages,
                response=res,
                adapters=self._log_adapters,
                span_format=span_format,
                enable_content_capture=enable_content_capture,
            )
        await tracer.export_async()

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[dict] = None,
    ):
        """Synchronous version of generate_async."""
        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate` inside async code. "
                "You should replace with `await generate_async(...)` or use `nest_asyncio.apply()`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.generate_async(
                prompt=prompt, messages=messages, options=options, state=state
            )
        )

    def _validate_streaming_with_output_rails(self) -> None:
        """Validate streaming configuration with output rails."""
        if len(self.config.rails.output.flows) > 0 and (
            not self.config.rails.output.streaming
            or not self.config.rails.output.streaming.enabled
        ):
            raise ValueError(
                "stream_async() cannot be used when output rails are configured but "
                "rails.output.streaming.enabled is False. Either set "
                "rails.output.streaming.enabled to True in your configuration, or use "
                "generate_async() instead of stream_async()."
            )

    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_generation_metadata: Optional[bool] = False,
        generator: Optional[AsyncIterator[str]] = None,
    ) -> AsyncIterator[str]:
        """Simplified interface for getting streamed tokens from the LLM."""
        self._validate_streaming_with_output_rails()

        # if external generator provided, use it directly
        if generator:
            if (
                self.config.rails.output.streaming
                and self.config.rails.output.streaming.enabled
            ):
                return self._run_output_rails_in_streaming(
                    streaming_handler=generator,
                    output_rails_streaming_config=self.config.rails.output.streaming,
                    messages=messages,
                    prompt=prompt,
                )
            else:
                return generator

        self.explain_info = self._ensure_explain_info()

        streaming_handler = StreamingHandler(
            include_generation_metadata=include_generation_metadata
        )

        # Create properly managed task with exception handling
        async def _generation_task():
            try:
                # When output rails streaming is enabled, we need to skip the normal
                # output rails execution during the main flow, as they will be run
                # separately in _run_output_rails_in_streaming
                generation_options = options
                if (
                    self.config.rails.output.streaming
                    and self.config.rails.output.streaming.enabled
                ):
                    # Disable output rails in generation_options so the llm_flows.co check prevents them from running
                    if generation_options is None:
                        generation_options = GenerationOptions(
                            rails=GenerationRailsOptions(output=False)
                        )
                    elif isinstance(generation_options, dict):
                        generation_options = dict(generation_options)
                        if "rails" not in generation_options:
                            generation_options["rails"] = {}
                        generation_options["rails"]["output"] = False
                    else:
                        # It's a GenerationOptions object
                        generation_options = generation_options.model_copy(deep=True)
                        generation_options.rails.output = False

                await self.generate_async(
                    prompt=prompt,
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=generation_options,
                    state=state,
                )
            except Exception as e:
                log.error(f"Error in generation task: {e}", exc_info=True)
                from nemoguardrails.utils import extract_error_json

                error_message = str(e)
                error_dict = extract_error_json(error_message)
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)

        task = asyncio.create_task(_generation_task())

        # Store task reference
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.add(task)

        def task_done_callback(task):
            self._active_tasks.discard(task)

        task.add_done_callback(task_done_callback)

        # Wrap with output rails if configured
        if (
            self.config.rails.output.streaming
            and self.config.rails.output.streaming.enabled
        ):
            return self._run_output_rails_in_streaming(
                streaming_handler=streaming_handler,
                output_rails_streaming_config=self.config.rails.output.streaming,
                messages=messages,
                prompt=prompt,
            )
        else:
            return streaming_handler

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[str],
        output_rails_streaming_config: OutputRailsStreamingConfig,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        stream_first: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        """Run output rails in streaming mode."""
        from nemoguardrails.context import explain_info_var
        from nemoguardrails.rails.llm.buffer import ChunkBatch

        # Ensure explain_info_var is set so LLM calls during streaming output rails
        # are tracked properly. Use self.explain_info if available, otherwise ensure
        # it's created and set in the context. We need to ensure both reference the
        # same object so LLM calls are tracked correctly.
        if self.explain_info is None:
            self.explain_info = self._ensure_explain_info()
        # Always set the context variable to point to self.explain_info
        explain_info_var.set(self.explain_info)

        # Get buffer strategy from config
        buffer_strategy = get_buffer_strategy(output_rails_streaming_config)

        # Determine stream_first behavior
        should_stream_first = (
            stream_first
            if stream_first is not None
            else output_rails_streaming_config.stream_first
        )

        # Get output flows
        output_flows = self.config.rails.output.flows or []
        is_parallel = self.config.rails.output.parallel

        # Get action details for each flow if parallel
        flows_with_params = {}
        if is_parallel and output_flows:
            try:
                flows = self.config.flows
                for flow_id in output_flows:
                    try:
                        action_name, action_params = get_action_details_from_flow_id(
                            flow_id, flows
                        )
                        flows_with_params[flow_id] = {
                            "action_name": action_name,
                            "params": action_params,
                        }
                    except (ValueError, KeyError) as e:
                        log.warning(
                            f"Could not get action details for flow {flow_id}: {e}"
                        )
            except Exception as e:
                log.error(f"Error getting flow details: {e}")

        # Process stream using buffer strategy
        log.info("Starting to process stream with buffer strategy")
        batch_count = 0
        async for chunk_batch in buffer_strategy.process_stream(streaming_handler):
            batch_count += 1
            log.info(
                f"Received chunk_batch #{batch_count} with {len(chunk_batch.user_output_chunks)} chunks"
            )
            # Format the processing context
            processing_text = buffer_strategy.format_chunks(
                chunk_batch.processing_context
            )

            # If stream_first is True, yield chunks immediately before checking rails
            if should_stream_first:
                for chunk in chunk_batch.user_output_chunks:
                    yield chunk

            # Create events with bot_message for rail processing
            events = []
            if messages:
                events.extend(self.event_translator.messages_to_events(messages))
            events.append(
                {
                    "type": "BotMessage",
                    "text": processing_text,
                }
            )
            # Add context update so actions can access bot_message
            log.info(
                f"Streaming output rail batch #{batch_count}: processing_text = {repr(processing_text)}"
            )
            events.append(
                {
                    "type": "ContextUpdate",
                    "data": {"bot_message": processing_text},
                }
            )

            # Run output rails
            blocked = False
            blocking_rail = None
            internal_error = None

            if is_parallel and flows_with_params:
                # Run parallel rails (only available in Colang 1.0)
                if hasattr(self.runtime, "_run_output_rails_in_parallel_streaming"):
                    result = await self.runtime._run_output_rails_in_parallel_streaming(  # type: ignore[attr-defined]
                        flows_with_params, events
                    )
                else:
                    raise RuntimeError(
                        "Parallel streaming output rails are only supported in Colang 1.0"
                    )
                if result.events:
                    event = result.events[0]
                    if event.get("error_type") == "internal_error":
                        internal_error = event.get("error_message")
                        blocked = True
                        blocking_rail = event.get("flow_id")
                    elif event.get("intent") == "stop":
                        blocked = True
                        blocking_rail = event.get("flow_id")
            else:
                # Run sequential rails
                for flow_id in output_flows:
                    try:
                        # Ensure explain_info_var is set before processing events
                        # so LLM calls are tracked properly
                        if self.explain_info is None:
                            self.explain_info = self._ensure_explain_info()
                        explain_info_var.set(self.explain_info)

                        # Create start event
                        start_event = {
                            "type": "StartOutputRail",
                            "flow_id": flow_id,
                        }
                        rail_events = events + [start_event]

                        # Process the rail
                        (
                            new_events,
                            _,
                        ) = await self.runtime_orchestrator.process_events_async(
                            rail_events, blocking=True
                        )

                        # Check if rail blocked (look for stop intent or exception)
                        for event in new_events:
                            if (
                                event.get("type") == "BotIntent"
                                and event.get("intent") == "stop"
                            ):
                                blocked = True
                                blocking_rail = flow_id
                                break
                            elif event.get("type", "").endswith("Exception"):
                                blocked = True
                                blocking_rail = flow_id
                                break
                            # Also check for action results with output_mapping
                            elif event.get("type") == "InternalSystemActionFinished":
                                action_name = event.get("action_name")
                                return_value = event.get("return_value")
                                log.info(
                                    f"Sequential rail {flow_id}: action {action_name} returned {return_value}"
                                )
                                if action_name and return_value is not None:
                                    action_func = (
                                        self.runtime.action_dispatcher.get_action(
                                            action_name
                                        )
                                    )
                                    if action_func:
                                        is_blocked = is_output_blocked(
                                            return_value, action_func
                                        )
                                        log.info(
                                            f"Action {action_name} output_blocked={is_blocked}"
                                        )
                                        if is_blocked:
                                            blocked = True
                                            blocking_rail = flow_id
                                            break

                        if blocked:
                            break
                    except Exception as e:
                        log.error(f"Error in sequential rail {flow_id}: {e}")
                        blocked = True
                        blocking_rail = flow_id
                        internal_error = str(e)
                        break

            # If blocked, yield error and stop
            if blocked:
                # Create error message
                if internal_error:
                    error_dict = {
                        "error": {
                            "message": f"Internal error in {blocking_rail} rail: {internal_error}",
                            "type": "internal_error",
                            "param": blocking_rail,
                            "code": "rail_execution_failure",
                        }
                    }
                else:
                    error_dict = {
                        "error": {
                            "message": f"Blocked by {blocking_rail} rails.",
                            "type": "guardrails_violation",
                            "param": blocking_rail,
                            "code": "content_blocked",
                        }
                    }
                yield json.dumps(error_dict)
                return

            # If not blocked and stream_first is False, yield chunks after rails check
            if not should_stream_first:
                for chunk in chunk_batch.user_output_chunks:
                    yield chunk

    async def generate_events_async(self, events: List[dict]) -> List[dict]:
        """Generate the next events based on the provided history."""
        import time

        from nemoguardrails.actions.llm.utils import get_colang_history
        from nemoguardrails.context import llm_stats_var
        from nemoguardrails.logging.stats import LLMStats

        t0 = time.time()

        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        processing_log = []
        new_events = await self.runtime.generate_events(
            events, processing_log=processing_log
        )

        if self.verbose:
            history = get_colang_history(events)
            log.info(f"Conversation history so far: \n{history}")

        log.info("--- :: Total processing took %.2f seconds." % (time.time() - t0))
        log.info("--- :: Stats: %s" % llm_stats)

        return new_events

    def generate_events(self, events: List[dict]) -> List[dict]:
        """Synchronous version of generate_events_async."""
        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.generate_events_async(events=events))

    async def process_events_async(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
        """Process a sequence of events in a given state."""
        return await self.runtime_orchestrator.process_events_async(
            events, state, blocking
        )

    def process_events(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
        """Synchronous version of process_events_async."""
        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `process_events` inside async code. "
                "You should replace with `await process_events_async(...)`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.process_events_async(events, state, blocking)
        )

    # Registration methods

    def register_action(self, action: Callable, name: Optional[str] = None) -> Self:
        """Register a custom action."""
        self.runtime.register_action(action, name)
        return self

    def register_action_param(self, name: str, value: Any) -> Self:
        """Register a custom action parameter."""
        self.runtime.register_action_param(name, value)
        return self

    def register_filter(self, filter_fn: Callable, name: Optional[str] = None) -> Self:
        """Register a custom filter."""
        self.runtime.llm_task_manager.register_filter(filter_fn, name)
        return self

    def register_output_parser(self, output_parser: Callable, name: str) -> Self:
        """Register a custom output parser."""
        self.runtime.llm_task_manager.register_output_parser(output_parser, name)
        return self

    def register_prompt_context(self, name: str, value_or_fn: Any) -> Self:
        """Register a value to be included in the prompt context."""
        self.runtime.llm_task_manager.register_prompt_context(name, value_or_fn)
        return self

    def register_embedding_search_provider(
        self, name: str, cls: Type[EmbeddingsIndex]
    ) -> Self:
        """Register a new embedding search provider."""
        self.embedding_search_providers[name] = cls
        return self

    def register_embedding_provider(
        self, cls: Type[EmbeddingModel], name: Optional[str] = None
    ) -> Self:
        """Register a custom embedding provider."""
        register_embedding_provider(engine_name=name, model=cls)
        return self

    def explain(self) -> ExplainInfo:
        """Return the latest ExplainInfo object."""
        if self.explain_info is None:
            self.explain_info = self._ensure_explain_info()
        return self.explain_info

    def _prepare_model_kwargs(self, model_config) -> dict:
        """Prepare kwargs for model initialization, including API key from environment.

        This method is maintained for backwards compatibility.
        It delegates to ModelFactory._prepare_model_kwargs.

        Args:
            model_config: The model configuration object.

        Returns:
            Dictionary of kwargs for model initialization.
        """
        return self.model_factory._prepare_model_kwargs(model_config)

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        if state["config"].config_path:
            config = RailsConfig.from_path(state["config"].config_path)
        else:
            config = state["config"]
        self.__init__(config=config, verbose=False)


# Re-export for backwards compatibility
__all__ = [
    "LLMRails",
    "get_action_details_from_flow_id",
    "GenerationOptions",
    "GenerationResponse",
]

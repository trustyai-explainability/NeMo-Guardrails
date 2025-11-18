"""Rails API - Main facade for NeMo Guardrails functionality."""

import asyncio
import logging
import threading
import time
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
from nemoguardrails.actions.llm.utils import get_colang_history
from nemoguardrails.actions.v2_x.generation import LLMGenerationActionsV2dotx
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.serialization import (
    json_to_state,
    state_to_json,
)
from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.config_loader import ConfigLoader
from nemoguardrails.rails.llm.event_translator import EventTranslator
from nemoguardrails.rails.llm.kb_builder import KnowledgeBaseBuilder
from nemoguardrails.rails.llm.model_factory import ModelFactory
from nemoguardrails.rails.llm.options import (
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.rails.llm.response_assembler import ResponseAssembler
from nemoguardrails.rails.llm.runtime_orchestrator import RuntimeOrchestrator
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.utils import (
    get_or_create_event_loop,
)

log = logging.getLogger(__name__)


class RailsAPI:
    """
    Main API facade for NeMo Guardrails.
    """

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initialize the RailsAPI.

        Args:
            config: A rails configuration.
            llm: An optional LLM engine to use.
            verbose: Whether the logging should be verbose.
        """
        self.config = config
        self.verbose = verbose
        self.explain_info: Optional[ExplainInfo] = None

        if self.verbose:
            set_verbose(True, llm_calls=True)

        # Initialize embedding configuration
        self.embedding_search_providers = {}
        self.default_embedding_model = "all-MiniLM-L6-v2"
        self.default_embedding_engine = "FastEmbed"
        self.default_embedding_params = {}

        # Initialize components
        self._init_components(llm)

    def _init_components(self, llm: Optional[Union[BaseLLM, BaseChatModel]]):
        """Initialize all components.

        Args:
            llm: Optional LLM to inject.
        """
        # 1.  is already done via config parameter
        # Additional config loading/processing could be added here

        # 2. Initialize RuntimeOrchestrator first (needs to be available for other components)
        self.runtime_orchestrator = RuntimeOrchestrator(
            config=self.config, verbose=self.verbose
        )
        self.runtime: Runtime = self.runtime_orchestrator.runtime

        # Registry for action parameters
        self.action_param_registry: Dict[str, Any] = {}

        # 3. Initialize ModelFactory
        self.model_factory = ModelFactory(config=self.config, injected_llm=llm)

        # Update embedding model if specified in config
        for model in self.config.models:
            if model.type == "embeddings":
                self.default_embedding_model = model.model
                self.default_embedding_engine = model.engine
                self.default_embedding_params = model.parameters or {}
                break

        # Initialize models and populate action registry
        self.model_factory.initialize_models(self.action_param_registry)

        # Register action parameters with runtime
        for param_name, param_value in self.action_param_registry.items():
            self.runtime.register_action_param(param_name, param_value)

        # 4. Initialize LLM Generation Actions
        llm_generation_actions_class = (
            LLMGenerationActions
            if self.config.colang_version == "1.0"
            else LLMGenerationActionsV2dotx
        )
        self.llm_generation_actions = llm_generation_actions_class(
            config=self.config,
            llm=self.model_factory.get_main_llm(),
            llm_task_manager=self.runtime.llm_task_manager,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
            verbose=self.verbose,
        )
        self.runtime.register_actions(self.llm_generation_actions, override=False)

        # 5. Initialize KnowledgeBaseBuilder
        self.kb_builder = KnowledgeBaseBuilder(
            config=self.config,
            get_embeddings_search_provider_instance=self._get_embeddings_search_provider_instance,
        )

        # Initialize KB (in separate thread to avoid async issues)
        loop = get_or_create_event_loop()
        if True or check_sync_call_from_async_loop():
            t = threading.Thread(target=asyncio.run, args=(self.kb_builder.build(),))
            t.start()
            t.join()
        else:
            loop.run_until_complete(self.kb_builder.build())

        # Register KB as action parameter
        self.runtime.register_action_param("kb", self.kb_builder.get_kb())

        # 6. Initialize EventTranslator
        self.event_translator = EventTranslator(config=self.config)

        # 7. Initialize ResponseAssembler
        self.response_assembler = ResponseAssembler(config=self.config)

        # Initialize tracing adapters if configured
        if self.config.tracing:
            from nemoguardrails.tracing import create_log_adapters

            self._log_adapters = create_log_adapters(self.config.tracing)
        else:
            self._log_adapters = None

    def _get_embeddings_search_provider_instance(self, esp_config=None):
        """Get an embeddings search provider instance."""
        from nemoguardrails.rails.llm.config import EmbeddingSearchProvider

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

    @staticmethod
    def _ensure_explain_info() -> ExplainInfo:
        """Ensure that the ExplainInfo variable is present in the current context."""
        explain_info = explain_info_var.get()
        if explain_info is None:
            explain_info = ExplainInfo()
            explain_info_var.set(explain_info)
        return explain_info

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> Union[str, dict, GenerationResponse]:
        """Generate a completion or next message.

        Args:
            prompt: The prompt to be used for completion.
            messages: The history of messages.
            options: Options specific for the generation.
            state: The state object.
            streaming_handler: Optional streaming handler.

        Returns:
            The completion or next message.
        """
        # Input validation
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided.")
        if prompt is not None and messages is not None:
            raise ValueError("Only one of prompt or messages can be provided.")

        # Convert prompt to messages format
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        # Handle state deserialization
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
        events = self.event_translator.messages_to_events(messages, state)

        # Generate new events using runtime orchestrator
        try:
            new_events, output_state, processing_log = (
                await self.runtime_orchestrator.generate_events(
                    events=events, state=state
                )
            )
        except Exception as e:
            log.error("Error in generate_async: %s", e, exc_info=True)
            streaming_handler = streaming_handler_var.get()
            if streaming_handler:
                import json

                from nemoguardrails.utils import extract_error_json

                error_message = str(e)
                error_dict = extract_error_json(error_message)
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)
            raise

        # Update event translator cache for Colang 1.0
        if self.config.colang_version == "1.0":
            # Build the new message for caching
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

        # Close streaming handler if present
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
            if self.config.tracing.enabled:
                await self._handle_tracing(messages, res, gen_options)

            return res
        else:
            return self.response_assembler.assemble_simple_response(
                new_events=new_events, prompt=prompt
            )

    def _process_options(
        self,
        options: Optional[Union[dict, GenerationOptions]],
        state: Optional[Any],
    ) -> Optional[GenerationOptions]:
        """Process and normalize generation options.

        Args:
            options: Raw options.
            state: State object.

        Returns:
            Normalized GenerationOptions or None.
        """
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

    async def _handle_tracing(
        self,
        messages: List[dict],
        res: GenerationResponse,
        gen_options: GenerationOptions,
    ):
        """Handle tracing export.

        Args:
            messages: Input messages.
            res: Generation response.
            gen_options: Generation options.
        """
        from nemoguardrails.tracing import Tracer

        span_format = getattr(self.config.tracing, "span_format", "opentelemetry")
        enable_content_capture = getattr(
            self.config.tracing, "enable_content_capture", False
        )

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

    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_generation_metadata: Optional[bool] = False,
        generator: Optional[AsyncIterator[str]] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM.

        Note: This is a simplified stub. Full streaming implementation would
        require additional logic from the original LLMRails.stream_async method.
        """
        # Simplified implementation - full version would include output rails streaming
        self.explain_info = self._ensure_explain_info()

        streaming_handler = StreamingHandler(
            include_generation_metadata=include_generation_metadata
        )

        async def _generation_task():
            try:
                await self.generate_async(
                    prompt=prompt,
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=options,
                    state=state,
                )
            except Exception as e:
                log.error(f"Error in generation task: {e}", exc_info=True)
                import json

                from nemoguardrails.utils import extract_error_json

                error_message = str(e)
                error_dict = extract_error_json(error_message)
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)

        task = asyncio.create_task(_generation_task())

        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.add(task)

        def task_done_callback(task):
            self._active_tasks.discard(task)

        task.add_done_callback(task_done_callback)

        return streaming_handler

    # Additional API methods for compatibility

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
        from nemoguardrails.embeddings.providers import register_embedding_provider

        register_embedding_provider(engine_name=name, model=cls)
        return self

    def explain(self) -> ExplainInfo:
        """Return the latest ExplainInfo object."""
        if self.explain_info is None:
            self.explain_info = self._ensure_explain_info()
        return self.explain_info

    def update_llm(self, llm: Union[BaseLLM, BaseChatModel]):
        """Replace the main LLM with the provided one."""
        self.model_factory.update_main_llm(llm, self.action_param_registry)
        self.llm_generation_actions.llm = llm

import logging
import os
from typing import Any, Dict, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.llm.cache import CacheInterface, LFUCache
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)


class ModelFactory:
    """Factory for initializing and configuring LLM models."""

    def __init__(
        self,
        config: RailsConfig,
        injected_llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
    ):
        """Initialize the ModelFactory.

        Args:
            config: The rails configuration.
            injected_llm: An optional LLM provided via constructor that takes precedence.
        """
        self.config = config
        self.injected_llm = injected_llm
        self.main_llm: Optional[Union[BaseLLM, BaseChatModel]] = None
        self.specialized_llms: Dict[str, Union[BaseLLM, BaseChatModel]] = {}
        self.model_caches: Dict[str, CacheInterface] = {}
        self.main_llm_supports_streaming = False

    def initialize_models(
        self, action_param_registry: Dict[str, Any]
    ) -> Dict[str, Union[BaseLLM, BaseChatModel]]:
        """Initialize all LLM models from configuration.

        Args:
            action_param_registry: Registry to store action parameters.

        Returns:
            Dictionary of specialized LLMs (excluding main).
        """
        # Handle injected LLM first
        if self.injected_llm:
            self.main_llm = self.injected_llm
            action_param_registry["llm"] = self.main_llm
            self._configure_streaming(self.main_llm)

            # Warn if main LLM also specified in config
            if any(model.type == "main" for model in self.config.models):
                log.warning(
                    "Both an LLM was provided via constructor and a main LLM is specified in the config. "
                    "The LLM provided via constructor will be used and the main LLM from config will be ignored."
                )
        else:
            # Initialize main LLM from config
            main_model = next(
                (model for model in self.config.models if model.type == "main"), None
            )

            if main_model and main_model.model:
                kwargs = self._prepare_model_kwargs(main_model)
                self.main_llm = init_llm_model(
                    model_name=main_model.model,
                    provider_name=main_model.engine,
                    mode="chat",
                    kwargs=kwargs,
                )
                action_param_registry["llm"] = self.main_llm
                self._configure_streaming(
                    self.main_llm,
                    model_name=main_model.model,
                    provider_name=main_model.engine,
                )
            else:
                log.warning(
                    "No main LLM specified in the config and no LLM provided via constructor."
                )

        # Initialize specialized LLMs
        for llm_config in self.config.models:
            if llm_config.type in ["embeddings", "jailbreak_detection"]:
                continue

            # Skip main model - already initialized above
            if llm_config.type == "main":
                continue

            model_name = llm_config.model
            if not model_name:
                raise ValueError(
                    f"LLM Config model field not set for {llm_config.type}"
                )

            provider_name = llm_config.engine
            kwargs = self._prepare_model_kwargs(llm_config)
            mode = llm_config.mode

            llm_model = init_llm_model(
                model_name=model_name,
                provider_name=provider_name,
                mode=mode,
                kwargs=kwargs,
            )

            # Configure based on type
            if llm_config.type == "main":
                if not self.main_llm:
                    self.main_llm = llm_model
                    action_param_registry["llm"] = self.main_llm
            else:
                param_name = f"{llm_config.type}_llm"
                self.specialized_llms[llm_config.type] = llm_model
                action_param_registry[param_name] = llm_model

        # Register specialized LLMs dictionary
        action_param_registry["llms"] = self.specialized_llms

        # Initialize model caches
        self._initialize_model_caches(action_param_registry)

        return self.specialized_llms

    def _prepare_model_kwargs(self, model_config) -> dict:
        """Prepare kwargs for model initialization, including API key from environment.

        Args:
            model_config: The model configuration object.

        Returns:
            Dictionary of kwargs for model initialization.
        """
        kwargs = model_config.parameters or {}

        # Add API key from environment if specified
        if model_config.api_key_env_var:
            api_key = os.environ.get(model_config.api_key_env_var)
            if api_key:
                kwargs["api_key"] = api_key

        # Enable streaming token usage when streaming is enabled
        if self.config.streaming:
            kwargs["stream_usage"] = True

        return kwargs

    def _configure_streaming(
        self,
        llm: Union[BaseLLM, BaseChatModel],
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        """Configure streaming support for the LLM.

        Args:
            llm: The LLM model instance.
            model_name: Optional model name for logging.
            provider_name: Optional provider name for logging.
        """
        if not self.config.streaming:
            return

        if hasattr(llm, "streaming"):
            setattr(llm, "streaming", True)
            self.main_llm_supports_streaming = True
        else:
            self.main_llm_supports_streaming = False
            if model_name and provider_name:
                log.warning(
                    "Model %s from provider %s does not support streaming.",
                    model_name,
                    provider_name,
                )
            else:
                log.warning("Provided main LLM does not support streaming.")

    def _create_model_cache(self, model) -> LFUCache:
        """Create cache instance for a model based on its configuration.

        Args:
            model: The model configuration object.

        Returns:
            LFUCache: The cache instance.
        """
        if model.cache.maxsize <= 0:
            raise ValueError(
                f"Invalid cache maxsize for model '{model.type}': {model.cache.maxsize}. "
                "Capacity must be greater than 0. Skipping cache creation."
            )

        stats_logging_interval = None
        if model.cache.stats.enabled and model.cache.stats.log_interval is not None:
            stats_logging_interval = model.cache.stats.log_interval

        cache = LFUCache(
            maxsize=model.cache.maxsize,
            track_stats=model.cache.stats.enabled,
            stats_logging_interval=stats_logging_interval,
        )

        log.info(
            "Created cache for model '%s' with maxsize %s",
            model.type,
            model.cache.maxsize,
        )

        return cache

    def _initialize_model_caches(self, action_param_registry: Dict[str, Any]) -> None:
        """Initialize caches for configured models.

        Args:
            action_param_registry: Registry to store action parameters.
        """
        for model in self.config.models:
            if model.type in ["main", "embeddings"]:
                continue

            if model.cache and model.cache.enabled:
                cache = self._create_model_cache(model)
                self.model_caches[model.type] = cache

                log.info(
                    "Initialized model '%s' with cache %s",
                    model.type,
                    "enabled" if cache else "disabled",
                )

        if self.model_caches:
            action_param_registry["model_caches"] = self.model_caches

    def get_main_llm(self) -> Optional[Union[BaseLLM, BaseChatModel]]:
        """Get the main LLM instance."""
        return self.main_llm

    def get_specialized_llm(
        self, llm_type: str
    ) -> Optional[Union[BaseLLM, BaseChatModel]]:
        """Get a specialized LLM by type."""
        return self.specialized_llms.get(llm_type)

    def supports_streaming(self) -> bool:
        """Check if the main LLM supports streaming."""
        return self.main_llm_supports_streaming

    def update_main_llm(
        self, llm: Union[BaseLLM, BaseChatModel], action_param_registry: Dict[str, Any]
    ):
        """Update the main LLM instance.

        Args:
            llm: The new LLM instance.
            action_param_registry: Registry to update action parameters.
        """
        self.main_llm = llm
        action_param_registry["llm"] = llm

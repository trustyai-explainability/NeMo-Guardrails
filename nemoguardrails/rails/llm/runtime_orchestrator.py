"""Runtime orchestrator for managing Colang runtime execution."""

import asyncio
import logging
from typing import Any, List, Optional, Tuple, cast

from nemoguardrails.colang.v1_0.runtime.runtime import Runtime, RuntimeV1_0
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.colang.v2_x.runtime.serialization import state_to_json
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

# Semaphore for protecting process_events calls
process_events_semaphore = asyncio.Semaphore(1)


class RuntimeOrchestrator:
    """Orchestrates the Colang runtime execution.

    Handles runtime initialization, event generation, and process coordination.
    """

    def __init__(self, config: RailsConfig, verbose: bool = False):
        """Initialize the RuntimeOrchestrator.

        Args:
            config: The rails configuration.
            verbose: Whether to enable verbose logging.
        """
        self.config = config
        self.verbose = verbose

        if self.verbose:
            set_verbose(True, llm_calls=True)

        # Initialize the appropriate runtime based on Colang version
        if config.colang_version == "1.0":
            self.runtime = RuntimeV1_0(config=config, verbose=verbose)
        elif config.colang_version == "2.x":
            self.runtime = RuntimeV2_x(config=config, verbose=verbose)
        else:
            raise ValueError(f"Unsupported colang version: {config.colang_version}.")

    async def generate_events(
        self,
        events: List[dict],
        state: Optional[Any] = None,
    ) -> Tuple[List[dict], Optional[Any], List[dict]]:
        """Generate new events based on the history of events.

        Args:
            events: The history of events.
            state: Optional state object (for Colang 2.x).

        Returns:
            Tuple of (new_events, output_state, processing_log).
        """
        processing_log = []

        if self.config.colang_version == "1.0":
            # For Colang 1.0, we use generate_events
            state_events = []
            if state:
                assert isinstance(state, dict)
                state_events = state.get("events", [])

            new_events = await self.runtime.generate_events(
                state_events + events, processing_log=processing_log
            )
            output_state = None

        else:
            # For Colang 2.x, we use process_events
            instant_actions = ["UtteranceBotAction"]
            if self.config.rails.actions.instant_actions is not None:
                instant_actions = self.config.rails.actions.instant_actions

            runtime: RuntimeV2_x = cast(RuntimeV2_x, self.runtime)

            new_events, output_state = await runtime.process_events(
                events, state=state, instant_actions=instant_actions, blocking=True
            )

            # Encode output state as JSON
            output_state = {"state": state_to_json(output_state), "version": "2.x"}

        return new_events, output_state, processing_log

    async def process_events_async(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], Any]:
        """Process a sequence of events in a given state.

        Args:
            events: A sequence of events to process.
            state: The starting state.
            blocking: Whether to block on all actions.

        Returns:
            Tuple of (output_events, output_state).
        """
        # Protect process_events to be called only once at a time
        async with process_events_semaphore:
            output_events, output_state = await self.runtime.process_events(
                events, state, blocking
            )

        return (output_events, output_state)

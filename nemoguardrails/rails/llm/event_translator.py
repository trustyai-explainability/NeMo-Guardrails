"""Event translator for converting between messages and events."""

import logging
from typing import Any, Dict, List, Optional

from nemoguardrails.colang.v2_x.runtime.flows import Action, State
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.utils import get_history_cache_key
from nemoguardrails.utils import new_event_dict, new_uuid

log = logging.getLogger(__name__)


class EventTranslator:
    """Translates between messages and Colang events."""

    def __init__(self, config: RailsConfig):
        """Initialize the EventTranslator.

        Args:
            config: The rails configuration.
        """
        self.config = config
        self.events_history_cache: Dict[str, List[dict]] = {}

    def messages_to_events(
        self, messages: List[dict], state: Optional[Any] = None
    ) -> List[dict]:
        """Convert messages to events.

        Tries to find a prefix of messages for which we have already a list of events
        in the cache. For the rest, they are converted as is.

        Args:
            messages: The list of messages.
            state: Optional state object (used for Colang 2.x).

        Returns:
            A list of events.
        """
        events = []

        if self.config.colang_version == "1.0":
            events = self._messages_to_events_v1(messages)
        else:
            events = self._messages_to_events_v2(messages, state)

        return events

    def _messages_to_events_v1(self, messages: List[dict]) -> List[dict]:
        """Convert messages to events for Colang 1.0.

        Args:
            messages: The list of messages.

        Returns:
            A list of events.
        """
        events = []

        # Try to find the longest prefix of messages for which we have a cache
        p = len(messages) - 1
        while p > 0:
            cache_key = get_history_cache_key(messages[0:p])
            if cache_key in self.events_history_cache:
                events = self.events_history_cache[cache_key].copy()
                break
            p -= 1

        # For the rest of the messages, transform them directly into events
        for idx in range(p, len(messages)):
            msg = messages[idx]
            if msg["role"] == "user":
                events.append(
                    {
                        "type": "UtteranceUserActionFinished",
                        "final_transcript": msg["content"],
                    }
                )

                # If it's not the last message, also add the `UserMessage` event
                if idx != len(messages) - 1:
                    events.append(
                        {
                            "type": "UserMessage",
                            "text": msg["content"],
                        }
                    )

            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    events.append(
                        {"type": "BotToolCalls", "tool_calls": msg["tool_calls"]}
                    )
                else:
                    action_uid = new_uuid()
                    start_event = new_event_dict(
                        "StartUtteranceBotAction",
                        script=msg["content"],
                        action_uid=action_uid,
                    )
                    finished_event = new_event_dict(
                        "UtteranceBotActionFinished",
                        final_script=msg["content"],
                        is_success=True,
                        action_uid=action_uid,
                    )
                    events.extend([start_event, finished_event])

            elif msg["role"] == "context":
                events.append({"type": "ContextUpdate", "data": msg["content"]})

            elif msg["role"] == "event":
                events.append(msg["event"])

            elif msg["role"] == "system":
                events.append({"type": "SystemMessage", "content": msg["content"]})

            elif msg["role"] == "tool":
                # For the last tool message, create grouped tool event or synthetic UserMessage
                if idx == len(messages) - 1:
                    # Find the original user message for response generation
                    user_message = None
                    for prev_msg in reversed(messages[:idx]):
                        if prev_msg["role"] == "user":
                            user_message = prev_msg["content"]
                            break

                    if user_message:
                        # If tool input rails are configured, group all tool messages
                        if self.config.rails.tool_input.flows:
                            tool_messages = []
                            for tool_idx in range(len(messages)):
                                if messages[tool_idx]["role"] == "tool":
                                    tool_messages.append(
                                        {
                                            "content": messages[tool_idx]["content"],
                                            "name": messages[tool_idx].get(
                                                "name", "unknown"
                                            ),
                                            "tool_call_id": messages[tool_idx].get(
                                                "tool_call_id", ""
                                            ),
                                        }
                                    )

                            events.append(
                                {
                                    "type": "UserToolMessages",
                                    "tool_messages": tool_messages,
                                }
                            )
                        else:
                            events.append({"type": "UserMessage", "text": user_message})

        return events

    def _messages_to_events_v2(
        self, messages: List[dict], state: Optional[Any]
    ) -> List[dict]:
        """Convert messages to events for Colang 2.x.

        Args:
            messages: The list of messages.
            state: The state object.

        Returns:
            A list of events.
        """
        events = []

        for idx in range(len(messages)):
            msg = messages[idx]
            if msg["role"] == "user":
                events.append(
                    {
                        "type": "UtteranceUserActionFinished",
                        "final_transcript": msg["content"],
                    }
                )

            elif msg["role"] == "assistant":
                raise ValueError(
                    "Providing `assistant` messages as input is not supported for Colang 2.0 configurations."
                )

            elif msg["role"] == "context":
                events.append({"type": "ContextUpdate", "data": msg["content"]})

            elif msg["role"] == "event":
                events.append(msg["event"])

            elif msg["role"] == "system":
                events.append({"type": "SystemMessage", "content": msg["content"]})

            elif msg["role"] == "tool":
                if state is None:
                    raise ValueError(
                        "State object is required for tool messages in Colang 2.0"
                    )
                action_uid = msg["tool_call_id"]
                return_value = msg["content"]
                action: Action = state.actions[action_uid]
                events.append(
                    new_event_dict(
                        f"{action.name}Finished",
                        action_uid=action_uid,
                        action_name=action.name,
                        status="success",
                        is_success=True,
                        return_value=return_value,
                        events=[],
                    )
                )

        return events

    def cache_events(self, messages: List[dict], events: List[dict]):
        """Cache events for a sequence of messages.

        Args:
            messages: The list of messages.
            events: The corresponding events.
        """
        if self.config.colang_version == "1.0":
            cache_key = get_history_cache_key(messages)
            self.events_history_cache[cache_key] = events

    def clear_cache(self):
        """Clear the events history cache."""
        self.events_history_cache.clear()

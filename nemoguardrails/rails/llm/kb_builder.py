"""Knowledge Base builder component."""

import logging
from typing import Callable, Optional

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, RailsConfig

log = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """Builder for initializing and managing knowledge bases."""

    def __init__(
        self,
        config: RailsConfig,
        get_embeddings_search_provider_instance: Callable[
            [Optional[EmbeddingSearchProvider]], EmbeddingsIndex
        ],
    ):
        """Initialize the KnowledgeBaseBuilder.

        Args:
            config: The rails configuration.
            get_embeddings_search_provider_instance: Function to get embeddings search provider.
        """
        self.config = config
        self.get_embeddings_search_provider_instance = (
            get_embeddings_search_provider_instance
        )
        self.kb: Optional[KnowledgeBase] = None

    async def build(self) -> Optional[KnowledgeBase]:
        """Build the knowledge base from configuration.

        Returns:
            The initialized KnowledgeBase or None if no docs configured.
        """
        if not self.config.docs:
            self.kb = None
            return None

        documents = [doc.content for doc in self.config.docs]
        self.kb = KnowledgeBase(
            documents=documents,
            config=self.config.knowledge_base,
            get_embedding_search_provider_instance=self.get_embeddings_search_provider_instance,
        )
        self.kb.init()
        await self.kb.build()

        log.info("Knowledge base initialized with %d documents", len(documents))
        return self.kb

    def get_kb(self) -> Optional[KnowledgeBase]:
        """Get the knowledge base instance."""
        return self.kb

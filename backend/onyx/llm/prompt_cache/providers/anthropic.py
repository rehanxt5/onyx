"""Anthropic provider adapter for prompt caching."""

from collections.abc import Sequence
from functools import partial

from onyx.llm.model_request import ChatCompletionMessage
from onyx.llm.prompt_cache.models import CacheMetadata
from onyx.llm.prompt_cache.providers.base import PromptCacheProvider
from onyx.llm.prompt_cache.utils import prepare_messages_with_cacheable_transform

_MAX_BREAKPOINTS = 4


def _add_anthropic_cache_control(
    messages: Sequence[ChatCompletionMessage],
    boundary_indices: Sequence[int] = (),
    mark_tail: bool = True,
) -> Sequence[ChatCompletionMessage]:
    """Add cache_control breakpoints for Anthropic caching.

    Head (usually the system prompt): stays readable when truncation or the
    per-turn segments move everything after it. Boundaries (end of kept
    history): the next turn re-reads [system, history] instead of rewriting
    it. Tail: lets later loop iterations in this turn re-read the whole
    prefix; skipped when nothing will (one-shot prompt, no tools).

    Args:
        messages: Messages to transform
        boundary_indices: Indices of messages that end a stable sub-prefix
        mark_tail: Whether the last message gets a breakpoint

    Returns:
        Messages with cache_control added
    """
    result: list[ChatCompletionMessage] = list(messages)
    wanted: set[int] = {0}
    wanted.update(i for i in boundary_indices if 0 <= i < len(result))
    if mark_tail:
        wanted.add(len(result) - 1)
    for idx in sorted(wanted)[:_MAX_BREAKPOINTS]:
        result[idx] = result[idx].model_copy(
            update={"cache_control": {"type": "ephemeral"}}
        )
    return result


class AnthropicPromptCacheProvider(PromptCacheProvider):
    """Anthropic adapter for prompt caching (explicit caching with cache_control).
    implicit caching = just need to ensure byte-equivalent prefixes, and the provider
                       auto-detects and reuses them.
    explicit caching = the caller must do _something_ to enable provider-side caching.
    In this case, anthropic supports explicit caching via the cache_control parameter:
    https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    """

    def supports_caching(self) -> bool:
        """Anthropic supports explicit prompt caching."""
        return True

    def prepare_messages_for_caching(
        self,
        cacheable_prefix: list[ChatCompletionMessage] | None,
        suffix: list[ChatCompletionMessage],
        continuation: bool,
        cache_metadata: CacheMetadata | None,  # noqa: ARG002
        boundary_indices: Sequence[int] = (),
        mark_tail: bool = True,
    ) -> list[ChatCompletionMessage]:
        """Prepare messages for Anthropic caching.

        Anthropic requires cache_control parameter on cacheable messages.
        We add cache_control={"type": "ephemeral"} to all cacheable prefix messages.

        Args:
            cacheable_prefix: Optional cacheable prefix
            suffix: Non-cacheable suffix
            continuation: Whether to append suffix to last prefix message
            cache_metadata: Cache metadata (for future explicit caching support)

        Returns:
            Combined messages with cache_control on cacheable messages
        """
        return prepare_messages_with_cacheable_transform(
            cacheable_prefix=cacheable_prefix,
            suffix=suffix,
            continuation=continuation,
            transform_cacheable=partial(
                _add_anthropic_cache_control,
                boundary_indices=boundary_indices,
                mark_tail=mark_tail,
            ),
        )

    def extract_cache_metadata(
        self,
        response: dict,  # noqa: ARG002
        cache_key: str,  # noqa: ARG002
    ) -> CacheMetadata | None:
        """Extract cache metadata from Anthropic response.

        Anthropic may return cache identifiers in the response.
        For now, we don't extract detailed metadata (future explicit caching support).

        Args:
            response: Anthropic API response dictionary
            cache_key: Cache key used for this request

        Returns:
            CacheMetadata if extractable, None otherwise
        """
        # TODO: Extract cache identifiers from response when implementing explicit caching
        return None

    def get_cache_ttl_seconds(self) -> int:
        """Get cache TTL for Anthropic (5 minutes default)."""
        return 300

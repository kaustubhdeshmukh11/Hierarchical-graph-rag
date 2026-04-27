"""
Shared Groq LLM utilities — rate limiter, safe call with retry, JSON parsing.

Every module that calls Groq should import from here instead of rolling its own.
This ensures consistent rate-limit behaviour across graph_builder, graph_rag,
baseline_rag, demo_queries, trace_query, and explain_retrieval.

Rate-limit strategy (Groq free tier: 30 req/min, 6000 req/day):
  - Sliding-window limiter: never exceed GROQ_MAX_RPM calls per 60s
  - Inter-call delay: sleep GROQ_INTER_CALL_DELAY seconds between calls
  - 429 retry: exponential backoff (60s, 120s, 180s)
  - Daily quota detection: graceful exit with cache saved
"""

import collections
import json
import re
import time

from groq import Groq

import config


# =============================================================================
#  RATE LIMITER
# =============================================================================

class RateLimiter:
    """Sliding-window rate limiter — never exceeds max_per_minute calls/60s."""

    def __init__(self, max_per_minute: int = None):
        self.max_per_minute = max_per_minute or config.GROQ_MAX_RPM
        self._timestamps: collections.deque = collections.deque()

    def wait(self):
        """Block until it is safe to fire another LLM call."""
        now = time.time()
        # purge timestamps older than 60 s
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_per_minute:
            oldest = self._timestamps[0]
            sleep_for = 61.0 - (now - oldest)
            if sleep_for > 0:
                print(f"       [RateLimit] {len(self._timestamps)} req/60s "
                      f"-- sleeping {sleep_for:.1f}s ...")
                time.sleep(sleep_for)
            # purge again after sleep
            now = time.time()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()

        self._timestamps.append(time.time())


# Module-level singleton so all callers share one window
_rate_limiter = RateLimiter()


# =============================================================================
#  SAFE GROQ CALL
# =============================================================================

class TokenLimitError(Exception):
    """Raised when the prompt exceeds Groq's token limit (413)."""
    pass


def safe_groq_call(
    client: Groq,
    prompt: str,
    *,
    system_message: str = None,
    retries: int = None,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """Rate-limited Groq call with retry + exponential backoff.

    Args:
        client:         Groq client instance.
        prompt:         User prompt string.
        system_message: Optional system-role message for instruction following.
        retries:        Max retry attempts on transient errors.
        temperature:    LLM temperature (default from config).
        max_tokens:     Max output tokens (default from config).

    Returns:
        Cleaned response string (think tags stripped for qwen3).

    Raises:
        TokenLimitError: When the prompt exceeds the token limit (413).
    """
    retries = retries or config.GROQ_RETRY_ATTEMPTS
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens = max_tokens or config.LLM_MAX_TOKENS

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    _rate_limiter.wait()

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            # qwen3-32b wraps reasoning in <think>...</think> -- strip it
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Inter-call delay to stay well under quota
            if config.GROQ_INTER_CALL_DELAY > 0:
                time.sleep(config.GROQ_INTER_CALL_DELAY)

            return content

        except Exception as e:
            err = str(e).lower()
            if "413" in str(e) or "too large" in err or "token" in err and "limit" in err:
                # Token limit — caller should trim the prompt, not retry
                raise TokenLimitError(f"Prompt too large for model: {str(e)[:200]}")
            elif "rate_limit" in err or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"       [429] Rate limited -- waiting {wait}s (attempt {attempt+1}/{retries}) ...")
                time.sleep(wait)
                _rate_limiter.wait()
            elif "quota" in err or "daily" in err:
                print("       DAILY QUOTA REACHED. Cached results saved -- re-run tomorrow.")
                raise SystemExit(1)
            elif any(kw in err for kw in ["connection", "ssl", "timeout", "eof",
                                           "reset", "broken pipe", "network"]):
                wait = 10 * (attempt + 1)
                print(f"       [Network] Connection error -- retrying in {wait}s "
                      f"(attempt {attempt+1}/{retries}) ...")
                time.sleep(wait)
            else:
                if attempt == retries - 1:
                    raise
                wait = 10 * (attempt + 1)
                print(f"       [Error] {str(e)[:120]} -- retrying in {wait}s ...")
                time.sleep(wait)

    return ""


# =============================================================================
#  JSON PARSING
# =============================================================================

def parse_json_from_llm(response: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try stripping markdown fences first
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", response, re.DOTALL)
    if match:
        response = match.group(1)
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON array or object boundaries
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            s = response.find(start_char)
            e = response.rfind(end_char)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(response[s: e + 1])
                except json.JSONDecodeError:
                    continue
    return {}


def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> tags from response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

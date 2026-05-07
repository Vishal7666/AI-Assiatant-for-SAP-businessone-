"""
gemini_module.py — Gemini general chat module for Mahesh Industries
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
  Handles ALL general questions that are NOT about company database data.
  Examples: weather, news, jokes, math, general knowledge, current events,
            greetings, translations, recipes — anything non-business.

FEATURES:
  • Multiple API key rotation (comma-separated keys)
  • Smart exponential backoff on rate limits (2,4,8,16,32,64s)
  • Auto model downgrade if a model fails or is unavailable
  • Google Search tool for real-time info (weather, news, cricket scores)
  • Conversation history (last 10 exchanges)
  • Thread-safe with lock
  • Always returns a response — never crashes silently
─────────────────────────────────────────────────────────────────────────────
Install: pip install google-genai
─────────────────────────────────────────────────────────────────────────────
"""

import time
import threading
from google import genai
from google.genai import types


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ✅ FIXED: Correct Gemini model names (as of 2024-2025)
GEMINI_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# System prompt: Gemini is the GENERAL assistant — it handles everything
# that is NOT a database/business data question (those go to Groq+SQL)
SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable assistant for Mahesh Industries.

Your role: Answer ALL general questions clearly, accurately, and concisely.
This includes: weather, news, sports scores, jokes, math, general knowledge,
current events, greetings, recipes, translations, science, history, and anything else.

Rules:
- NEVER say you can't answer a general question.
- NEVER redirect the user to "check a website" if you can answer directly.
- IF user is asking for large information (only for large detailed information) redirect them to google gemini for that detailed information.
- Always try to give a direct answer first, then optionally suggest "For more details, you can check Google Gemini." if the topic is broad or complex.
- Use the Google Search tool for real-time info (weather, cricket scores, news).
- For math: calculate and give the exact answer.
- For jokes: be genuinely funny.
- Keep answers helpful, warm, and to the point.
- Use ₹ for Indian Rupee. Be aware this is an Indian business context.
- If someone greets you, respond warmly and offer to help."""

MAX_RETRIES     = 6
BASE_BACKOFF    = 2      # doubles each retry: 2, 4, 8, 16, 32, 64
MIN_REQUEST_GAP = 1.0    # minimum seconds between calls


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI MODULE
# ══════════════════════════════════════════════════════════════════════════════

class GeminiModule:
    """
    General chat handler powered by Gemini.
    Handles all non-DB questions. Returns (answer: str, sources: list).
    """

    def __init__(self, api_key: str):
        # Support comma-separated keys: "key1,key2,key3"
        self._api_keys   = [k.strip() for k in api_key.split(",") if k.strip()]
        self._key_index  = 0
        self.model_name  = None
        self.model_index = 0
        self._use_search = True
        self._lock       = threading.Lock()
        self._last_ts    = 0.0
        self._history: list = []   # [{user: str, model: str}]

        print("\n" + "=" * 60)
        print("🔍 Initialising Gemini Module (General Chat)")
        print(f"   API keys loaded: {len(self._api_keys)}")
        print("=" * 60)

        self._client = self._make_client()
        self._detect_model()

    # ── Client / key helpers ──────────────────────────────────────────────────

    def _make_client(self) -> genai.Client:
        return genai.Client(api_key=self._api_keys[self._key_index])

    def _rotate_key(self):
        if len(self._api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            self._client    = self._make_client()
            print(f"  🔄 [Gemini] Rotated to API key #{self._key_index + 1}")
        else:
            print("  ℹ️  [Gemini] Single key — waiting for quota reset...")

    def _rotate_model(self) -> bool:
        if self.model_index < len(GEMINI_MODELS) - 1:
            self.model_index += 1
            self.model_name   = GEMINI_MODELS[self.model_index]
            print(f"  ⬇️  [Gemini] Downgraded to: {self.model_name}")
            return True
        return False

    # ── Model detection ───────────────────────────────────────────────────────

    def _detect_model(self):
        """Try each model by sending a minimal ping. First one that works wins."""
        for i, name in enumerate(GEMINI_MODELS):
            try:
                resp = self._client.models.generate_content(
                    model=name,
                    contents=[types.Content(role="user", parts=[types.Part(text="Say OK")])],
                    config=types.GenerateContentConfig(
                        max_output_tokens=5,
                        temperature=0.0,
                    ),
                )
                # Try to read the response text — will raise if invalid
                text = ""
                try:
                    text = resp.text or ""
                except Exception:
                    # Some models return candidates instead of .text
                    try:
                        text = resp.candidates[0].content.parts[0].text or ""
                    except Exception:
                        text = "OK"  # assume success if no exception on generate

                self.model_name  = name
                self.model_index = i
                print(f"  ✅ [Gemini] Model : {name}")
                print(f"  ✅ [Gemini] Status: Ready")
                return

            except Exception as e:
                err = str(e)
                print(f"  ❌ [Gemini] {name}: {err[:120]}")

                # Hard stop on invalid key
                if any(x in err for x in ["API_KEY_INVALID", "API key not valid", "401", "UNAUTHENTICATED"]):
                    print("  ❌ Invalid API key → https://aistudio.google.com/app/apikey")
                    return

                # Rate limit on detection — just move to next model
                if any(x in err.lower() for x in ["429", "quota", "rate", "resource_exhausted"]):
                    time.sleep(2)
                    continue

        print("  ❌ [Gemini] No models available. Check API key and quota.")

    # ── Public chat ───────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> tuple:
        """
        Send a message, get (answer, sources).
        Thread-safe. Never raises — always returns a string.
        """
        if not self.model_name:
            return (
                "⚠️ Gemini is not configured. "
                "Check your API key at https://aistudio.google.com/app/apikey",
                [],
            )
        with self._lock:
            return self._send(user_message)

    # ── Internal retry engine ─────────────────────────────────────────────────

    def _send(self, user_message: str) -> tuple:
        for attempt in range(MAX_RETRIES):
            self._enforce_gap()
            try:
                resp = self._client.models.generate_content(
                    model   = self.model_name,
                    contents= self._build_contents(user_message),
                    config  = self._build_config(),
                )
                answer = self._extract_text(resp)

                if not answer:
                    print(f"  ⚠️  [Gemini] Empty response attempt {attempt+1} — retrying")
                    time.sleep(BASE_BACKOFF)
                    continue

                self._save_turn(user_message, answer)
                print(f"  ✅ [Gemini] OK — attempt {attempt+1}, model={self.model_name}")
                return answer, []

            except Exception as e:
                err     = str(e)
                err_low = err.lower()

                # 429 / quota / rate limit
                if (
                    "429" in err
                    or "quota" in err_low
                    or "rate_limit" in err_low
                    or "resource_exhausted" in err_low
                    or "too_many_requests" in err_low
                ):
                    wait = BASE_BACKOFF * (2 ** attempt)
                    print(f"  ⏳ [Gemini] Rate limit — waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    if attempt >= 2:
                        self._rotate_key()

                # 503 overloaded
                elif "503" in err or "overload" in err_low or "unavailable" in err_low:
                    wait = BASE_BACKOFF * (attempt + 1)
                    print(f"  ⏳ [Gemini] Overloaded — waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)

                # Search/tool error → disable search and retry immediately
                elif self._use_search and (
                    "tool" in err_low or "grounding" in err_low
                    or "function_call" in err_low
                    or "invalid_argument" in err_low
                    or "unsupported" in err_low
                    or "400" in err
                ):
                    print("  ⚠️  [Gemini] Search tool error — disabling, retrying")
                    self._use_search = False
                    # Don't count this as a full attempt
                    continue

                # History/content error → clear history and retry
                elif (
                    "turn" in err_low or "role" in err_low
                    or "history" in err_low
                ):
                    print("  ⚠️  [Gemini] Content/history error — clearing history, retrying")
                    self._history = []

                # Model not found → downgrade
                elif "not found" in err_low or ("404" in err and "model" in err_low):
                    if not self._rotate_model():
                        print("  ❌ [Gemini] No more models.")
                        break

                # Auth error → stop immediately
                elif (
                    "401" in err
                    or "api_key" in err_low
                    or "authentication" in err_low
                    or "API_KEY_INVALID" in err
                    or "unauthenticated" in err_low
                ):
                    print("  ❌ [Gemini] Auth error")
                    return (
                        "⚠️ Gemini API key error. "
                        "Get a key at https://aistudio.google.com/app/apikey",
                        [],
                    )

                # Unknown error
                else:
                    print(f"  ❌ [Gemini] Unknown error attempt {attempt+1}: {err[:120]}")
                    if attempt >= 2:
                        time.sleep(BASE_BACKOFF * 2)

        # All retries exhausted
        print(f"  ❌ [Gemini] All {MAX_RETRIES} attempts failed.")
        return (
            "I'm temporarily unable to respond — Gemini API quota may be exhausted. "
            "Please wait a moment and try again.",
            [],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enforce_gap(self):
        elapsed = time.time() - self._last_ts
        if elapsed < MIN_REQUEST_GAP:
            time.sleep(MIN_REQUEST_GAP - elapsed)
        self._last_ts = time.time()

    def _build_contents(self, user_message: str) -> list:
        """Build contents from history + new message (text parts only)."""
        contents = []
        for turn in self._history[-10:]:  # last 5 exchanges
            contents.append(types.Content(
                role="user",  parts=[types.Part(text=turn["user"])]
            ))
            contents.append(types.Content(
                role="model", parts=[types.Part(text=turn["model"])]
            ))
        contents.append(types.Content(
            role="user", parts=[types.Part(text=user_message)]
        ))
        return contents

    def _build_config(self) -> types.GenerateContentConfig:
        """
        ✅ FIXED: Build config with optional search tool.
        Creates base config first, then conditionally adds tools
        to avoid ArgumentError with older SDK versions.
        """
        cfg_kwargs = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.7,
            "max_output_tokens": 1024,
        }

        if self._use_search:
            try:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                cfg_kwargs["tools"] = [search_tool]
            except Exception as e:
                print(f"  ⚠️  [Gemini] Could not attach search tool: {e} — disabling")
                self._use_search = False

        return types.GenerateContentConfig(**cfg_kwargs)

    def _extract_text(self, resp) -> str:
        """Safely extract text from any Gemini response shape."""
        # Method 1: direct .text attribute
        try:
            if resp.text:
                return resp.text.strip()
        except Exception:
            pass

        # Method 2: iterate candidates
        try:
            for candidate in resp.candidates:
                parts = [
                    p.text for p in candidate.content.parts
                    if hasattr(p, "text") and p.text
                ]
                if parts:
                    return " ".join(parts).strip()
        except Exception:
            pass

        # Method 3: prompt_feedback check (blocked response)
        try:
            if resp.prompt_feedback:
                return f"Response blocked: {resp.prompt_feedback}"
        except Exception:
            pass

        return ""

    def _save_turn(self, user: str, model: str):
        self._history.append({"user": user, "model": model})
        if len(self._history) > 20:
            self._history = self._history[-20:]

    # ── Public utilities ──────────────────────────────────────────────────────

    def clear_history(self):
        self._history    = []
        self._use_search = True
        print("  [Gemini] History cleared.")

    def simple_ask(self, question: str) -> str:
        answer, _ = self.chat(question)
        return answer

    @property
    def status(self) -> dict:
        return {
            "model":      self.model_name,
            "search":     self._use_search,
            "api_keys":   len(self._api_keys),
            "active_key": self._key_index + 1,
            "history":    len(self._history),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    gemini = GeminiModule(api_key=API_KEY)
    print(f"\nStatus: {gemini.status}\n{'='*60}")
    tests = [
        "What is 25 * 48?",
        "Tell me a short joke",
        "What is the capital of Maharashtra?",
        "What is today's weather in Ahilyanagar?",
        "Who won the last IPL match?",
        "Hello, how are you?",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        ans, _ = gemini.chat(q)
        print(f"A: {ans[:400]}")
        print("-" * 60)
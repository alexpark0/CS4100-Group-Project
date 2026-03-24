"""
llmAPIs.py – LLM-powered plot description utilities.

Supports OpenAI, Groq, and local (Ollama) backends via the OpenAI-compatible API.
Configure your provider in a .env file (see .env.example).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider config – switch between openai / groq / local by setting ENV vars
# ---------------------------------------------------------------------------
#   PROVIDER        | LLM_BASE_URL                          | LLM_API_KEY
#   openai          | https://api.openai.com/v1  (default)  | your-openai-key
#   groq            | https://api.groq.com/openai/v1        | your-groq-key
#   local (ollama)  | http://localhost:11434/v1              | ollama  (dummy)
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

if not LLM_API_KEY:
    raise EnvironmentError(
        "LLM_API_KEY is not set. Add it to your .env file. "
        "See .env.example for reference."
    )

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chat(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Send a chat completion request and return the assistant's text."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 1.  Summarize / rewrite an OMDB plot into a consistent style
# ---------------------------------------------------------------------------

def rewrite_plot(plot: str, title: str = "", style: str = "concise and engaging") -> str:
    """
    Rewrite a raw OMDB plot summary into a consistent style.

    Args:
        plot:   The original plot text (usually from OMDB's 'Plot' field).
        title:  Optional movie title for context.
        style:  Desired tone / style descriptor.

    Returns:
        A rewritten plot string.
    """
    system = (
        "You are a film critic AI. Rewrite the provided movie plot summary "
        f"in a {style} style. Keep it to one paragraph (3-5 sentences). "
        "Do NOT add spoilers beyond what the original contains."
    )
    user = f"Movie: {title}\n\nOriginal plot:\n{plot}" if title else plot
    return _chat(system, user, temperature=0.6)


# ---------------------------------------------------------------------------
# 2.  Generate 10 keywords describing a movie plot
# ---------------------------------------------------------------------------

def generate_plot_keywords(plot: str, title: str = "") -> list[str]:
    """
    Generate 10 keywords that describe a movie plot.

    Args:
        plot:  The plot text (from OMDB's 'Plot' field or similar).
        title: Optional movie title for context.

    Returns:
        A list of exactly 10 keyword strings.
    """
    system = (
        "You are a film analyst AI. Given a movie plot, output exactly 10 "
        "single-word or short-phrase keywords that best describe the plot's "
        "themes, tone, setting, and key elements. "
        "Return ONLY the 10 keywords as a comma-separated list with no "
        "numbering, bullet points, or extra text."
    )
    user = f"Movie: {title}\n\nPlot:\n{plot}" if title else plot
    raw = _chat(system, user, temperature=0.5, max_tokens=128)
    keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
    return keywords[:10]


# ---------------------------------------------------------------------------
# 3.  Compare two movie plots and describe their "vibe" similarity
# ---------------------------------------------------------------------------

def compare_vibes(plot_a: str, title_a: str, plot_b: str, title_b: str) -> str:
    """
    Compare two movies by their plots and explain the shared 'vibe.'

    Args:
        plot_a / title_a:  Plot text and title for movie A.
        plot_b / title_b:  Plot text and title for movie B.

    Returns:
        A natural-language comparison highlighting thematic and tonal overlap.
    """
    system = (
        "You are a movie recommendation assistant. Compare the two movies below "
        "and describe their shared 'vibe' — themes, tone, mood, pacing, visual "
        "style, emotional arc. Be specific and helpful so a viewer can decide "
        "if they'd enjoy one based on liking the other. Keep it to one short paragraph."
    )
    user = (
        f"Movie A – {title_a}:\n{plot_a}\n\n"
        f"Movie B – {title_b}:\n{plot_b}"
    )
    return _chat(system, user, temperature=0.7)


# ---------------------------------------------------------------------------
# Example usage (requires a valid .env)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from omdbAPIs import get_movie_by_title

    # --- Rewrite ---
    movie = get_movie_by_title("Inception")
    print("=== Rewritten Plot ===")
    print(rewrite_plot(movie["Plot"], movie["Title"]))
    print()

    # --- Generate keywords ---
    print("=== Plot Keywords ===")
    print(generate_plot_keywords(movie["Plot"], movie["Title"]))
    print()

    # --- Vibe comparison ---
    movie2 = get_movie_by_title("Interstellar")
    print("=== Vibe Comparison ===")
    print(compare_vibes(movie["Plot"], movie["Title"], movie2["Plot"], movie2["Title"]))

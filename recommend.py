"""
recommend.py  –  CS4100 Movie Recommendation CLI
=================================================

Run with:
    python recommend.py

Flow
----
1. Enter 1+ movies you already like (fetched from OMDB + LLM keywords).
2. Optionally lock a genre constraint (e.g. "Sci-Fi").
3. The system scores every candidate against your profile using the
   heuristic scorer + neural network and presents the best match.
4. Reply y / n:
     y → great, enjoy!  (NN trains on positive feedback)
     n → try again      (NN trains on negative feedback, next-best shown)
     q → quit
"""

import sys
from main import build_movie_row
from nn_model import MovieRecommender, compute_heuristic_score

# ---------------------------------------------------------------------------
# Candidate pool  –  a curated set of well-known movies spanning all genres.
# OMDB title-search is keyword-only (not genre-searchable), so we maintain a
# fixed pool and fetch full profiles on demand.
# ---------------------------------------------------------------------------
CANDIDATE_TITLES = [
    # Action / Adventure
    "Mad Max: Fury Road", "John Wick", "The Dark Knight", "Die Hard",
    "Mission: Impossible – Fallout", "Top Gun: Maverick", "Gladiator",
    "The Avengers", "Black Panther", "Spider-Man: Into the Spider-Verse",
    # Sci-Fi
    "Interstellar", "The Matrix", "Blade Runner 2049", "Arrival",
    "Ex Machina", "2001: A Space Odyssey", "Gravity", "Dune",
    # Thriller / Mystery
    "Se7en", "Gone Girl", "Parasite", "Knives Out", "Prisoners",
    "Zodiac", "Memento", "Oldboy", "The Silence of the Lambs",
    # Drama
    "The Shawshank Redemption", "Schindler's List", "Forrest Gump",
    "The Godfather", "12 Angry Men", "Whiplash", "A Beautiful Mind",
    "Good Will Hunting", "The Social Network", "Marriage Story",
    # Comedy
    "The Grand Budapest Hotel", "Superbad", "Game Night",
    "Knives Out", "About Time", "The Princess Bride",
    # Romance
    "Eternal Sunshine of the Spotless Mind", "La La Land",
    "Before Sunrise", "Titanic", "Crazy Rich Asians",
    # Horror
    "Get Out", "Hereditary", "A Quiet Place", "The Conjuring",
    "It Follows", "Midsommar", "The Shining",
    # Animation
    "Spirited Away", "WALL-E", "Inside Out", "Coco",
    "Princess Mononoke", "Your Name",
    # Crime
    "Pulp Fiction", "No Country for Old Men", "Goodfellas",
    "The Departed", "Heat", "Fargo",
    # War / History
    "Dunkirk", "1917", "Saving Private Ryan", "Apocalypse Now",
    "Full Metal Jacket",
]


def _fetch_profile(title: str) -> dict | None:
    """Return a normalized movie profile dict, or None on error."""
    try:
        return build_movie_row(title)
    except Exception as e:
        print(f"  [warning] Could not fetch '{title}': {e}")
        return None


def _display_recommendation(movie: dict, score_breakdown: dict, nn_prob: float):
    actors = ", ".join(
        a for a in [movie.get("actor_1"), movie.get("actor_2"), movie.get("actor_3")]
        if a and a != "N/A"
    )
    keywords = ", ".join(movie.get("plot_keywords", [])[:6]) or "N/A"

    print()
    print("=" * 56)
    print(f"  Recommendation: {movie['title']}")
    print("=" * 56)
    print(f"  Genre:    {movie.get('genre', 'N/A')}")
    print(f"  Rating:   {movie.get('rating', 'N/A')}/10  "
          f"({movie.get('runtime_min', '?')} min)")
    print(f"  Cast:     {actors or 'N/A'}")
    print(f"  Keywords: {keywords}")
    print(f"  Score:    {score_breakdown['total']}  "
          f"(actor:{score_breakdown['actor']}  "
          f"genre:{score_breakdown['genre']}  "
          f"rating:{score_breakdown['rating']}  "
          f"keyword:{score_breakdown['keyword']})")
    print(f"  NN confidence: {nn_prob:.0%}")
    print("-" * 56)


def run():
    rec = MovieRecommender()

    # ------------------------------------------------------------------
    # Step 1: collect liked movies
    # ------------------------------------------------------------------
    print()
    print("=" * 56)
    print("  CS4100 Movie Recommender")
    print("=" * 56)
    print("Enter movies you already like (blank line when done):")

    liked: list[dict] = []
    while True:
        raw = input("  > ").strip()
        if not raw:
            if not liked:
                print("  Please enter at least one movie.")
                continue
            break
        profile = _fetch_profile(raw)
        if profile:
            liked.append(profile)
            print(f"  Added: {profile['title']}  "
                  f"[{profile.get('genre', '')}  {profile.get('rating', '')}]")

    # ------------------------------------------------------------------
    # Step 2: optional fixed genre constraint
    # ------------------------------------------------------------------
    print()
    raw_genre = input(
        "Lock a genre? (e.g. Sci-Fi, Horror — press Enter to skip): "
    ).strip()
    fixed_genres = [g.strip() for g in raw_genre.split(",") if g.strip()]
    if fixed_genres:
        print(f"  Genre filter: {', '.join(fixed_genres)}")

    # ------------------------------------------------------------------
    # Step 3: fetch + score all candidates (lazy: on first use)
    # ------------------------------------------------------------------
    liked_titles_lower = {m["title"].lower() for m in liked}
    rejected_titles_lower: set[str] = set()

    print()
    print("Fetching candidate movies …  (this may take a moment)")

    candidates: list[dict] = []
    for title in CANDIDATE_TITLES:
        if title.lower() in liked_titles_lower:
            continue
        p = _fetch_profile(title)
        if p:
            candidates.append(p)

    if not candidates:
        print("No candidates available. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 4: recommendation loop
    # ------------------------------------------------------------------
    while True:
        # Score remaining candidates (skip rejected)
        scored = []
        for cand in candidates:
            if cand["title"].lower() in rejected_titles_lower:
                continue
            sb = compute_heuristic_score(cand, liked, fixed_genres or None)
            if sb["total"] == -1:          # failed fixed-genre filter
                continue
            nn_prob = rec.score(cand, sb)
            scored.append((cand, sb, nn_prob))

        if not scored:
            print("\nNo more candidates match your preferences. "
                  "Try relaxing the genre filter or adding more liked movies.")
            break

        # Sort by NN probability descending
        scored.sort(key=lambda x: x[2], reverse=True)
        best_cand, best_sb, best_prob = scored[0]

        _display_recommendation(best_cand, best_sb, best_prob)

        reply = input("  Like this recommendation? (y / n / q): ").strip().lower()

        if reply == "q":
            print("\nGoodbye!")
            break

        elif reply == "y":
            rec.update(best_cand, best_sb, liked=True)
            print(f"\nEnjoy '{best_cand['title']}'!")
            break

        elif reply == "n":
            rec.update(best_cand, best_sb, liked=False)
            rejected_titles_lower.add(best_cand["title"].lower())
            print("Got it — finding another recommendation …")

        else:
            print("  Please enter y, n, or q.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)

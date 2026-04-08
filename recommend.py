#!/usr/bin/env python3
"""
recommend.py  --  Interactive Movie Recommendation CLI
=====================================================
CS4100 Group Project  |  Park, Rifai, Shi
"""

import sys
from main import build_movie_row
from omdbAPIs import search_movies
from nn_model import MovieRecommender, compute_heuristic_score, ALL_GENRES

# ── Display helpers ──────────────────────────────────────────────────

WIDTH = 52

def banner():
    print()
    print("=" * WIDTH)
    print("   CS4100 Movie Recommender")
    print("   Park  |  Rifai  |  Shi")
    print("=" * WIDTH)
    print()
    print("  How it works:")
    print("  1. Enter movies you already like")
    print("  2. Optionally lock to a genre")
    print("  3. Get personalized recommendations")
    print("  4. Give feedback so the model learns")
    print()


def star_bar(rating, max_rating=10):
    filled = int(round(rating))
    return "*" * filled + "-" * (max_rating - filled)


def print_recommendation(cand, score, prob):
    genres = ", ".join(cand["genres"]) if cand["genres"] else "N/A"
    cast = [cand.get(f"actor_{i}", "") for i in [1, 2, 3]]
    cast = [a for a in cast if a and a != "N/A"]

    print()
    print("+" + "-" * WIDTH + "+")
    print(f"|  RECOMMENDATION: {cand['title']:<{WIDTH - 19}}|")
    print("+" + "-" * WIDTH + "+")
    print(f"|  Genres:   {genres:<{WIDTH - 12}}|")
    print(f"|  Rating:   {cand['rating']}/10  {star_bar(cand['rating']):<{WIDTH - 24}}|")
    if cand["runtime_min"]:
        print(f"|  Runtime:  {cand['runtime_min']} min{'':<{WIDTH - 17 - len(str(cand['runtime_min']))}}|")
    if cast:
        cast_str = ", ".join(cast)
        print(f"|  Cast:     {cast_str:<{WIDTH - 12}}|")
    print("+" + "-" * WIDTH + "+")
    print(f"|  Match Breakdown{'':<{WIDTH - 17}}|")
    print(f"|    Actors:   {score['actor']:<5} Keywords: {score['keyword']:<{WIDTH - 29}}|")
    print(f"|    Genres:   {score['genre']:<5} Rating:   {score['rating']:<{WIDTH - 29}}|")
    print(f"|    Total:    {score['total']:<5} NN Conf:  {prob:.1%}{'':<{WIDTH - 32}}|")
    print("+" + "-" * WIDTH + "+")


def print_summary(liked_titles, shown_count, accepted):
    print()
    print("=" * WIDTH)
    print("  Session Summary")
    print("-" * WIDTH)
    print(f"  Liked movies:       {', '.join(liked_titles)}")
    print(f"  Recommendations:    {shown_count} shown")
    if accepted:
        print(f"  Accepted:           {accepted}")
    print("=" * WIDTH)


# ── Input helpers ────────────────────────────────────────────────────

def prompt_choice(prompt_text, valid):
    """Keep prompting until input is in the valid set."""
    while True:
        choice = input(prompt_text).strip().lower()
        if choice in valid:
            return choice
        print(f"  Please enter one of: {', '.join(valid)}")


def get_liked_movies():
    """Prompt user to enter movie titles they like."""
    print("--- Add Movies You Like ---")
    print("  Enter movie titles one at a time.")
    print("  Press Enter on an empty line when done.\n")
    liked = []
    while True:
        title = input("  Movie title: ").strip()
        if not title:
            if not liked:
                print("  You need at least one movie. Try again.")
                continue
            break
        print(f"    Fetching \"{title}\"...", end=" ", flush=True)
        try:
            row = build_movie_row(title)
            liked.append(row)
            print(f"OK  [{row['title']}]")
        except ValueError as e:
            print(f"NOT FOUND  ({e})")
        except Exception as e:
            print(f"ERROR  ({e})")
    return liked


def get_fixed_genres():
    """Optionally lock recommendations to a genre."""
    print("\n--- Optional Genre Constraint ---")
    print(f"  Available: {', '.join(ALL_GENRES[:26])}")
    genre = input("  Lock to a genre (Enter to skip): ").strip()
    if not genre:
        return []
    # Case-insensitive match
    for g in ALL_GENRES:
        if g.lower() == genre.lower():
            print(f"    Locked to: {g}")
            return [g]
    print(f"    \"{genre}\" not recognized, skipping constraint.")
    return []


# ── Candidate search ─────────────────────────────────────────────────

def find_candidates(liked_rows, exclude_titles):
    """Search OMDB for candidate movies based on liked movie attributes."""
    seen = set(t.lower() for t in exclude_titles)
    candidates = []

    # Collect search terms from liked movies
    genres = set()
    actors = set()
    for row in liked_rows:
        for g in row.get("genres", []):
            genres.add(g)
        for i in [1, 2, 3]:
            a = row.get(f"actor_{i}", "").strip()
            if a and a != "N/A":
                actors.add(a)

    search_terms = list(genres)[:2] + list(actors)[:2]
    total = len(search_terms)

    for idx, term in enumerate(search_terms, 1):
        print(f"  Searching [{idx}/{total}] \"{term}\"...", end=" ", flush=True)
        try:
            results = search_movies(term)
        except Exception:
            print("failed")
            continue

        added = 0
        for r in results:
            if added >= 3:  # limit per term to save API quota
                break
            title = r.get("Title", "")
            if not title or title.lower() in seen:
                continue
            try:
                row = build_movie_row(title)
                candidates.append(row)
                seen.add(title.lower())
                added += 1
            except Exception:
                pass
        print(f"{added} found")

    return candidates


# ── Main loop ─────────────────────────────────────────────────────────

def run():
    banner()
    rec = MovieRecommender()

    liked = get_liked_movies()
    liked_titles = [r["title"] for r in liked]
    print(f"\n  Profile built from {len(liked)} movie(s).")

    fixed_genres = get_fixed_genres()

    rejected = set(t.lower() for t in liked_titles)
    shown_count = 0
    accepted = None

    while True:
        print("\n--- Finding Recommendations ---")
        candidates = find_candidates(liked, rejected)

        if not candidates:
            print("\n  No candidates found.")
            choice = prompt_choice(
                "  [a] Add more movies  [q] Quit\n  > ", {"a", "q"}
            )
            if choice == "a":
                extra = get_liked_movies()
                liked.extend(extra)
                liked_titles.extend(r["title"] for r in extra)
                continue
            break

        # Score and rank
        scored = []
        for cand in candidates:
            sb = compute_heuristic_score(cand, liked, fixed_genres or None)
            if sb["total"] == -1:
                continue
            prob = rec.score(cand, sb)
            scored.append((cand, sb, prob))

        if not scored:
            print("  No candidates matched your genre constraint.")
            choice = prompt_choice(
                "  [c] Change genre  [a] Add movies  [q] Quit\n  > ",
                {"c", "a", "q"},
            )
            if choice == "c":
                fixed_genres = get_fixed_genres()
                continue
            elif choice == "a":
                extra = get_liked_movies()
                liked.extend(extra)
                liked_titles.extend(r["title"] for r in extra)
                continue
            break

        scored.sort(key=lambda x: x[2], reverse=True)
        best_cand, best_sb, best_prob = scored[0]
        shown_count += 1

        print_recommendation(best_cand, best_sb, best_prob)

        choice = prompt_choice(
            "\n  [y] Yes, I'd watch this  [n] No, next  [q] Quit\n  > ",
            {"y", "n", "q", "yes", "no", "quit"},
        )

        if choice in ("q", "quit"):
            break
        elif choice in ("y", "yes"):
            rec.update(best_cand, best_sb, liked=True)
            accepted = best_cand["title"]
            print(f"\n  Great choice! Enjoy \"{accepted}\".")
            break
        else:
            rec.update(best_cand, best_sb, liked=False)
            rejected.add(best_cand["title"].lower())
            print("  Noted. Finding another...")

    print_summary(liked_titles, shown_count, accepted)
    print("  Thanks for using the recommender!\n")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!")
        sys.exit(0)

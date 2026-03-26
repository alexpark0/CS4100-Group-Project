from nn_model import MovieRecommender, compute_heuristic_score

# Create the recommender (loads saved weights if nn_weights.pt exists)
rec = MovieRecommender()

# ── Define movies the user already likes ──────────────────────────
liked = [{
    "title": "Inception",
    "genres": ["Action", "Adventure", "Sci-Fi"],
    "rating": 8.8,
    "runtime_min": 148,
    "actor_1": "Leonardo DiCaprio",
    "actor_2": "Joseph Gordon-Levitt",
    "actor_3": "Elliot Page",
    "plot_keywords": ["dreams", "heist", "espionage", "deception", "reality"],
}]

# ── Define a candidate movie you want to evaluate ─────────────────
candidate = {
    "title": "Interstellar",
    "genres": ["Adventure", "Drama", "Sci-Fi"],
    "rating": 8.7,
    "runtime_min": 169,
    "actor_1": "Matthew McConaughey",
    "actor_2": "Anne Hathaway",
    "actor_3": "Jessica Chastain",
    "plot_keywords": ["space", "wormhole", "survival", "family", "sacrifice"],
}

# ── Step A: compute the heuristic score breakdown ─────────────────
score = compute_heuristic_score(candidate, liked)
print(score)
# → {'actor': 0, 'genre': 4, 'rating': 8, 'keyword': 0, 'total': 12}

# ── Step B: ask the NN "should I recommend this?" ─────────────────
prob = rec.score(candidate, score)
print(f"P(recommend) = {prob:.3f}")
# → P(recommend) = 0.552   (higher = more likely to recommend)

# ── Step C: record user feedback so the NN learns ─────────────────
rec.update(candidate, score, liked=True)   # user said YES → score goes up
rec.update(candidate, score, liked=False)  # user said NO  → score goes down

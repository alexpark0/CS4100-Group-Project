#!/usr/bin/env python3
"""
pretrain.py  --  Generate synthetic training data and pre-train the NN
======================================================================
No API calls needed. Uses hardcoded movie dicts to create
(feature_vector, label) pairs from simulated user taste profiles.

Run once:  python pretrain.py
Output:    nn_weights.pt
"""

from nn_model import (
    MovieRecommender, compute_heuristic_score, build_feature_vector,
)
import torch

# ── Hardcoded movie catalogue (no API calls) ─────────────────────────

MOVIES = [
    # Action / Sci-Fi
    {"title": "Inception", "genres": ["Action", "Adventure", "Sci-Fi"], "rating": 8.8, "runtime_min": 148,
     "actor_1": "Leonardo DiCaprio", "actor_2": "Joseph Gordon-Levitt", "actor_3": "Elliot Page",
     "plot_keywords": ["dreams", "heist", "subconscious", "espionage", "deception", "reality", "mind-bending", "infiltration", "betrayal", "redemption"]},
    {"title": "The Matrix", "genres": ["Action", "Sci-Fi"], "rating": 8.7, "runtime_min": 136,
     "actor_1": "Keanu Reeves", "actor_2": "Laurence Fishburne", "actor_3": "Carrie-Anne Moss",
     "plot_keywords": ["simulation", "rebellion", "AI", "reality", "hacking", "prophecy", "martial-arts", "dystopia", "awakening", "freedom"]},
    {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"], "rating": 9.0, "runtime_min": 152,
     "actor_1": "Christian Bale", "actor_2": "Heath Ledger", "actor_3": "Aaron Eckhart",
     "plot_keywords": ["gotham", "vigilante", "chaos", "heroism", "morality", "duality", "crime", "corruption", "psychology", "batman"]},
    {"title": "Interstellar", "genres": ["Adventure", "Drama", "Sci-Fi"], "rating": 8.7, "runtime_min": 169,
     "actor_1": "Matthew McConaughey", "actor_2": "Anne Hathaway", "actor_3": "Jessica Chastain",
     "plot_keywords": ["space", "wormhole", "survival", "family", "sacrifice", "astrophysics", "hope", "dystopian", "future", "teamwork"]},
    {"title": "Mad Max: Fury Road", "genres": ["Action", "Adventure", "Sci-Fi"], "rating": 8.1, "runtime_min": 120,
     "actor_1": "Tom Hardy", "actor_2": "Charlize Theron", "actor_3": "Nicholas Hoult",
     "plot_keywords": ["wasteland", "pursuit", "freedom", "rebellion", "survival", "desert", "vehicles", "tyranny", "redemption", "courage"]},
    {"title": "Gladiator", "genres": ["Action", "Adventure", "Drama"], "rating": 8.5, "runtime_min": 155,
     "actor_1": "Russell Crowe", "actor_2": "Joaquin Phoenix", "actor_3": "Connie Nielsen",
     "plot_keywords": ["revenge", "honor", "slavery", "arena", "empire", "betrayal", "courage", "glory", "death", "freedom"]},

    # Drama
    {"title": "The Shawshank Redemption", "genres": ["Drama"], "rating": 9.3, "runtime_min": 142,
     "actor_1": "Tim Robbins", "actor_2": "Morgan Freeman", "actor_3": "Bob Gunton",
     "plot_keywords": ["prison", "hope", "friendship", "escape", "injustice", "patience", "redemption", "corruption", "perseverance", "freedom"]},
    {"title": "Forrest Gump", "genres": ["Drama", "Romance"], "rating": 8.8, "runtime_min": 142,
     "actor_1": "Tom Hanks", "actor_2": "Robin Wright", "actor_3": "Gary Sinise",
     "plot_keywords": ["destiny", "love", "war", "history", "running", "innocence", "friendship", "comedy", "life", "perseverance"]},
    {"title": "The Godfather", "genres": ["Crime", "Drama"], "rating": 9.2, "runtime_min": 175,
     "actor_1": "Marlon Brando", "actor_2": "Al Pacino", "actor_3": "James Caan",
     "plot_keywords": ["mafia", "family", "power", "loyalty", "violence", "crime", "honor", "betrayal", "corruption", "legacy"]},
    {"title": "Schindler's List", "genres": ["Biography", "Drama", "History"], "rating": 9.0, "runtime_min": 195,
     "actor_1": "Liam Neeson", "actor_2": "Ralph Fiennes", "actor_3": "Ben Kingsley",
     "plot_keywords": ["holocaust", "salvation", "war", "humanity", "courage", "sacrifice", "tragedy", "morality", "survival", "compassion"]},
    {"title": "Good Will Hunting", "genres": ["Drama", "Romance"], "rating": 8.3, "runtime_min": 126,
     "actor_1": "Robin Williams", "actor_2": "Matt Damon", "actor_3": "Ben Affleck",
     "plot_keywords": ["genius", "therapy", "friendship", "love", "identity", "class", "potential", "trauma", "growth", "mentorship"]},
    {"title": "Fight Club", "genres": ["Drama"], "rating": 8.8, "runtime_min": 139,
     "actor_1": "Brad Pitt", "actor_2": "Edward Norton", "actor_3": "Meat Loaf",
     "plot_keywords": ["identity", "anarchy", "consumerism", "violence", "rebellion", "masculinity", "insomnia", "chaos", "twist", "underground"]},

    # Comedy
    {"title": "Superbad", "genres": ["Comedy"], "rating": 7.6, "runtime_min": 113,
     "actor_1": "Jonah Hill", "actor_2": "Michael Cera", "actor_3": "Christopher Mintz-Plasse",
     "plot_keywords": ["high-school", "party", "friendship", "coming-of-age", "awkward", "alcohol", "humor", "prom", "adolescence", "misadventure"]},
    {"title": "The Hangover", "genres": ["Comedy"], "rating": 7.7, "runtime_min": 100,
     "actor_1": "Bradley Cooper", "actor_2": "Ed Helms", "actor_3": "Zach Galifianakis",
     "plot_keywords": ["vegas", "bachelor-party", "amnesia", "chaos", "friendship", "wild", "humor", "mystery", "misadventure", "wedding"]},
    {"title": "Step Brothers", "genres": ["Comedy"], "rating": 6.9, "runtime_min": 98,
     "actor_1": "Will Ferrell", "actor_2": "John C. Reilly", "actor_3": "Mary Steenburgen",
     "plot_keywords": ["rivalry", "family", "immaturity", "humor", "bonding", "absurd", "slapstick", "jobs", "friendship", "childish"]},
    {"title": "The Grand Budapest Hotel", "genres": ["Adventure", "Comedy", "Crime"], "rating": 8.1, "runtime_min": 99,
     "actor_1": "Ralph Fiennes", "actor_2": "F. Murray Abraham", "actor_3": "Mathieu Amalric",
     "plot_keywords": ["hotel", "mystery", "adventure", "friendship", "whimsy", "war", "art", "nostalgia", "elegance", "heist"]},

    # Horror / Thriller
    {"title": "The Silence of the Lambs", "genres": ["Crime", "Drama", "Thriller"], "rating": 8.6, "runtime_min": 118,
     "actor_1": "Jodie Foster", "actor_2": "Anthony Hopkins", "actor_3": "Lawrence A. Bonney",
     "plot_keywords": ["serial-killer", "FBI", "psychology", "manipulation", "suspense", "fear", "intelligence", "cannibal", "investigation", "tension"]},
    {"title": "Get Out", "genres": ["Horror", "Mystery", "Thriller"], "rating": 7.7, "runtime_min": 104,
     "actor_1": "Daniel Kaluuya", "actor_2": "Allison Williams", "actor_3": "Bradley Whitford",
     "plot_keywords": ["racism", "horror", "suspense", "social-commentary", "manipulation", "hypnosis", "escape", "paranoia", "identity", "suburban"]},
    {"title": "A Quiet Place", "genres": ["Drama", "Horror", "Sci-Fi"], "rating": 7.5, "runtime_min": 90,
     "actor_1": "Emily Blunt", "actor_2": "John Krasinski", "actor_3": "Millicent Simmonds",
     "plot_keywords": ["silence", "survival", "family", "creatures", "fear", "tension", "sacrifice", "isolation", "post-apocalyptic", "sound"]},
    {"title": "The Shining", "genres": ["Drama", "Horror"], "rating": 8.4, "runtime_min": 146,
     "actor_1": "Jack Nicholson", "actor_2": "Shelley Duvall", "actor_3": "Danny Lloyd",
     "plot_keywords": ["hotel", "madness", "isolation", "supernatural", "family", "terror", "winter", "ghosts", "psychic", "violence"]},

    # Animation / Family
    {"title": "Toy Story", "genres": ["Animation", "Adventure", "Comedy"], "rating": 8.3, "runtime_min": 81,
     "actor_1": "Tom Hanks", "actor_2": "Tim Allen", "actor_3": "Don Rickles",
     "plot_keywords": ["toys", "friendship", "adventure", "jealousy", "loyalty", "imagination", "childhood", "courage", "belonging", "fun"]},
    {"title": "Finding Nemo", "genres": ["Animation", "Adventure", "Comedy"], "rating": 8.2, "runtime_min": 100,
     "actor_1": "Albert Brooks", "actor_2": "Ellen DeGeneres", "actor_3": "Alexander Gould",
     "plot_keywords": ["ocean", "fatherhood", "journey", "friendship", "courage", "loss", "adventure", "fish", "family", "perseverance"]},
    {"title": "Up", "genres": ["Animation", "Adventure", "Comedy"], "rating": 8.3, "runtime_min": 96,
     "actor_1": "Edward Asner", "actor_2": "Jordan Nagai", "actor_3": "John Ratzenberger",
     "plot_keywords": ["adventure", "grief", "friendship", "balloons", "dream", "exploration", "aging", "courage", "loyalty", "wilderness"]},
    {"title": "Spider-Man: Into the Spider-Verse", "genres": ["Animation", "Action", "Adventure"], "rating": 8.4, "runtime_min": 117,
     "actor_1": "Shameik Moore", "actor_2": "Jake Johnson", "actor_3": "Hailee Steinfeld",
     "plot_keywords": ["multiverse", "hero", "identity", "courage", "teamwork", "animation", "coming-of-age", "responsibility", "sacrifice", "belonging"]},

    # Romance / Mixed
    {"title": "Titanic", "genres": ["Drama", "Romance"], "rating": 7.9, "runtime_min": 194,
     "actor_1": "Leonardo DiCaprio", "actor_2": "Kate Winslet", "actor_3": "Billy Zane",
     "plot_keywords": ["love", "ship", "disaster", "class", "sacrifice", "ocean", "tragedy", "romance", "survival", "loss"]},
    {"title": "La La Land", "genres": ["Comedy", "Drama", "Music"], "rating": 8.0, "runtime_min": 128,
     "actor_1": "Ryan Gosling", "actor_2": "Emma Stone", "actor_3": "Rosemarie DeWitt",
     "plot_keywords": ["music", "dreams", "love", "ambition", "sacrifice", "hollywood", "jazz", "romance", "nostalgia", "bittersweet"]},

    # Classics / Misc
    {"title": "Pulp Fiction", "genres": ["Crime", "Drama"], "rating": 8.9, "runtime_min": 154,
     "actor_1": "John Travolta", "actor_2": "Uma Thurman", "actor_3": "Samuel L. Jackson",
     "plot_keywords": ["crime", "nonlinear", "violence", "dialogue", "redemption", "gangsters", "dark-humor", "loyalty", "fate", "underworld"]},
    {"title": "Goodfellas", "genres": ["Biography", "Crime", "Drama"], "rating": 8.7, "runtime_min": 146,
     "actor_1": "Robert De Niro", "actor_2": "Ray Liotta", "actor_3": "Joe Pesci",
     "plot_keywords": ["mafia", "crime", "rise-and-fall", "loyalty", "betrayal", "violence", "greed", "ambition", "gangsters", "power"]},
    {"title": "The Prestige", "genres": ["Drama", "Mystery", "Sci-Fi"], "rating": 8.5, "runtime_min": 130,
     "actor_1": "Christian Bale", "actor_2": "Hugh Jackman", "actor_3": "Scarlett Johansson",
     "plot_keywords": ["magic", "rivalry", "obsession", "deception", "sacrifice", "twist", "ambition", "mystery", "duality", "revenge"]},
    {"title": "Django Unchained", "genres": ["Drama", "Western"], "rating": 8.5, "runtime_min": 165,
     "actor_1": "Jamie Foxx", "actor_2": "Christoph Waltz", "actor_3": "Leonardo DiCaprio",
     "plot_keywords": ["slavery", "revenge", "freedom", "western", "violence", "justice", "bounty-hunter", "love", "courage", "liberation"]},
]

# ── Taste profiles (sets of liked-movie indices) ─────────────────────

PROFILES = {
    "action_scifi": [0, 1, 4],        # Inception, Matrix, Mad Max
    "drama_romance": [7, 10, 24],      # Forrest Gump, Good Will Hunting, Titanic
    "comedy":        [12, 13, 14],      # Superbad, Hangover, Step Brothers
    "horror_thriller": [16, 17, 18],    # Silence of the Lambs, Get Out, A Quiet Place
    "animation":     [20, 21, 22],      # Toy Story, Finding Nemo, Up
    "crime_classic":  [8, 26, 27],      # Godfather, Pulp Fiction, Goodfellas
}


def generate_training_data():
    """Build (feature_vector, label) pairs from synthetic taste profiles."""
    xs, ys = [], []

    for profile_name, liked_indices in PROFILES.items():
        liked_rows = [MOVIES[i] for i in liked_indices]

        # Score every movie against this profile
        scores = []
        for cand in MOVIES:
            sb = compute_heuristic_score(cand, liked_rows)
            scores.append((cand, sb))

        # Use median total as threshold for labeling
        totals = [sb["total"] for _, sb in scores]
        totals.sort()
        median = totals[len(totals) // 2]

        for cand, sb in scores:
            fv = build_feature_vector(cand, sb)
            label = 1.0 if sb["total"] >= median else 0.0
            xs.append(fv)
            ys.append(label)

    return torch.stack(xs), torch.tensor(ys)


def main():
    print("Generating synthetic training data...")
    X, Y = generate_training_data()
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive: {int(Y.sum())}, Negative: {int(len(Y) - Y.sum())}")

    print("\nTraining NN...")
    rec = MovieRecommender()
    rec.model.train()

    for epoch in range(1, 201):
        rec.optimizer.zero_grad()
        loss = rec.criterion(rec.model(X), Y)
        loss.backward()
        rec.optimizer.step()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>3}/200  loss={loss.item():.4f}")

    rec.save()
    print(f"\nWeights saved to nn_weights.pt")

    # Quick sanity check
    rec.model.eval()
    with torch.no_grad():
        preds = rec.model(X)
        correct = ((preds > 0.5).float() == Y).sum().item()
        print(f"Training accuracy: {correct}/{len(Y)} ({100*correct/len(Y):.1f}%)")


if __name__ == "__main__":
    main()

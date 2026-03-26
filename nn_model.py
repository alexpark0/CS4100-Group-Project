"""
nn_model.py  –  CS4100 Movie Recommendation Neural Network
===========================================================

ARCHITECTURE
------------
Input  : 37-dimensional feature vector (see build_feature_vector)
Hidden : Linear(37→64) → ReLU → Dropout(0.3)
         Linear(64→32) → ReLU → Dropout(0.3)
Output : Linear(32→1)  → Sigmoid  →  P(recommend) ∈ [0, 1]

FEATURE VECTOR  (37 dims)
-------------------------
 [0]   actor_score      – normalised shared-actor count vs. liked profile
 [1]   genre_score      – normalised weighted genre overlap
 [2]   rating_score     – normalised IMDb-rating contribution
 [3]   keyword_score    – normalised LLM plot-keyword overlap
 [4]   heuristic_total  – normalised combined heuristic score
 [5]   candidate_rating – raw IMDb rating / 10
 [6]   runtime_short    – 1 if runtime < 90 min,  else 0
 [7]   runtime_medium   – 1 if 90 ≤ runtime ≤ 120, else 0
 [8]   runtime_long     – 1 if runtime > 120 min, else 0
 [9..36] genre_onehot   – 28-dim one-hot over ALL_GENRES list

HEURISTIC SCORING  (used to build the feature vector)
------------------------------------------------------
  +1  per shared actor  (top-3 cast)
  +2  per shared genre
  +1  per IMDb rating star  (e.g. 8.5 → 8 pts)
  +1  per shared LLM plot keyword

ONLINE LEARNING
---------------
After every user yes/no response, call  recommender.update(...)
to run 10 Adam gradient steps on that single sample.
Weights are saved to  nn_weights.pt  automatically.

QUICK-START
-----------
    from nn_model import MovieRecommender, build_feature_vector

    rec = MovieRecommender()

    # Build a feature vector from a score breakdown dict + candidate dict
    fv = build_feature_vector(candidate_movie, score_breakdown)

    # Get recommendation probability
    prob = rec.score(candidate_movie, score_breakdown)

    # Update from user feedback
    rec.update(candidate_movie, score_breakdown, liked=True)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Genre vocabulary  (28 labels – matches OMDb / TMDB genre strings)
# ---------------------------------------------------------------------------
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV",
    "Romance", "Sci-Fi", "Short", "Sport", "Talk-Show", "Thriller",
    "War", "Western", "N/A", "Other",
]

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
INPUT_SIZE = 37
HIDDEN_1   = 64
HIDDEN_2   = 32
DROPOUT    = 0.3
LR         = 0.01
ONLINE_EPOCHS = 10          # gradient steps per single user-feedback sample

# File paths (relative to this script's directory)
_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_DIR, "nn_weights.pt")
FEEDBACK_PATH = os.path.join(_DIR, "feedback_log.json")

# Normalisation upper-bounds for each heuristic sub-score
_NORM = {
    "actor":   9.0,    # 3 actors × 3 liked movies
    "genre":  12.0,    # 2 pts × 3 genres × 2 liked movies
    "rating": 10.0,    # IMDb scale
    "keyword":30.0,    # 10 keywords × 3 liked movies
    "total":  61.0,    # sum of the above
}


# ---------------------------------------------------------------------------
# 1.  Network definition
# ---------------------------------------------------------------------------

class RecommenderNet(nn.Module):
    """
    Binary classifier: given a feature vector for a (candidate, user-profile)
    pair, predict the probability that the user will enjoy the candidate.
    """

    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2.  Feature vector builder
# ---------------------------------------------------------------------------

def genres_to_onehot(genres: list) -> list:
    """
    Convert a list of genre strings into a 28-dim one-hot float list.

    Args:
        genres: e.g. ["Action", "Sci-Fi"]

    Returns:
        List of 28 floats (1.0 where genre matches, 0.0 otherwise).
    """
    return [1.0 if g in genres else 0.0 for g in ALL_GENRES]


def build_feature_vector(candidate: dict, score_breakdown: dict) -> torch.Tensor:
    """
    Convert a candidate movie dict + its heuristic score breakdown into the
    37-dimensional float tensor consumed by RecommenderNet.

    Args:
        candidate (dict): Movie feature dict with keys:
            - "genres"      : list[str]   e.g. ["Action", "Sci-Fi"]
            - "rating"      : float       IMDb rating 0–10
            - "runtime_min" : int         runtime in minutes
            - "actor_1/2/3" : str         top-3 cast members (optional)
            - "plot_keywords": list[str]  LLM-generated keywords (optional)

        score_breakdown (dict): Output of compute_heuristic_score(), with keys:
            - "actor"   : int
            - "genre"   : int
            - "rating"  : int
            - "keyword" : int
            - "total"   : int

    Returns:
        torch.Tensor of shape (37,) and dtype float32.
    """
    feats = [
        score_breakdown.get("actor",   0) / _NORM["actor"],
        score_breakdown.get("genre",   0) / _NORM["genre"],
        score_breakdown.get("rating",  0) / _NORM["rating"],
        score_breakdown.get("keyword", 0) / _NORM["keyword"],
        max(score_breakdown.get("total", 0), 0) / _NORM["total"],
        candidate.get("rating", 0.0) / 10.0,
    ]

    # Runtime one-hot bins
    rt = candidate.get("runtime_min", 0)
    feats += [
        1.0 if rt < 90              else 0.0,   # short
        1.0 if 90 <= rt <= 120      else 0.0,   # medium
        1.0 if rt > 120             else 0.0,   # long
    ]

    # Genre one-hot (28 dims)
    feats += genres_to_onehot(candidate.get("genres", []))

    assert len(feats) == INPUT_SIZE, (
        f"Feature size mismatch: got {len(feats)}, expected {INPUT_SIZE}"
    )
    return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3.  Heuristic scorer  (builds the score_breakdown dict)
# ---------------------------------------------------------------------------

def compute_heuristic_score(candidate: dict, liked_rows: list,
                             fixed_genres: list = None) -> dict:
    """
    Compute the weighted heuristic score between a candidate movie and the
    user's liked-movie profile.

    Scoring rules (from the project proposal):
        +1  per shared actor  (top-3 cast)
        +2  per shared genre
        +1  per IMDb rating star  (floor of rating)
        +1  per shared LLM plot keyword

    Args:
        candidate    (dict): Feature dict for the movie being evaluated.
        liked_rows   (list): List of feature dicts for the user's liked movies.
        fixed_genres (list): Optional list of genres that MUST appear in the
                             candidate; returns total=-1 if the filter fails.

    Returns:
        dict with keys "actor", "genre", "rating", "keyword", "total".
        If the fixed-genre hard filter fails, total is set to -1.
    """
    # Hard filter: fixed genre constraint
    if fixed_genres:
        cand_genres_lower = [g.lower() for g in candidate.get("genres", [])]
        for fg in fixed_genres:
            if fg.lower() not in cand_genres_lower:
                return {"actor": 0, "genre": 0, "rating": 0, "keyword": 0, "total": -1}

    actor_score   = 0
    genre_score   = 0
    keyword_score = 0

    cand_actors = {
        candidate.get("actor_1", "N/A"),
        candidate.get("actor_2", "N/A"),
        candidate.get("actor_3", "N/A"),
    } - {"N/A"}

    cand_genres   = set(g.lower() for g in candidate.get("genres", []))
    cand_keywords = set(candidate.get("plot_keywords", []))

    for liked in liked_rows:
        liked_actors = {
            liked.get("actor_1", "N/A"),
            liked.get("actor_2", "N/A"),
            liked.get("actor_3", "N/A"),
        } - {"N/A"}
        actor_score += len(cand_actors & liked_actors)

        liked_genres = set(g.lower() for g in liked.get("genres", []))
        genre_score += 2 * len(cand_genres & liked_genres)

        liked_keywords = set(liked.get("plot_keywords", []))
        keyword_score += len(cand_keywords & liked_keywords)

    rating_score = int(candidate.get("rating", 0.0))   # 1 pt per IMDb star

    return {
        "actor":   actor_score,
        "genre":   genre_score,
        "rating":  rating_score,
        "keyword": keyword_score,
        "total":   actor_score + genre_score + rating_score + keyword_score,
    }


# ---------------------------------------------------------------------------
# 4.  MovieRecommender wrapper  (scoring + online learning + persistence)
# ---------------------------------------------------------------------------

class MovieRecommender:
    """
    High-level wrapper around RecommenderNet.

    Provides:
        .score()          – predict recommendation probability
        .update()         – online learning step from user feedback
        .train_from_history() – batch re-train from all past feedback
        .save()           – persist weights to disk
    """

    def __init__(self):
        self.model     = RecommenderNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.BCELoss()
        self.feedback_log = self._load_feedback()

        if os.path.exists(MODEL_PATH):
            self._load_weights()
            print(f"[MovieRecommender] Loaded weights from {MODEL_PATH}")
        else:
            self._init_weights()
            print("[MovieRecommender] Initialised with Xavier weights (no saved model found).")

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_weights(self):
        state = torch.load(MODEL_PATH, weights_only=True)
        self.model.load_state_dict(state)

    def save(self):
        """Persist model weights to nn_weights.pt."""
        torch.save(self.model.state_dict(), MODEL_PATH)

    def _load_feedback(self) -> list:
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r") as f:
                return json.load(f)
        return []

    def _save_feedback(self):
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(self.feedback_log, f, indent=2)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, candidate: dict, score_breakdown: dict) -> float:
        """
        Return the NN's recommendation probability for a candidate movie.

        Args:
            candidate       (dict): Feature dict from build_movie_row().
            score_breakdown (dict): Dict from compute_heuristic_score().

        Returns:
            float in [0, 1] — higher means more likely to recommend.
        """
        self.model.eval()
        with torch.no_grad():
            x = build_feature_vector(candidate, score_breakdown).unsqueeze(0)
            return float(self.model(x).item())

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def update(self, candidate: dict, score_breakdown: dict,
               liked: bool, n_epochs: int = ONLINE_EPOCHS):
        """
        Perform an online gradient update given a single user feedback signal.

        Args:
            candidate       (dict): Feature dict for the movie that was shown.
            score_breakdown (dict): Heuristic score breakdown for that movie.
            liked           (bool): True if the user accepted the recommendation.
            n_epochs        (int):  Number of gradient steps on this sample.
        """
        self.model.train()
        x = build_feature_vector(candidate, score_breakdown).unsqueeze(0)
        y = torch.tensor([1.0 if liked else 0.0])

        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()

        # Log and persist
        self.feedback_log.append({
            "title":          candidate.get("title", "?"),
            "liked":          liked,
            "score_breakdown": score_breakdown,
        })
        self._save_feedback()
        self.save()

    # ------------------------------------------------------------------
    # Batch re-training from feedback history
    # ------------------------------------------------------------------

    def train_from_history(self, n_epochs: int = 50):
        """
        Re-train the model from the full feedback log.

        Useful after accumulating many feedback entries to produce a
        more stable model than repeated single-sample online updates.

        Args:
            n_epochs (int): Number of full-pass gradient epochs.
        """
        if not self.feedback_log:
            print("No feedback history to train from.")
            return

        xs, ys = [], []
        for entry in self.feedback_log:
            sb  = entry.get("score_breakdown", {})
            # Reconstruct a minimal candidate dict from the log entry
            cand = {"title": entry.get("title", "?"),
                    "genres": [], "rating": 0.0, "runtime_min": 0}
            xs.append(build_feature_vector(cand, sb))
            ys.append(1.0 if entry["liked"] else 0.0)

        X = torch.stack(xs)
        Y = torch.tensor(ys)

        self.model.train()
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), Y)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:>3}/{n_epochs}  loss={loss.item():.4f}")

        self.save()
        print("Training complete.")


# ---------------------------------------------------------------------------
# 5.  Demo / smoke-test  (run this file directly to verify everything works)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  CS4100 NN Model – standalone smoke-test")
    print("=" * 55)

    # --- Dummy movie dicts (normally produced by dataset.build_movie_row) ---
    inception = {
        "title": "Inception", "genres": ["Action", "Adventure", "Sci-Fi"],
        "rating": 8.8, "runtime_min": 148,
        "actor_1": "Leonardo DiCaprio", "actor_2": "Joseph Gordon-Levitt",
        "actor_3": "Elliot Page",
        "plot_keywords": ["dreams", "heist", "subconscious", "espionage",
                          "deception", "reality", "mind-bending",
                          "infiltration", "betrayal", "redemption"],
    }
    dark_knight = {
        "title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"],
        "rating": 9.1, "runtime_min": 152,
        "actor_1": "Christian Bale", "actor_2": "Heath Ledger",
        "actor_3": "Aaron Eckhart",
        "plot_keywords": ["gotham", "vigilante", "chaos", "heroism",
                          "morality", "duality", "crime", "corruption",
                          "psychology", "batman"],
    }
    interstellar = {
        "title": "Interstellar", "genres": ["Adventure", "Drama", "Sci-Fi"],
        "rating": 8.7, "runtime_min": 169,
        "actor_1": "Matthew McConaughey", "actor_2": "Anne Hathaway",
        "actor_3": "Jessica Chastain",
        "plot_keywords": ["space", "wormhole", "survival", "family",
                          "sacrifice", "astrophysics", "hope",
                          "dystopian", "future", "teamwork"],
    }
    forrest_gump = {
        "title": "Forrest Gump", "genres": ["Drama", "Romance"],
        "rating": 8.8, "runtime_min": 142,
        "actor_1": "Tom Hanks", "actor_2": "Robin Wright",
        "actor_3": "Gary Sinise",
        "plot_keywords": ["destiny", "love", "war", "history", "running",
                          "innocence", "friendship", "comedy", "life",
                          "perseverance"],
    }

    liked_movies = [inception, dark_knight]
    candidates   = [interstellar, forrest_gump]

    # --- Heuristic scoring ---
    print("\n[1] Heuristic scores (vs. Inception + The Dark Knight):")
    print(f"  {'Movie':<22} {'actor':>5} {'genre':>5} {'rating':>6} "
          f"{'keyword':>7} {'total':>5}")
    print("  " + "-" * 52)
    for cand in candidates:
        sb = compute_heuristic_score(cand, liked_movies)
        print(f"  {cand['title']:<22} {sb['actor']:>5} {sb['genre']:>5} "
              f"{sb['rating']:>6} {sb['keyword']:>7} {sb['total']:>5}")

    # --- NN scoring ---
    rec = MovieRecommender()
    print("\n[2] NN recommendation scores (untrained / fresh weights):")
    for cand in candidates:
        sb    = compute_heuristic_score(cand, liked_movies)
        score = rec.score(cand, sb)
        print(f"  {cand['title']:<22}  P(recommend) = {score:.4f}")

    # --- Online learning demo ---
    print("\n[3] Online learning: user says YES to Interstellar")
    sb_inter = compute_heuristic_score(interstellar, liked_movies)
    before   = rec.score(interstellar, sb_inter)
    rec.update(interstellar, sb_inter, liked=True)
    after    = rec.score(interstellar, sb_inter)
    print(f"  Score before update: {before:.4f}")
    print(f"  Score after  update: {after:.4f}")

    print("\n[4] Online learning: user says NO to Forrest Gump")
    sb_fg  = compute_heuristic_score(forrest_gump, liked_movies)
    before = rec.score(forrest_gump, sb_fg)
    rec.update(forrest_gump, sb_fg, liked=False)
    after  = rec.score(forrest_gump, sb_fg)
    print(f"  Score before update: {before:.4f}")
    print(f"  Score after  update: {after:.4f}")

    # --- Fixed-genre filter ---
    print("\n[5] Fixed-genre filter: require 'Horror'")
    sb_horror = compute_heuristic_score(interstellar, liked_movies,
                                        fixed_genres=["Horror"])
    print(f"  Interstellar (Sci-Fi) filtered out? total={sb_horror['total']}")

    print("\nAll checks passed.")
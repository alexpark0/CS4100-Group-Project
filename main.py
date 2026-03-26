import pandas as pd
from omdbAPIs import get_movie_by_title
from llmAPIs import generate_plot_keywords

def build_movie_row(title: str) -> dict:
    """
    Fetch OMDB data and LLM keywords for a movie, return a flat dict.

    Fields returned:
        title         str         movie title
        actor_1/2/3   str         top-3 cast members
        genre         str         raw comma-separated genre string (e.g. "Action, Sci-Fi")
        genres        list[str]   parsed genre list  ← used by nn_model
        rating        float       IMDb rating 0-10   ← used by nn_model
        runtime_min   int         runtime in minutes ← used by nn_model
        plot_keywords list[str]   LLM-generated keywords
    """
    movie = get_movie_by_title(title)

    actors = [a.strip() for a in movie.get("Actors", "").split(",")]
    top3_actors = actors[:3] + ["N/A"] * (3 - len(actors))

    genre_str = movie.get("Genre", "N/A")
    genres = [g.strip() for g in genre_str.split(",") if g.strip() and g.strip() != "N/A"]

    try:
        rating = float(movie.get("imdbRating", "0") or "0")
    except ValueError:
        rating = 0.0

    runtime_str = movie.get("Runtime", "0 min")
    try:
        runtime_min = int(runtime_str.split()[0])
    except (ValueError, IndexError):
        runtime_min = 0

    keywords = generate_plot_keywords(movie.get("Plot", ""), movie.get("Title", title))

    return {
        "title":        movie.get("Title", title),
        "actor_1":      top3_actors[0],
        "actor_2":      top3_actors[1],
        "actor_3":      top3_actors[2],
        "genre":        genre_str,
        "genres":       genres,
        "rating":       rating,
        "runtime_min":  runtime_min,
        "plot_keywords": keywords,
    }


if __name__ == "__main__":
    titles = ["Inception", "The Dark Knight", "Interstellar"]

    rows = [build_movie_row(t) for t in titles]
    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

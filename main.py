import pandas as pd
from omdbAPIs import get_movie_by_title
from llmAPIs import generate_plot_keywords

def build_movie_row(title: str) -> dict:
    # Fetch OMDB data and LLM keywords for a movie, return a flat dict.
    movie = get_movie_by_title(title)

    actors = [a.strip() for a in movie.get("Actors", "").split(",")]
    top3_actors = actors[:3] + ["N/A"] * (3 - len(actors))  # pad if < 3 actors

    keywords = generate_plot_keywords(movie.get("Plot", ""), movie.get("Title", title))

    return {
        "title": movie.get("Title", title),
        "actor_1": top3_actors[0],
        "actor_2": top3_actors[1],
        "actor_3": top3_actors[2],
        "genre": movie.get("Genre", "N/A"),
        "rating": movie.get("imdbRating", "N/A"),
        "plot_keywords": keywords,
    }


if __name__ == "__main__":
    titles = ["Inception", "The Dark Knight", "Interstellar"]

    rows = [build_movie_row(t) for t in titles]
    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

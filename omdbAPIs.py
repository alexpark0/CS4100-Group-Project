import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OMDB_API_KEY", "e9b4e218")
BASE_URL = "http://www.omdbapi.com/"

# Session-scoped cache to avoid re-fetching the same movie (OMDB free = 1000 req/day)
_cache = {}


def get_movie_by_title(title: str, year: int = None) -> dict:
    """Fetch movie details by title."""
    cache_key = f"title:{title.lower()}:{year}"
    if cache_key in _cache:
        return _cache[cache_key]

    params = {
        "apikey": API_KEY,
        "t": title,
        "plot": "full",
        "type": "movie",
    }
    if year:
        params["y"] = year

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("Response") == "True":
        _cache[cache_key] = data
        return data
    else:
        raise ValueError(f"Movie not found: {data.get('Error', 'Unknown error')}")


def get_movie_by_id(imdb_id: str) -> dict:
    """Fetch movie details by IMDb ID."""
    if imdb_id in _cache:
        return _cache[imdb_id]

    params = {
        "apikey": API_KEY,
        "i": imdb_id,
        "plot": "full",
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("Response") == "True":
        _cache[imdb_id] = data
        return data
    else:
        raise ValueError(f"Movie not found: {data.get('Error', 'Unknown error')}")


def search_movies(query: str, page: int = 1) -> list:
    """Search for movies by keyword. Returns a list of brief results."""
    cache_key = f"search:{query.lower()}:{page}"
    if cache_key in _cache:
        return _cache[cache_key]

    params = {
        "apikey": API_KEY,
        "s": query,
        "type": "movie",
        "page": page,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("Search", []) if data.get("Response") == "True" else []
    _cache[cache_key] = results
    return results


# --- Example usage ---
if __name__ == "__main__":
    # 1. Fetch by title
    movie = get_movie_by_title("Inception")
    print(f"Title:    {movie['Title']} ({movie['Year']})")
    print(f"Genre:    {movie['Genre']}")
    print(f"Director: {movie['Director']}")
    print(f"IMDb:     {movie['imdbRating']}/10")
    print(f"Plot:     {movie['Plot'][:120]}...")
    print()

    # 2. Fetch by IMDb ID
    movie2 = get_movie_by_id("tt0111161")  # The Shawshank Redemption
    print(f"By ID:    {movie2['Title']} ({movie2['Year']})")
    print()

    # 3. Search by keyword
    results = search_movies("Batman")
    print(f"Search results for 'Batman':")
    for r in results[:5]:
        print(f"  - {r['Title']} ({r['Year']}) [{r['imdbID']}]")

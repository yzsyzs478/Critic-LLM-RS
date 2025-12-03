import os
import json
import re
import argparse
import openai


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM movie recommendation (no Critic)."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="user_movie_history_sample.jsonl",
        help="Path to the user–movie interaction file in JSONL format."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm_movie_zero-shot.json",
        help="Path to save the LLM recommendation results (JSON)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model name used for recommendations."
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="",
        help="Base URL of the OpenAI-compatible API endpoint."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Maximum number of tokens to generate in each LLM call."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of retries for the LLM API call upon failure."
    )

    return parser.parse_args()


def suggestMovie(
    watchedMoviesSubset,
    model_name: str,
    max_tokens: int,
    temperature: float,
    attempt: int = 1,
    max_attempts: int = 5,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation system. Recommend movies based on user "
                "preferences derived from previously watched movies. Focus on movies "
                "from 1880 to 2020 with ratings above 4."
            ),
        },
        {
            "role": "user",
            "content": (
                "Can you recommend 10 movies for me? Rank them by how much I might "
                "like them, formatted as title(year)."
            ),
        },
    ]

    for index, movie in enumerate(watchedMoviesSubset):
        messages.append(
            {
                "role": "system",
                "content": (
                    f"User watched: {movie.get('title', '')} directed by "
                    f"{movie.get('directedBy', '')} starring "
                    f"{movie.get('starring', '')} Rating {movie.get('rating', '')}."
                ),
            }
        )

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        recommendations = response.choices[0].message.content.split("\n")
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(
                watchedMoviesSubset,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
        else:
            return ["Failed to get recommendations after several attempts."]

    return [line.strip() for line in recommendations if line.strip()]


def clean_title(title: str) -> str:
    return re.sub(r"^\d+\.\s*", "", title).strip()


def processBatch(
    file_path: str,
    output_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int = 5,
):
    with open(file_path, "r", encoding="utf-8") as file:
        users = [json.loads(line.strip()) for line in file if line.strip()]

    output_data = []

    for user in users:
        history = user.get("History", [])
        split_idx = len(history) // 3 if len(history) > 0 else 0
        watched_movies_subset = history[:split_idx]
        validation_set = [
            {"title": movie.get("title", ""), "rating": movie.get("rating", "")}
            for movie in history[split_idx:]
        ]
        recommendations = suggestMovie(
            watched_movies_subset,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )

        output_data.append(
            {
                "user_id": user.get("user_id"),
                "watched_movies_subset": watched_movies_subset,
                "validation_set": validation_set,
                "recommendations": recommendations,
            }
        )

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.api_base:
        openai.api_base = args.api_base
    else:
        env_base = os.getenv("OPENAI_API_BASE")
        if env_base:
            openai.api_base = env_base

    if args.api_key:
        openai.api_key = args.api_key
    else:
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            openai.api_key = env_key

    if not openai.api_key:
        print("[WARN] openai.api_key is empty. Please pass --api_key or set OPENAI_API_KEY.")

    processBatch(
        file_path=args.data_path,
        output_path=args.output_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
    )

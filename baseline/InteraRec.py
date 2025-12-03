import os
import json
import numpy as np
import re
import argparse
import openai


def parse_args():
    parser = argparse.ArgumentParser(
        description="InteraRec-style poster-only movie recommendation with preference extraction and HR/NDCG evaluation."
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
        default="llm_movie_inter_poster_only.json",
        help="Path to save the LLM recommendation results and metrics (JSON)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model name used for both preference extraction and recommendation."
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
        "--pref_max_tokens",
        type=int,
        default=800,
        help="Maximum tokens for the preference extraction LLM call."
    )
    parser.add_argument(
        "--rec_max_tokens",
        type=int,
        default=1500,
        help="Maximum tokens for the recommendation LLM call."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for both LLM calls."
    )
    parser.add_argument(
        "--pref_max_attempts",
        type=int,
        default=3,
        help="Maximum retry attempts for preference extraction."
    )
    parser.add_argument(
        "--rec_max_attempts",
        type=int,
        default=5,
        help="Maximum retry attempts for recommendation generation."
    )

    return parser.parse_args()


def clean_title(title):
    title = re.sub(r'^\s*[\d\.\)\-]+[\s\-]*', '', title).strip()
    title = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
    return title


def extract_preferences(
    watchedMoviesSubset,
    model_name,
    max_tokens=800,
    temperature=0.7,
    attempt=1,
    max_attempts=3,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an InteraRec-style interaction understanding module. "
                "You only receive poster images of movies (as URLs, no textual metadata). "
                "From these poster images, you must infer a structured summary of the user's "
                "current preferences.\n\n"
                "Output a single JSON object with keys:\n"
                "- preferred_genres: list of strings\n"
                "- preferred_actors: list of strings\n"
                "- style_keywords: list of short keywords describing themes/tones\n"
                "- visual_style_keywords: list of short keywords describing visual style\n"
                "- constraints: list of textual constraints\n\n"
                "You must rely only on the poster URLs as hints. "
                "Do not add any explanation outside the JSON. Output ONLY valid JSON."
            )
        },
        {
            "role": "user",
            "content": "Here is the user's recent viewing history (only poster images). Summarize the preferences as requested."
        }
    ]

    for index, movie in enumerate(watchedMoviesSubset, start=1):
        poster_url = movie.get('poster_url', '')
        messages.append(
            {
                "role": "user",
                "content": f"Movie {index}: poster_url={poster_url}."
            }
        )

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        raw_text = response.choices[0].message.content.strip()
        prefs = json.loads(raw_text)
        return prefs
    except Exception as e:
        print(f"Preference extraction error (attempt {attempt}): {e}")
        if attempt < max_attempts:
            return extract_preferences(
                watchedMoviesSubset,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
        else:
            return {
                "preferred_genres": [],
                "preferred_actors": [],
                "style_keywords": [],
                "visual_style_keywords": [],
                "constraints": []
            }


def suggestMovie(
    watchedMoviesSubset,
    model_name,
    pref_max_tokens=800,
    rec_max_tokens=1500,
    temperature=0.7,
    pref_max_attempts=3,
    rec_max_attempts=5,
):
    prefs = extract_preferences(
        watchedMoviesSubset,
        model_name=model_name,
        max_tokens=pref_max_tokens,
        temperature=temperature,
        max_attempts=pref_max_attempts,
    )

    instruction = (
        "You are an InteraRec-style interactive recommender. "
        "You receive a structured summary of the user's current preferences inferred only from movie posters, "
        "and a list of poster URLs representing the user's recent viewing session. "
        "Your goal is to generate a re-ranked list of 10 movies that best match "
        "the user's current visual and thematic preferences.\n\n"
        "Requirements:\n"
        "1. Use the preference summary (genres, style_keywords, visual_style_keywords, constraints) as the main signal.\n"
        "2. Recommend movies released between 1880 and 2020 with rating > 4.\n"
        "3. OUTPUT FORMAT: return EXACTLY 20 lines, each line is one movie in the form "
        "   title (year). Do not add bullets, numbering, JSON, or explanations.\n"
    )

    prefs_text = json.dumps(prefs, ensure_ascii=False)

    history_lines = []
    for idx, movie in enumerate(watchedMoviesSubset, start=1):
        poster_url = movie.get("poster_url", "")
        history_lines.append(f"{idx}. poster_url={poster_url}")

    history_block = "User recent session history (only poster URLs):\n" + "\n".join(history_lines)

    user_prompt = (
        instruction
        + "\nPREFERENCE SUMMARY (from interaction understanding module):\n"
        + prefs_text
        + "\n\n"
        + history_block
        + "\n\nNow generate the final re-ranked recommendation list:\n"
    )

    messages = [
        {"role": "system", "content": "You are a large language model acting as the InteraRec re-ranking module."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=rec_max_tokens,
            temperature=temperature
        )
        gptSuggestion = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during recommendation call: {e}")
        if rec_max_attempts > 1:
            return suggestMovie(
                watchedMoviesSubset,
                model_name=model_name,
                pref_max_tokens=pref_max_tokens,
                rec_max_tokens=rec_max_tokens,
                temperature=temperature,
                pref_max_attempts=pref_max_attempts,
                rec_max_attempts=rec_max_attempts - 1,
            )
        else:
            return ["Failed to get recommendations after several attempts."]

    recs = []
    for line in gptSuggestion:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("recommendations"):
            continue
        recs.append(line)

    return recs


def calculate_hr_ndcg(recommendations, validation_set, k=10):
    cleaned_recommendations = [clean_title(rec) for rec in recommendations[:k]]
    validation_titles = set(validation_set)

    hr_at_k = any(rec in validation_titles for rec in cleaned_recommendations)
    dcg_at_k = sum(
        1 / np.log2(i + 2)
        for i, rec in enumerate(cleaned_recommendations)
        if rec in validation_titles
    )
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(validation_titles), k)))
    ndcg_at_k = dcg_at_k / ideal_dcg if ideal_dcg > 0 else 0

    return hr_at_k, ndcg_at_k


def processBatch(
    file_path,
    output_path="llm_movie_inter_poster_only.json",
    model_name="gpt-3.5-turbo",
    pref_max_tokens=800,
    rec_max_tokens=1500,
    temperature=0.7,
    pref_max_attempts=3,
    rec_max_attempts=5,
    k_values=[10, 5, 3],
):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file if line.strip()]

    results = []

    for user in users:
        history = user.get('History', [])
        split_idx = len(history) // 3 if len(history) > 0 else 0
        watched_movies_subset = history[:split_idx]
        validation_set = [movie.get('title', '') for movie in history[split_idx:]]
        recommendations = suggestMovie(
            watchedMoviesSubset=watched_movies_subset,
            model_name=model_name,
            pref_max_tokens=pref_max_tokens,
            rec_max_tokens=rec_max_tokens,
            temperature=temperature,
            pref_max_attempts=pref_max_attempts,
            rec_max_attempts=rec_max_attempts,
        )

        user_result = {
            "user_id": user.get("user_id"),
            "watched_movies_subset": watched_movies_subset,
            "validation_set": validation_set,
            "recommendations": recommendations,
            "metrics": {}
        }

        for k in k_values:
            hr, ndcg = calculate_hr_ndcg(recommendations, validation_set, k)
            user_result["metrics"][f"HR@{k}"] = hr
            user_result["metrics"][f"NDCG@{k}"] = ndcg

        results.append(user_result)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

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
        pref_max_tokens=args.pref_max_tokens,
        rec_max_tokens=args.rec_max_tokens,
        temperature=args.temperature,
        pref_max_attempts=args.pref_max_attempts,
        rec_max_attempts=args.rec_max_attempts,
    )

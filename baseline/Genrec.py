import os
import json
import re
import argparse
import numpy as np
import openai


def parse_args():
    parser = argparse.ArgumentParser(
        description="GenRec-style generative movie recommendation with HR/NDCG evaluation."
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
        default="llm_movie_genrec.json",
        help="Path to save the LLM recommendation results and metrics (JSON)."
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
        default=0.1,
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of retries for the LLM API call upon failure."
    )

    return parser.parse_args()


def clean_title(title: str) -> str:
    title = re.sub(r'^\s*[\d\.\)\-]+[\s\-]*', '', title).strip()
    title = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
    return title


def suggestMovie(
    watchedMoviesSequence,
    model_name: str,
    max_tokens: int,
    temperature: float,
    attempt: int = 1,
    max_attempts: int = 5,
):
    instruction = (
        "INSTRUCTION:\n"
        "You are a generative recommendation model (GenRec-style). "
        "Given a chronological sequence of movies that a user has already watched, "
        "predict the next movies the user is most likely to watch and generate a "
        "ranked list of 10 candidate movies.\n\n"
        "Requirements:\n"
        "1. Use the movie titles (and optionally their semantics) as the main signal.\n"
        "2. Only recommend movies released between 1880 and 2020.\n"
        "3. Only recommend movies with rating > 4.\n"
        "4. Do NOT repeat movies that already appear in the history.\n"
        "5. OUTPUT FORMAT: return EXACTLY 20 lines, each line is one movie in the form "
        "title(year). Do not add bullets, numbering, or explanations.\n\n"
        "Example:\n"
        "History:\n"
        "1. The Lord of the Rings: The Fellowship of the Ring (2001)\n"
        "2. The Lord of the Rings: The Two Towers (2002)\n"
        "Output:\n"
        "The Lord of the Rings: The Return of the King (2003)"
    )

    history_lines = []
    for idx, movie in enumerate(watchedMoviesSequence, start=1):
        title = movie.get("title", "")
        year = movie.get("year", "")
        history_lines.append(f"{idx}. {title} ({year})")

    input_block = (
        "INPUT:\nUser's watched movie sequence (in chronological order):\n"
        + "\n".join(history_lines)
        + "\n\nOUTPUT:\n"
    )

    messages = [
        {"role": "system", "content": "You are a large language model for generative recommendation."},
        {"role": "user", "content": instruction + "\n\n" + input_block},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_text = response.choices[0].message.content
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        recommendations = [clean_title(l) for l in lines]
        return recommendations
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(
                watchedMoviesSequence,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
        else:
            return []


def calculate_hr_ndcg(recommendations, validation_item, k=10):
    if not validation_item:
        return 0.0, 0.0

    gt_title = clean_title(validation_item["title"])
    rec_at_k = [clean_title(r) for r in recommendations[:k]]

    hr_at_k = float(gt_title in rec_at_k)

    dcg_at_k = 0.0
    for i, rec in enumerate(rec_at_k):
        if rec == gt_title:
            dcg_at_k = 1.0 / np.log2(i + 2.0)
            break

    ideal_dcg = 1.0
    ndcg_at_k = dcg_at_k / ideal_dcg if ideal_dcg > 0 else 0.0

    return hr_at_k, ndcg_at_k


def processBatch(
    file_path: str,
    output_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
    k_values=[10, 20, 50],
):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file if line.strip()]

    results = []
    agg_metrics = {k: {"HR": [], "NDCG": []} for k in k_values}

    for user in users:
        history = user.get("History", [])
        if len(history) < 2:
            continue

        watched_sequence = history[:-1]
        validation_item = history[-1]

        recommendations = suggestMovie(
            watchedMoviesSequence=watched_sequence,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        )

        user_result = {
            "user_id": user.get("user_id"),
            "watched_sequence": watched_sequence,
            "validation_item": {
                "title": validation_item.get("title"),
                "rating": validation_item.get("rating", None),
            },
            "raw_recommendations": recommendations,
            "metrics": {},
        }

        for k in k_values:
            hr, ndcg = calculate_hr_ndcg(recommendations, validation_item, k=k)
            user_result["metrics"][f"HR@{k}"] = hr
            user_result["metrics"][f"NDCG@{k}"] = ndcg
            agg_metrics[k]["HR"].append(hr)
            agg_metrics[k]["NDCG"].append(ndcg)

        results.append(user_result)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

    for k in k_values:
        hr_list = agg_metrics[k]["HR"]
        ndcg_list = agg_metrics[k]["NDCG"]
        if hr_list:
            avg_hr = float(np.mean(hr_list))
            avg_ndcg = float(np.mean(ndcg_list))
            print(f"Average HR@{k}: {avg_hr:.4f}, Average NDCG@{k}: {avg_ndcg:.4f}")
        else:
            print(f"Average HR@{k}: N/A, Average NDCG@{k}: N/A")


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

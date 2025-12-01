import openai
import json
import re
import numpy as np

openai.api_base = ""
openai.api_key = ""


def clean_title(title: str) -> str:
    title = re.sub(r'^\s*[\d\.\)\-]+[\s\-]*', '', title).strip()
    title = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
    return title


def suggestMovie(watchedMoviesSequence, attempt=1, max_attempts=5):
    # the inference interface of a LLaMA-based generative recommendation model fine-tuned with LoRA
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
        {"role": "user", "content": instruction + "\n\n" + input_block}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        recommendations = [clean_title(l) for l in lines]
        return recommendations
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSequence, attempt + 1, max_attempts=max_attempts)
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


def processBatch(file_path, k_values=[10, 20, 50]):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file]

    results = []
    agg_metrics = {k: {"HR": [], "NDCG": []} for k in k_values}

    for user in users:
        history = user.get("History", [])
        if len(history) < 2:
            continue

        watched_sequence = history[:-1]
        validation_item = history[-1]

        recommendations = suggestMovie(watched_sequence)

        user_result = {
            "user_id": user["user_id"],
            "watched_sequence": watched_sequence,
            "validation_item": {
                "title": validation_item["title"],
                "rating": validation_item.get("rating", None)
            },
            "raw_recommendations": recommendations,
            "metrics": {}
        }

        for k in k_values:
            hr, ndcg = calculate_hr_ndcg(recommendations, validation_item, k=k)
            user_result["metrics"][f"HR@{k}"] = hr
            user_result["metrics"][f"NDCG@{k}"] = ndcg
            agg_metrics[k]["HR"].append(hr)
            agg_metrics[k]["NDCG"].append(ndcg)

        results.append(user_result)

    with open("llm_movie_genrec.json", "w", encoding="utf-8") as outfile:
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
    file_path = "user_movie_history_sample.jsonl"
    processBatch(file_path)

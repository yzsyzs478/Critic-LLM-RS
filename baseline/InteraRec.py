import openai
import json
import numpy as np
import re

openai.api_base = ""
openai.api_key = ""


def clean_title(title):
    title = re.sub(r'^\s*[\d\.\)\-]+[\s\-]*', '', title).strip()
    title = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
    return title


def extract_preferences(watchedMoviesSubset, attempt=1, max_attempts=3):
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
                "- preferred_actors: list of strings (if you can infer them from poster style/known movies)\n"
                "- style_keywords: list of short keywords describing themes/tones\n"
                "- visual_style_keywords: list of short keywords describing visual style inferred "
                "  from posters/images (e.g., dark, colorful, minimalist, fantasy, sci-fi)\n"
                "- constraints: list of textual constraints (e.g., preferred years, languages) if inferable\n\n"
                "You must rely only on the poster URLs as hints (assume you know what the posters look like). "
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
        messages.append({
            "role": "user",
            "content": f"Movie {index}: poster_url={poster_url}."
        })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        raw_text = response.choices[0].message.content.strip()
        prefs = json.loads(raw_text)
        return prefs
    except Exception as e:
        print(f"Preference extraction error (attempt {attempt}): {e}")
        if attempt < max_attempts:
            return extract_preferences(watchedMoviesSubset, attempt + 1, max_attempts)
        else:
            return {
                "preferred_genres": [],
                "preferred_actors": [],
                "style_keywords": [],
                "visual_style_keywords": [],
                "constraints": []
            }


def suggestMovie(watchedMoviesSubset, attempt=1, max_attempts=5):
    prefs = extract_preferences(watchedMoviesSubset)

    instruction = (
        "You are an InteraRec-style interactive recommender. "
        "You receive a structured summary of the user's current preferences inferred only from movie posters, "
        "and a list of poster URLs representing the user's recent viewing session. "
        "Your goal is to generate a re-ranked list of 10 movies that best match "
        "the user's current visual and thematic preferences.\n\n"
        "Requirements:\n"
        "1. Use the preference summary (genres, style_keywords, visual_style_keywords, constraints) as the main signal.\n"
        "2. Recommend movies released between 1880 and 2020 with rating > 4 (assume such items exist).\n"
        "3. OUTPUT FORMAT: return EXACTLY 20 lines, each line is one movie in the form "
        "   title (year). Do not add bullets, numbering, JSON, or explanations.\n"
    )

    prefs_text = json.dumps(prefs, ensure_ascii=False)

    history_lines = []
    for idx, movie in enumerate(watchedMoviesSubset, start=1):
        poster_url = movie.get("poster_url", "")
        history_lines.append(f"{idx}. poster_url={poster_url}")

    history_block = (
        "User recent session history (only poster URLs):\n"
        + "\n".join(history_lines)
    )

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
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        gptSuggestion = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during recommendation call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSubset, attempt + 1)
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
        [1 / np.log2(i + 2) for i, rec in enumerate(cleaned_recommendations) if rec in validation_titles]
    )
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(validation_titles), k))])
    ndcg_at_k = dcg_at_k / ideal_dcg if ideal_dcg > 0 else 0

    return hr_at_k, ndcg_at_k


def processBatch(file_path, output_path='llm_movie_inter_poster_only.json', k_values=[10, 5, 3]):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file]

    results = []

    for user in users:
        watched_movies_subset = user['History'][:len(user['History']) // 3]
        validation_set = [movie['title'] for movie in user['History'][len(user['History']) // 3:]]
        recommendations = suggestMovie(watched_movies_subset)

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


if __name__ == "__main__":
    file_path = 'user_movie_history_sample.jsonl'
    processBatch(file_path)

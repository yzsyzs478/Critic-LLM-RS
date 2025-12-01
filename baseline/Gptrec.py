import openai
import json
import re

openai.api_base = ""
openai.api_key = ""


def clean_title(raw_line: str) -> str:
    if not raw_line:
        return ""
    line = re.sub(r'^\s*[\-\*\d\.\)\(]+\s*', '', raw_line).strip()
    line = re.sub(r'\([^)]*\)', '', line).strip()
    line = line.strip(" -–—·:：")
    return line


def parse_recommendations(raw_lines):
    titles = []
    seen = set()
    for line in raw_lines:
        title = clean_title(line)
        if not title:
            continue
        if title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def suggest_movies_next_k(
    watched_movies_sequence,
    K=10,
    attempt=1,
    max_attempts=5,
    max_history_len=50
):
    if len(watched_movies_sequence) > max_history_len:
        watched_movies_sequence = watched_movies_sequence[-max_history_len:]

    history_lines = []
    for idx, movie in enumerate(watched_movies_sequence, start=1):
        history_lines.append(
            f"{idx}. Title: {movie.get('title', '')}; "
            f"Director: {movie.get('directedBy', '')}; "
            f"Cast: {movie.get('starring', '')}; "
            f"Rating: {movie.get('rating', '')}"
        )
    history_text = "\n".join(history_lines)

    system_prompt = (
        "You are a GPT-based sequential movie recommender model (similar to GPTRec). "
        "You receive a chronological sequence of the user's watched movies and must "
        "generate the NEXT-K movies the user is most likely to watch.\n"
        "- Treat the history as an ordered sequence.\n"
        "- Focus on semantic similarity of genre, style, director, actors, etc.\n"
        "- Only recommend movies released between 1880 and 2020.\n"
        "- Only recommend movies with an average rating above 4.\n"
        "- DO NOT repeat any movie that already appears in the history.\n"
        "- Output EXACTLY {K} lines, each line containing ONLY the movie title "
        "in English (optionally with the year in parentheses). "
        "Do not include explanations, bullets, or extra text."
    ).format(K=K)

    user_prompt = (
        "Here is the user's viewing history in chronological order:\n"
        f"{history_text}\n\n"
        f"Now, based on this sequence, recommend the NEXT {K} movies "
        "the user is most likely to watch. "
        "Remember: output exactly {K} lines, each line is one movie title (no other text)."
    ).format(K=K)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        raw_text = response.choices[0].message.content
        raw_lines = raw_text.splitlines()
        return parse_recommendations(raw_lines)
    except Exception as e:
        print(f"Error during API call (attempt {attempt}): {e}")
        if attempt < max_attempts:
            return suggest_movies_next_k(
                watched_movies_sequence,
                K=K,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                max_history_len=max_history_len
            )
        else:
            return []


def process_batch(file_path, output_path, K_max=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        users = [json.loads(line.strip()) for line in f]

    user_details = []

    for user in users:
        user_id = user.get('user_id', None)
        history = user.get('History', [])

        if not history or len(history) < 3:
            print(f"Skip user {user_id}: history too short.")
            continue

        split_index = max(1, int(len(history) * 0.7))
        watched_prefix = history[:split_index]
        validation_part = history[split_index:]
        validation_titles = [clean_title(m['title']) for m in validation_part if 'title' in m]

        rec_titles = suggest_movies_next_k(watched_prefix, K=K_max)

        user_details.append({
            "user_id": user_id,
            "history_prefix_len": len(watched_prefix),
            "validation_len": len(validation_titles),
            "watched_prefix": watched_prefix,
            "validation_titles": validation_titles,
            "recommendations": rec_titles
        })

        print(f"User ID: {user_id}, rec_count: {len(rec_titles)}")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(user_details, outfile, ensure_ascii=False, indent=4)

    return user_details


if __name__ == "__main__":
    input_file = "user_movie_history_sample.jsonl"
    output_file = "llm_movie_gptrec.json"
    process_batch(input_file, output_file)

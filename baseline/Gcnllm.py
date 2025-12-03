import os
import json
import random
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import requests
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph-aware LLM movie recommendation with BERT user profiles."
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
        default="llm_movie_gcnllm.json",
        help="Path to save the recommendation results (JSON)."
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="",
        help="URL of the OpenAI-compatible chat completions endpoint."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model name used for all chat-completion calls."
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="BERT model name used to build text embeddings."
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=20,
        help="Maximum number of movies in the sampled user history."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Maximum tokens for the recommendation LLM call."
    )
    parser.add_argument(
        "--profile_max_tokens",
        type=int,
        default=512,
        help="Maximum tokens for the graph-based profile LLM call."
    )
    parser.add_argument(
        "--max_neighbor_items",
        type=int,
        default=20,
        help="Maximum number of 2-hop neighbor items in the user–item graph."
    )

    return parser.parse_args()


def clean_title(title):
    return re.sub(r'^\d+\.\s*', '', title).strip()


def load_json_objects(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {e}")
    return data


def generate_embedding(text, tokenizer, bert_model, device, embedding_size):
    try:
        if not isinstance(text, str):
            text = str(text)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception as e:
        print(f"generate_embedding error: {e}")
        return np.zeros(embedding_size, dtype=np.float32)


def calculate_similarity(embedding_u, embedding_i):
    norm_u = np.linalg.norm(embedding_u)
    norm_i = np.linalg.norm(embedding_i)
    if norm_u == 0 or norm_i == 0:
        return 0.0
    return float(np.dot(embedding_u, embedding_i) / (norm_u * norm_i))


def call_llm(api_url, api_key, model_name, messages, temperature=0.7, max_tokens=1500):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""


def suggestMovies(
    watchedMoviesSubset,
    api_url,
    api_key,
    model_name,
    template="{title}",
    temperature=0.7,
    max_tokens=1500,
):
    system_prompt = (
        "You are a movie recommendation system. "
        "Given a set of movies (each including the title, directedBy, starring, and rating) that a user watched, "
        "recommend 10 movies based on the user's preferences. "
        f"The format for the recommended movies should be {template}."
    )
    user_content = (
        "Here are the movies that the user watched:\n"
        + "\n".join(
            [
                f"title: {m['title']}, directedBy: {m.get('directedBy','')}, "
                f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
                for m in watchedMoviesSubset
            ]
        )
        + "\nPlease recommend 10 movies and rank them according to how much the user might like them. "
        "The ranking should be based on a rating scale from 1 to 5, where 5 means I like them the most "
        "and 1 means I don't like them at all."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    content = call_llm(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not content:
        return "Failed to get recommendations."
    return content


def build_user_item_graph(file_path):
    user2items = defaultdict(set)
    item2users = defaultdict(set)
    item_info = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line.strip())
            user_id = obj.get("user_id")
            history = obj.get("History", [])
            if not user_id or not history:
                continue
            for movie in history:
                title = movie.get("title")
                if not title:
                    continue
                title_clean = clean_title(title)
                user2items[user_id].add(title_clean)
                item2users[title_clean].add(user_id)
                if title_clean not in item_info:
                    item_info[title_clean] = {
                        "title": title_clean,
                        "directedBy": movie.get("directedBy", ""),
                        "starring": movie.get("starring", ""),
                        "rating": movie.get("rating", "")
                    }
    print(f"[Graph] users: {len(user2items)}, items: {len(item2users)}")
    return user2items, item2users, item_info


def graph_conv_user_profile(
    user_id,
    watchedMoviesSubset,
    user2items,
    item2users,
    item_info,
    api_url,
    api_key,
    model_name,
    temperature=0.7,
    max_tokens=512,
    max_neighbor_items=20,
):
    history_lines = []
    for m in watchedMoviesSubset:
        history_lines.append(
            f"title: {m.get('title','')}, directedBy: {m.get('directedBy','')}, "
            f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
        )
    user_history_text = "\n".join(history_lines)

    neighbor_item_titles = set()
    watched_titles_clean = [clean_title(x.get("title", "")) for x in watchedMoviesSubset]
    watched_titles_clean_set = set(watched_titles_clean)

    for m in watchedMoviesSubset:
        t = clean_title(m.get("title", ""))
        for other_uid in item2users.get(t, []):
            if other_uid == user_id:
                continue
            for other_title in user2items.get(other_uid, []):
                if other_title not in watched_titles_clean_set:
                    neighbor_item_titles.add(other_title)
                if len(neighbor_item_titles) >= max_neighbor_items:
                    break
            if len(neighbor_item_titles) >= max_neighbor_items:
                break
        if len(neighbor_item_titles) >= max_neighbor_items:
            break

    neighbor_lines = []
    for t in neighbor_item_titles:
        info = item_info.get(t, {"title": t})
        neighbor_lines.append(
            f"title: {info.get('title', t)}, directedBy: {info.get('directedBy','')}, "
            f"starring: {info.get('starring','')}, rating: {info.get('rating','')}"
        )
    neighbor_text = "\n".join(neighbor_lines) if neighbor_lines else "None."

    system_prompt = (
        "You are a graph-aware profile summarizer for a recommendation system. "
        "Given a target user node and its neighboring movie nodes in a user–item interaction graph, "
        "you need to aggregate all the information and output a concise user preference profile. "
        "The profile should only describe the user's stable movie preferences (for example, preferred genres, eras, styles, actors, directors). "
        "Do not list all movies; instead, summarize and generalize the patterns."
    )
    user_prompt = (
        f"Target user id: {user_id}\n\n"
        "Movies directly watched by the user (1-hop neighbors):\n"
        f"{user_history_text}\n\n"
        "Movies watched by similar users (2-hop neighbors in the graph):\n"
        f"{neighbor_text}\n\n"
        "Task: Aggregate these graph neighbors and write a short paragraph (3–6 sentences) summarizing this user's movie preferences."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    profile = call_llm(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not profile:
        profile = "This user likes movies similar to the following list:\n" + user_history_text
    return profile


def processBatch(
    file_path,
    tokenizer,
    bert_model,
    device,
    embedding_size,
    user2items,
    item2users,
    item_info,
    api_url,
    api_key,
    model_name,
    history_length,
    temperature,
    max_tokens,
    profile_max_tokens,
    max_neighbor_items,
    output_path,
):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file if line.strip()]

    all_user_data = []

    for user in users:
        user_id = user.get('user_id')
        watched_movies = user.get('History', [])
        if not watched_movies:
            continue

        random.shuffle(watched_movies)

        if history_length is not None and len(watched_movies) > history_length:
            watched_movies = watched_movies[:history_length]

        split_index = len(watched_movies) // 3 if len(watched_movies) >= 3 else 1
        recommendation_subset = watched_movies[:split_index]
        validation_movies = watched_movies[split_index:]

        if not recommendation_subset:
            continue

        relevant_movies = {
            clean_title(movie['title']): float(movie.get('rating', 0))
            for movie in validation_movies
        }

        recommendation_response = suggestMovies(
            recommendation_subset,
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        recommendations = [
            clean_title(line)
            for line in recommendation_response.split('\n')
            if line.strip()
        ]

        raw_user_text = "User watched the following movies:\n" + "\n".join(
            [
                f"title: {m['title']}, directedBy: {m.get('directedBy','')}, "
                f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
                for m in recommendation_subset
            ]
        )
        emb_raw = generate_embedding(
            raw_user_text,
            tokenizer=tokenizer,
            bert_model=bert_model,
            device=device,
            embedding_size=embedding_size,
        )

        conv_profile_text = graph_conv_user_profile(
            user_id=user_id,
            watchedMoviesSubset=recommendation_subset,
            user2items=user2items,
            item2users=item2users,
            item_info=item_info,
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=profile_max_tokens,
            max_neighbor_items=max_neighbor_items,
        )

        emb_conv = generate_embedding(
            conv_profile_text,
            tokenizer=tokenizer,
            bert_model=bert_model,
            device=device,
            embedding_size=embedding_size,
        )

        user_embedding = (emb_raw + emb_conv) / 2.0

        recommendation_embeddings = [
            generate_embedding(
                title,
                tokenizer=tokenizer,
                bert_model=bert_model,
                device=device,
                embedding_size=embedding_size,
            )
            for title in recommendations
        ]

        matching_scores = [
            calculate_similarity(user_embedding, item_embedding)
            for item_embedding in recommendation_embeddings
        ]

        sorted_pairs = sorted(
            zip(matching_scores, recommendations),
            key=lambda pair: pair[0],
            reverse=True
        )
        sorted_recommendations = [x for _, x in sorted_pairs]

        user_data = {
            "user_id": user_id,
            "recommendations": sorted_recommendations[:10],
            "recommendation_input": [
                {
                    "title": m['title'],
                    "directedBy": m.get('directedBy', ''),
                    "starring": m.get('starring', ''),
                    "rating": m.get('rating', '')
                }
                for m in recommendation_subset
            ],
            "validation_set": [
                {"title": title, "rating": rating}
                for title, rating in relevant_movies.items()
            ],
            "user_profile_raw": raw_user_text,
            "user_profile_conv": conv_profile_text
        }
        all_user_data.append(user_data)
        print(f"Processed recommendations (with graph conv) for user {user_id}")

    if all_user_data:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(all_user_data, outfile, indent=4, ensure_ascii=False)
    else:
        print("No user data to save. All recommendations failed or no users were processed.")


if __name__ == "__main__":
    args = parse_args()

    if not args.api_url:
        env_url = os.getenv("LLM_API_URL")
        if env_url:
            args.api_url = env_url

    if not args.api_key:
        env_key = os.getenv("LLM_API_KEY")
        if env_key:
            args.api_key = env_key

    if not args.api_url:
        print("[WARN] api_url is empty. Please pass --api_url or set LLM_API_URL.")
    if not args.api_key:
        print("[WARN] api_key is empty. Please pass --api_key or set LLM_API_KEY.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    bert_model = AutoModel.from_pretrained(args.bert_model_name).to(device)
    bert_model.eval()
    embedding_size = bert_model.config.hidden_size

    user2items, item2users, item_info = build_user_item_graph(args.data_path)

    processBatch(
        file_path=args.data_path,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=device,
        embedding_size=embedding_size,
        user2items=user2items,
        item2users=item2users,
        item_info=item_info,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        history_length=args.history_length,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        profile_max_tokens=args.profile_max_tokens,
        max_neighbor_items=args.max_neighbor_items,
        output_path=args.output_path,
    )

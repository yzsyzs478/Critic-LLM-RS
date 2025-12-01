import os
import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import requests
from collections import defaultdict

API_URL = ""
API_KEY = ""

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

def generate_embedding(text, tokenizer, bert_model, device):
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
        return np.zeros(768)

def calculate_similarity(embedding_u, embedding_i):
    norm_u = np.linalg.norm(embedding_u)
    norm_i = np.linalg.norm(embedding_i)
    if norm_u == 0 or norm_i == 0:
        return 0.0
    return float(np.dot(embedding_u, embedding_i) / (norm_u * norm_i))

def call_llm(messages, temperature=0.7, max_tokens=1500):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""

def suggestMovies(watchedMoviesSubset, template="{title}", temperature=0.7):
    system_prompt = (
        "You are a movie recommendation system. "
        "Given a set of movies (each including the title, directedBy, starring, and rating) that a user watched, "
        "recommend 10 movies based on the user's preferences. "
        f"The format for the recommended movies should be {template}."
    )
    user_content = (
        "Here are the movies that the user watched:\n" +
        "\n".join([
            f"title: {m['title']}, directedBy: {m.get('directedBy','')}, "
            f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
            for m in watchedMoviesSubset
        ]) +
        "\nPlease recommend 10 movies and rank them according to how much the user might like them. "
        "The ranking should be based on a rating scale from 1 to 5, where 5 means I like them the most "
        "and 1 means I don't like them at all."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    content = call_llm(messages, temperature=temperature, max_tokens=1500)
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
    temperature=0.7,
    max_neighbor_items=20
):
    history_lines = []
    for m in watchedMoviesSubset:
        history_lines.append(
            f"title: {m.get('title','')}, directedBy: {m.get('directedBy','')}, "
            f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
        )
    user_history_text = "\n".join(history_lines)
    neighbor_item_titles = set()
    for m in watchedMoviesSubset:
        t = clean_title(m.get("title", ""))
        for other_uid in item2users.get(t, []):
            if other_uid == user_id:
                continue
            for other_title in user2items.get(other_uid, []):
                if other_title not in [clean_title(x.get("title","")) for x in watchedMoviesSubset]:
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
        "The profile should only describe the user's stable movie preferences (e.g., preferred genres, eras, styles, actors, directors). "
        "Do not list all movies verbatim; instead, summarize and generalize the patterns."
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
    profile = call_llm(messages, temperature=temperature, max_tokens=512)
    if not profile:
        profile = (
            "This user likes movies similar to the following list:\n" +
            user_history_text
        )
    return profile

def processBatch(
    file_path,
    tokenizer,
    bert_model,
    device,
    user2items,
    item2users,
    item_info,
    history_length=20
):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file if line.strip()]
    all_user_data = []
    for user in users:
        user_id = user['user_id']
        watched_movies = user['History']
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
        recommendation_response = suggestMovies(recommendation_subset, temperature=0.7)
        recommendations = [
            clean_title(line)
            for line in recommendation_response.split('\n')
            if line.strip()
        ]
        raw_user_text = "User watched the following movies:\n" + "\n".join([
            f"title: {m['title']}, directedBy: {m.get('directedBy','')}, "
            f"starring: {m.get('starring','')}, rating: {m.get('rating','')}"
            for m in recommendation_subset
        ])
        emb_raw = generate_embedding(raw_user_text, tokenizer, bert_model, device)
        conv_profile_text = graph_conv_user_profile(
            user_id=user_id,
            watchedMoviesSubset=recommendation_subset,
            user2items=user2items,
            item2users=item2users,
            item_info=item_info,
            temperature=0.7
        )
        emb_conv = generate_embedding(conv_profile_text, tokenizer, bert_model, device)
        user_embedding = (emb_raw + emb_conv) / 2.0
        recommendation_embeddings = [
            generate_embedding(title, tokenizer, bert_model, device)
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
        with open('llm_movie_gcnllm.json', 'w', encoding='utf-8') as outfile:
            json.dump(all_user_data, outfile, indent=4, ensure_ascii=False)
    else:
        print("No user data to save. All recommendations failed or no users were processed.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    file_path = 'user_movie_history_sample.jsonl'
    user2items, item2users, item_info = build_user_item_graph(file_path)
    processBatch(
        file_path=file_path,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=device,
        user2items=user2items,
        item2users=item2users,
        item_info=item_info,
        history_length=20
    )

# -*- coding: utf-8 -*-
import os
import json
import random
import re
import time
import sys
import signal
import argparse
from typing import List, Dict, Any, Set

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

_SHOULD_STOP = False


def _sigint_handler(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    print("\n[INFO] Interrupt received; will finish the current user, save a checkpoint, and exit safely.", file=sys.stderr)


signal.signal(signal.SIGINT, _sigint_handler)


class RatingPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10):
        super(RatingPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def clean_title(title: str) -> str:
    return re.sub(r'^\d+\.\s*', '', title).strip()


def load_json_objects(filepath: str) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON decoding error (skipping this line): {e}", file=sys.stderr)
    return data


@torch.no_grad()
def generate_embedding(text, tokenizer, bert_model, device, max_seq_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    ).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


@torch.no_grad()
def generate_batch_embeddings(texts, tokenizer, bert_model, device, max_seq_length):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    ).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def calculate_hr_ndcg(recommended_titles: List[str], relevant_titles: Set[str], top_k: int):
    hits = [1 if title in relevant_titles else 0 for title in recommended_titles[:top_k]]
    hr = 1 if any(hits) else 0
    dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits) if hit])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant_titles), top_k))])
    ndcg = dcg / idcg if idcg > 0 else 0
    return hr, ndcg


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def load_done_ids(done_path: str) -> Set[str]:
    ids = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(line.strip())
    return ids


def append_done_id(done_path: str, uid: str):
    with open(done_path, "a", encoding="utf-8") as f:
        f.write(uid + "\n")


def append_jsonl(jsonl_path: str, obj: Dict[str, Any]):
    line = json.dumps(obj, ensure_ascii=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_checkpoint(ckpt_path: str, obj: Dict[str, Any]):
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ckpt_path)


def suggestMovie(
    client,
    watchedMoviesSubset,
    previous_feedback=None,
    ratings=None,
    template="{title(year)}",
    temperature=0.7,
    model_name="qwen3-8b",
    max_tokens=1500,
    api_retry=5,
):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation system. Given a set of movies (each including the title, directedBy, starring) "
                "that a user watched, the user preference is based on the ratings that user give to these movies. Now, please "
                "recommend 10 movies based on the user's preferences, and the recommended years for the movie are 1880 to 2021. "
                f"The format for the recommended movies should be {template}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here are the movies that user watched and rated:\n" +
                "\n".join([
                    f"title: {m['title']}, directedBy: {m.get('directedBy','Unknown')}, "
                    f"starring: {m.get('starring','Unknown')}, rating: {m.get('rating',0)}"
                    for m in watchedMoviesSubset
                ]) +
                "\nPlease recommend 10 movies and rank them according to how much the user might like them? "
                "Please base the ranking on a rating scale from 1 to 5, where 5 means I like them the most and 1 means I don't like them at all."
            ),
        },
    ]

    if previous_feedback:
        messages.extend(previous_feedback)

    if ratings:
        messages.extend([
            {"role": "assistant", "content": ratings["response"]},
            {
                "role": "user",
                "content": (
                    "I have provided ratings for the movies in the initial recommendations. Based on the ratings I've provided, "
                    "please adjust your recommendations to ensure all the movies are ones I tend to enjoy. "
                    f"The provided ranking: {ratings['ranking']}"
                ),
            },
        ])

    backoff = 1.0
    for _ in range(api_retry):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False}
            )
            return [response.choices[0].message.content]
        except Exception as e:
            print(f"[WARN] API call failed: {e}", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)

    return ["Failed to get recommendations."]


def processBatch(
    file_path: str,
    movie_lookup: Dict[str, Dict[str, Any]],
    tokenizer,
    bert_model,
    model,
    device,
    client,
    output_dir: str,
    history_length: int,
    force: bool,
    max_seq_length: int,
    embedding_size: int,
    llm_model_name: str,
    llm_max_tokens: int,
    llm_temperature: float,
    llm_outer_attempts: int,
    llm_api_retry: int,
    template: str,
):
    ensure_dir(output_dir)
    results_path = os.path.join(output_dir, "llm_critic_movie.jsonl")
    done_ids_path = os.path.join(output_dir, "done_ids.txt")
    ckpt_path = os.path.join(output_dir, "checkpoint.json")

    users = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                users.append(json.loads(line.strip()))

    done_ids = set() if force else load_done_ids(done_ids_path)
    total = len(users)
    start_time_all = time.time()
    finished = 0 if force else len(done_ids)

    print(f"[INFO] Total users: {total}; completed: {finished}; force re-run: {force}")

    for idx, user in enumerate(users):
        if _SHOULD_STOP:
            print("[INFO] Stop flag received; saving checkpoint and exiting.", file=sys.stderr)
            break

        user_id = str(user.get("user_id"))
        if (not force) and (user_id in done_ids):
            continue

        per_user_start = time.time()
        try:
            watched_movies = user['History']
            random.shuffle(watched_movies)
            split_index = max(1, len(watched_movies) // 3)
            recommendation_subset = watched_movies[:split_index]
            validation_movies = watched_movies[split_index:]

            watched_movies_subset = recommendation_subset
            validation_set = validation_movies

            relevant_movies = {m['title']: float(m.get('rating', 0)) for m in validation_set}

            recommendation_subset_details = [{
                "title": m['title'],
                "directedBy": m.get('directedBy', 'Unknown'),
                "starring": m.get('starring', 'Unknown'),
                "rating": float(m.get('rating', 0))
            } for m in recommendation_subset]

            previous_feedback = []
            recommendation_responses = []

            gpt_t0 = time.time()
            for _ in range(llm_outer_attempts):
                recommendation_response = suggestMovie(
                    client,
                    recommendation_subset,
                    previous_feedback=previous_feedback,
                    template=template,
                    temperature=llm_temperature,
                    model_name=llm_model_name,
                    max_tokens=llm_max_tokens,
                    api_retry=llm_api_retry
                )
                recommendations = []
                for resp in recommendation_response:
                    recommendations.extend([clean_title(line) for line in resp.split('\n') if line.strip()])
                recommendation_responses.extend(recommendation_response)
                if any(title in relevant_movies for title in recommendations):
                    break
            gpt_t1 = time.time()

            ratings = {"response": recommendation_response[-1], "ranking": ""}
            rating_prediction_t0 = time.time()

            recommendation_texts = [
                f"{m['title']} directed by {m.get('directedBy','null')} starring {m.get('starring','null')}"
                for m in recommendation_subset
            ]
            recommendation_embeddings = generate_batch_embeddings(
                recommendation_texts, tokenizer, bert_model, device, max_seq_length
            )

            if len(recommendation_embeddings) < history_length:
                padding = np.zeros((history_length - len(recommendation_embeddings), embedding_size))
                recommendation_embeddings = np.vstack([recommendation_embeddings, padding])
            elif len(recommendation_embeddings) > history_length:
                recommendation_embeddings = recommendation_embeddings[:history_length]

            for title in recommendations:
                t = clean_title(title)
                movie_details = movie_lookup.get(t, {"directedBy": "null", "starring": "null"})
                predicted_text = f"{t} directed by {movie_details['directedBy']} starring {movie_details['starring']}"
                pred_emb = generate_embedding(predicted_text, tokenizer, bert_model, device, max_seq_length)
                combined = np.vstack([recommendation_embeddings, pred_emb]).flatten()
                with torch.no_grad():
                    out = model(torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device))
                    predicted_rating = torch.argmax(out, dim=1).item() * 0.5 + 0.5
                ratings['ranking'] += f"{t}: {predicted_rating}, "
            rating_prediction_t1 = time.time()

            prev_fb = [{"role": "assistant", "content": recommendation_response[-1]},
                       {"role": "user", "content": ratings['ranking']}]
            adjusted_t0 = time.time()
            adjusted_resp = suggestMovie(
                client,
                recommendation_subset,
                previous_feedback=prev_fb,
                ratings=ratings,
                template=template,
                temperature=llm_temperature,
                model_name=llm_model_name,
                max_tokens=llm_max_tokens,
                api_retry=llm_api_retry
            )
            adjusted_t1 = time.time()

            adjusted_pairs = re.findall(r'(\d+\.\s*(.*?)(?:\s*[-–]\s*\d(?:\.\d)?))', adjusted_resp[-1])
            adjusted_titles = [clean_title(t) for _, t in adjusted_pairs] if adjusted_pairs else [
                clean_title(x) for x in adjusted_resp[-1].splitlines() if x.strip()
            ]

            second_ratings = {"response": adjusted_resp[-1], "ranking": ""}
            second_rating_t0 = time.time()
            for title in adjusted_titles:
                t = clean_title(title)
                movie_details = movie_lookup.get(t, {"directedBy": "null", "starring": "null"})
                predicted_text = f"{t} directed by {movie_details['directedBy']} starring {movie_details['starring']}"
                pred_emb = generate_embedding(predicted_text, tokenizer, bert_model, device, max_seq_length)
                combined = np.vstack([recommendation_embeddings, pred_emb]).flatten()
                with torch.no_grad():
                    out = model(torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device))
                    predicted_rating = torch.argmax(out, dim=1).item() * 0.5 + 0.5
                second_ratings['ranking'] += f"{t}: {predicted_rating}, "
            second_rating_t1 = time.time()

            prev_fb += [{"role": "assistant", "content": adjusted_resp[-1]},
                        {"role": "user", "content": second_ratings['ranking']}]
            second_adj_t0 = time.time()
            second_adj_resp = suggestMovie(
                client,
                recommendation_subset,
                previous_feedback=prev_fb,
                ratings=second_ratings,
                template=template,
                temperature=llm_temperature,
                model_name=llm_model_name,
                max_tokens=llm_max_tokens,
                api_retry=llm_api_retry
            )
            second_adj_t1 = time.time()

            second_pairs = re.findall(r'(\d+\.\s*(.*?)(?:\s*[-–]\s*\d(?:\.\d)?))', second_adj_resp[-1])
            second_titles = [clean_title(t) for _, t in second_pairs] if second_pairs else [
                clean_title(x) for x in second_adj_resp[-1].splitlines() if x.strip()
            ]

            third_ratings = {"response": second_adj_resp[-1], "ranking": ""}
            third_rating_t0 = time.time()
            for title in second_titles:
                t = clean_title(title)
                movie_details = movie_lookup.get(t, {"directedBy": "null", "starring": "null"})
                predicted_text = f"{t} directed by {movie_details['directedBy']} starring {movie_details['starring']}"
                pred_emb = generate_embedding(predicted_text, tokenizer, bert_model, device, max_seq_length)
                combined = np.vstack([recommendation_embeddings, pred_emb]).flatten()
                with torch.no_grad():
                    out = model(torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device))
                    predicted_rating = torch.argmax(out, dim=1).item() * 0.5 + 0.5
                third_ratings['ranking'] += f"{t}: {predicted_rating}, "
            third_rating_t1 = time.time()

            prev_fb += [{"role": "assistant", "content": second_adj_resp[-1]},
                        {"role": "user", "content": third_ratings['ranking']}]
            third_adj_t0 = time.time()
            third_adj_resp = suggestMovie(
                client,
                recommendation_subset,
                previous_feedback=prev_fb,
                ratings=third_ratings,
                template=template,
                temperature=llm_temperature,
                model_name=llm_model_name,
                max_tokens=llm_max_tokens,
                api_retry=llm_api_retry
            )
            third_adj_t1 = time.time()

            third_pairs = re.findall(r'(\d+\.\s*(.*?)(?:\s*[-–]\s*\d(?:\.\d)?))', third_adj_resp[-1])
            third_titles = [clean_title(t) for _, t in third_pairs] if third_pairs else [
                clean_title(x) for x in third_adj_resp[-1].splitlines() if x.strip()
            ]

            rel_title_set = {t for t in relevant_movies.keys()}
            hr10, ndcg10 = calculate_hr_ndcg(third_titles, rel_title_set, 10)
            hr5, ndcg5 = calculate_hr_ndcg(third_titles, rel_title_set, 5)
            hr3, ndcg3 = calculate_hr_ndcg(third_titles, rel_title_set, 3)

            per_user_end = time.time()

            user_out = {
                "user_id": user_id,
                "adjusted_recommendations": adjusted_resp[-1],
                "second_adjusted_recommendations": second_adj_resp[-1],
                "third_adjusted_recommendations": third_adj_resp[-1],
                "watched_movies_subset": watched_movies_subset,
                "validation_set": validation_set
            }

            append_jsonl(results_path, user_out)
            append_done_id(done_ids_path, user_id)
            done_ids.add(user_id)
            finished += 1

            if finished % 10 == 0 or _SHOULD_STOP:
                save_checkpoint(ckpt_path, {
                    "processed": finished,
                    "total": total,
                    "progress": f"{finished}/{total}",
                    "last_user_id": user_id,
                    "elapsed_sec": time.time() - start_time_all,
                    "results_path": results_path
                })

            print(f"[OK] Completed processing user {user_id} (progress {finished}/{total})")

        except Exception as e:
            err = {
                "user_id": user_id,
                "error": str(e),
                "stage": "process_user",
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            append_jsonl(results_path, err)
            print(f"[ERR] Failed to process user {user_id}: {e} (logged; continuing with remaining users)", file=sys.stderr)

        if _SHOULD_STOP:
            break

    save_checkpoint(ckpt_path, {
        "processed": finished,
        "total": total,
        "progress": f"{finished}/{total}",
        "elapsed_sec": time.time() - start_time_all,
        "results_path": results_path
    })
    print(f"[DONE] Resumed run completed. Final progress {finished}/{total} ({finished/total:.1%}). Results written to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Critic-LLM-RS with an LLM backend and a trained rating predictor."
    )
    parser.add_argument(
        "--input",
        default="user_movie_history_sample.jsonl",
        help="Path to the user–movie interaction file (JSONL), e.g., user_movie_history_sample.jsonl."
    )
    parser.add_argument(
        "--movie_json",
        default="movie.json",
        help="Path to the movie metadata file (JSON or JSONL) containing title/director/cast info."
    )
    parser.add_argument(
        "--critic_ckpt",
        default="critic.pth",
        help="Path to the trained rating predictor checkpoint (.pth)."
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory where results, checkpoints, and logs will be saved."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all users even if they are already recorded in done_ids.txt."
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=20,
        help="Maximum number of historical movies per user used as input to the critic model."
    )
    parser.add_argument(
        "--bert_model_name",
        default="bert-base-uncased",
        help="Name or path of the BERT (or other Transformer) model used to encode texts."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Hidden layer size of the rating predictor MLP."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of discrete rating classes predicted by the critic model."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum token length for BERT tokenization when encoding movie texts."
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="API key for the OpenAI-compatible LLM endpoint."
    )
    parser.add_argument(
        "--base_url",
        default=" ",
        help="Base URL of the OpenAI-compatible LLM endpoint, e.g., http://127.0.0.1:8000/v1."
    )
    parser.add_argument(
        "--llm_model_name",
        default="qwen3-8b",
        help="Model name exposed by the LLM endpoint, e.g., qwen3-8b."
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM when generating recommendations."
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=1500,
        help="Maximum number of tokens to generate in each LLM call."
    )
    parser.add_argument(
        "--llm_outer_attempts",
        type=int,
        default=5,
        help="Maximum number of outer LLM recommendation attempts before giving up."
    )
    parser.add_argument(
        "--llm_api_retry",
        type=int,
        default=5,
        help="Maximum number of retries for failed LLM API calls (with exponential backoff)."
    )
    parser.add_argument(
        "--llm_template",
        default="{title(year)}",
        help="Output format template for LLM-recommended movies (e.g., {title(year)})."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    bert_model = AutoModel.from_pretrained(args.bert_model_name).to(device)
    bert_model.eval()
    embedding_size = bert_model.config.hidden_size

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    input_size = (args.history_len + 1) * embedding_size
    model = RatingPredictor(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.critic_ckpt, map_location=device))
    model.eval()

    movie_data = []
    with open(args.movie_json, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            movie_data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    movie_data.append(json.loads(line))

    movie_lookup = {clean_title(m["title"]): m for m in movie_data if "title" in m}

    processBatch(
        file_path=args.input,
        movie_lookup=movie_lookup,
        tokenizer=tokenizer,
        bert_model=bert_model,
        model=model,
        device=device,
        client=client,
        output_dir=args.output_dir,
        history_length=args.history_len,
        force=args.force,
        max_seq_length=args.max_seq_length,
        embedding_size=embedding_size,
        llm_model_name=args.llm_model_name,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_outer_attempts=args.llm_outer_attempts,
        llm_api_retry=args.llm_api_retry,
        template=args.llm_template,
    )


if __name__ == "__main__":
    main()

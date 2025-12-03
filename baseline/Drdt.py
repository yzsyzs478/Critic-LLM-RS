import os
import json
import re
import random
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from openai import OpenAI

client: OpenAI | None = None
MODEL_NAME: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="DRDT-style multi-step preference refinement and recommendation."
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
        default="llm_movie_drdt.json",
        help="Path to save the DRDT-style recommendation results (JSON)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the OpenAI-compatible endpoint (fallback: DASHSCOPE_API_KEY / OPENAI_API_KEY env)."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base URL of the OpenAI-compatible API endpoint (fallback: OPENAI_API_BASE env)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-8b",
        help="LLM model name used for all chat-completion calls."
    )
    parser.add_argument(
        "--max_cic_examples",
        type=int,
        default=5,
        help="Maximum number of collaborative in-context examples to retrieve per step."
    )
    parser.add_argument(
        "--pref_temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for preference inference and reflection."
    )
    parser.add_argument(
        "--rec_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for final recommendation generation."
    )
    parser.add_argument(
        "--pref_max_tokens",
        type=int,
        default=1500,
        help="Maximum tokens for preference-related LLM calls (inference + reflection)."
    )
    parser.add_argument(
        "--rec_max_tokens",
        type=int,
        default=1500,
        help="Maximum tokens for recommendation-generation LLM calls."
    )

    return parser.parse_args()


def clean_title_prefix(line: str) -> str:
    s = re.sub(r'^\s*[-\*\u2022]?\s*', '', line)
    s = re.sub(r'^\s*\d+[\.\)\-:]\s*', '', s)
    return s.strip()


def extract_title_only(s: str) -> str:
    s = s.strip()
    m = re.match(r'^(.*?)(?:\s*\((\d{4})\))?$', s)
    if m:
        return m.group(1).strip()
    return re.split(r'\s*-\s*|\s*\|\s*', s)[0].strip()


def normalize_title(s: str) -> str:
    return extract_title_only(s).lower()


def parse_llm_list(text: str, topk: int = 10) -> List[str]:
    lines = [l for l in text.split("\n") if l.strip()]
    out = []
    seen = set()
    for l in lines:
        t = clean_title_prefix(l)
        if not t:
            continue
        t = extract_title_only(t)
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            out.append(t)
            seen.add(key)
        if len(out) >= topk:
            break
    return out


def load_users_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def build_cic_index(
    users: List[Dict[str, Any]],
    min_prefix_len: int = 1
) -> Dict[str, List[Dict[str, Any]]]:
    cic_index: Dict[str, List[Dict[str, Any]]] = {}
    for user in users:
        history = user.get("History", [])
        if len(history) < min_prefix_len + 1:
            continue
        prefix = history[:-1]
        next_movie = history[-1]
        last_prefix_movie = prefix[-1]
        last_title = last_prefix_movie.get("title", "").strip()
        if not last_title:
            continue
        key = normalize_title(last_title)
        entry = {
            "prefix_history": prefix,
            "next_title": next_movie.get("title", "").strip()
        }
        cic_index.setdefault(key, []).append(entry)
    return cic_index


def format_history_for_prompt(history: List[Dict[str, Any]]) -> str:
    lines = []
    for m in history:
        title = m.get("title", "Unknown title")
        directedBy = m.get("directedBy", "Unknown")
        starring = m.get("starring", "Unknown")
        rating = m.get("rating", "Unknown")
        lines.append(
            f"title: {title}, directedBy: {directedBy}, "
            f"starring: {starring}, rating: {rating}"
        )
    return "\n".join(lines)


def format_cic_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "No collaborative examples are available.\n"
    parts = []
    for idx, ex in enumerate(examples, start=1):
        prefix_text = format_history_for_prompt(ex["prefix_history"])
        next_title = ex["next_title"]
        parts.append(
            f"Example {idx}:\n"
            f"  History:\n{prefix_text}\n"
            f"  Next movie chosen by this user: {next_title}\n"
        )
    return "\n".join(parts)


def get_cic_examples_for_last(
    cic_index: Dict[str, List[Dict[str, Any]]],
    history: List[Dict[str, Any]],
    max_cic_examples: int
) -> List[Dict[str, Any]]:
    if not history:
        return []
    last_title = history[-1].get("title", "").strip()
    if not last_title:
        return []
    key = normalize_title(last_title)
    examples = cic_index.get(key, [])
    if len(examples) > max_cic_examples:
        examples = random.sample(examples, max_cic_examples)
    return examples


def call_llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1500,
    max_attempts: int = 2
) -> Dict[str, Any]:
    global client, MODEL_NAME
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False},
                timeout=120,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                continue
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            jtxt = raw[start:end + 1]
            return json.loads(jtxt)
        except Exception as e:
            print(f"[WARN] call_llm_json error at attempt {attempt}: {e}")
            continue
    return {}


def divergent_thinking_preferences(
    history: List[Dict[str, Any]],
    cic_examples: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    history_text = format_history_for_prompt(history)
    cic_text = format_cic_examples_for_prompt(cic_examples)
    system_msg = (
        "You are an expert recommendation analyst. "
        "You must infer a user's multi-aspect movie preferences (genre, director, actors, style, etc.) "
        "from their watch history and some collaborative examples. "
        "You must output a single JSON object only, with no extra text."
    )
    user_msg = (
        "Here is the user's watch history with ratings:\n"
        f"{history_text}\n\n"
        "Here are collaborative examples from other users with similar behaviors:\n"
        f"{cic_text}\n\n"
        "Please infer a structured multi-aspect preference profile for this user. "
        "Use the following JSON schema (this is an example, you can adjust the content but keep the same structure):\n"
        "{\n"
        '  "genre": {\n'
        '    "preferred": ["Drama", "Romance"],\n'
        '    "disliked": ["Horror"],\n'
        '    "weight": 0.35,\n'
        '    "description": "Prefers emotionally rich dramas and romantic stories."\n'
        "  },\n"
        '  "director": {\n'
        '    "preferred": ["Christopher Nolan"],\n'
        '    "disliked": [],\n'
        '    "weight": 0.25,\n'
        '    "description": "Cares about famous directors with distinctive styles."\n'
        "  },\n"
        '  "actor": {\n'
        '    "preferred": ["Leonardo DiCaprio"],\n'
        '    "disliked": [],\n'
        '    "weight": 0.2,\n'
        '    "description": "Follows movies starring favorite actors."\n'
        "  },\n"
        '  "style": {\n'
        '    "preferred": ["slow-paced", "character-driven"],\n'
        '    "disliked": ["excessive violence"],\n'
        '    "weight": 0.2,\n'
        '    "description": "Likes character-driven plots with emotional depth."\n'
        "  }\n"
        "}\n\n"
        "Now output only one valid JSON object following this schema, with weights summing approximately to 1.0."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prefs = call_llm_json(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=2
    )
    return prefs


def reflect_preferences_with_feedback(
    current_profile: Dict[str, Any],
    prefix_history: List[Dict[str, Any]],
    true_next_movie: Dict[str, Any],
    cic_examples: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    history_text = format_history_for_prompt(prefix_history)
    cic_text = format_cic_examples_for_prompt(cic_examples)
    profile_json = json.dumps(current_profile, ensure_ascii=False, indent=2)
    true_title = true_next_movie.get("title", "Unknown title")
    true_directedBy = true_next_movie.get("directedBy", "Unknown")
    true_starring = true_next_movie.get("starring", "Unknown")
    system_msg = (
        "You are an expert recommendation analyst. "
        "You receive a current multi-aspect preference profile (as JSON), "
        "a prefix of the user's watch history, and the actual next movie they chose. "
        "You must update the preference profile to better fit this observed behavior. "
        "You must output a single JSON object only, with the same schema as the input profile."
    )
    user_msg = (
        "Current preference profile (JSON):\n"
        f"{profile_json}\n\n"
        "User's watch history prefix:\n"
        f"{history_text}\n\n"
        "Collaborative examples from other users:\n"
        f"{cic_text}\n\n"
        "The actual next movie watched by this user is:\n"
        f"title: {true_title}, directedBy: {true_directedBy}, starring: {true_starring}\n\n"
        "Please analyze whether the current preference profile is consistent with this actual next choice. "
        "Then adjust the aspect weights and possibly the preferred/disliked lists to better explain this behavior. "
        "If an aspect strongly supports this choice, slightly increase its weight; "
        "if an aspect contradicts this choice, slightly decrease its weight. "
        "Maintain the same overall JSON schema and output only the updated JSON object."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    updated = call_llm_json(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=2
    )
    if updated:
        return updated
    return current_profile


def generate_recommendations_with_profile(
    history: List[Dict[str, Any]],
    preference_profile: Dict[str, Any],
    cic_examples: List[Dict[str, Any]],
    temperature: float = 0.7,
    template: str = "{title(year)}",
    max_tokens: int = 1500
) -> List[str]:
    global client, MODEL_NAME
    history_text = format_history_for_prompt(history)
    cic_text = format_cic_examples_for_prompt(cic_examples)
    profile_json = json.dumps(preference_profile, ensure_ascii=False, indent=2)
    watched_titles = set(m.get("title", "").strip() for m in history if m.get("title"))
    system_msg = (
        "You are a movie recommendation system. "
        "You are given a user's watch history, an explicit multi-aspect preference profile, "
        "and some collaborative examples. You must recommend 10 new movies."
    )
    user_msg = (
        "User's watch history with ratings:\n"
        f"{history_text}\n\n"
        "Explicit multi-aspect preference profile (JSON):\n"
        f"{profile_json}\n\n"
        "Collaborative examples from other users:\n"
        f"{cic_text}\n\n"
        "Based on all the information above, please recommend 10 movies that this user is most likely to enjoy next.\n"
        "Requirements:\n"
        "1. Use the preference profile as the primary signal to guide your recommendations.\n"
        "2. Use the collaborative examples as an auxiliary signal when appropriate.\n"
        "3. Recommend only movies released between 1880 and 2021.\n"
        "4. Do NOT recommend movies the user has already watched.\n"
        f"5. The format for the recommended movies should be {template}.\n"
        "6. Return exactly 10 lines, numbered 1..10, each line containing one movie title (optionally with year), "
        "without any extra commentary."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"enable_thinking": False},
            timeout=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise RuntimeError("Empty response from model.")
        recs = parse_llm_list(raw, topk=10)
        filtered = []
        seen = set()
        for r in recs:
            rt = r.strip()
            if not rt:
                continue
            if rt in watched_titles:
                continue
            key = rt.lower()
            if key in seen:
                continue
            filtered.append(rt)
            seen.add(key)
            if len(filtered) >= 10:
                break
        if not filtered:
            return recs
        return filtered
    except Exception as e:
        print(f"[WARN] generate_recommendations_with_profile error: {e}")
        return []


def calculate_hr_ndcg(
    recommendations: List[str],
    validation_set: List[Dict[str, Any]]
) -> Tuple[float, float]:
    cleaned_recs = [r.strip() for r in recommendations[:10] if r.strip()]
    val_titles = [v["title"] for v in validation_set]
    hr_at_10 = 1.0 if any(r in val_titles for r in cleaned_recs) else 0.0
    dcg = 0.0
    for i, rec in enumerate(cleaned_recs):
        if rec in val_titles:
            dcg += 1.0 / np.log2(i + 2.0)
    ideal = min(len(val_titles), 10)
    idcg = (
        sum(1.0 / np.log2(i + 2.0) for i in range(ideal))
        if ideal > 0 else 1.0
    )
    ndcg_at_10 = (dcg / idcg) if idcg > 0 else 0.0
    return float(hr_at_10), float(ndcg_at_10)


def processBatch_drdt_style(
    file_path: str,
    output_path: str,
    max_cic_examples: int,
    pref_temperature: float,
    rec_temperature: float,
    pref_max_tokens: int,
    rec_max_tokens: int
):
    users = load_users_from_jsonl(file_path)
    cic_index = build_cic_index(users)
    print(f"[INFO] Loaded {len(users)} users.")
    print(f"[INFO] CIC index keys (unique last movies) = {len(cic_index)}")

    hr_list: List[float] = []
    ndcg_list: List[float] = []
    user_details: List[Dict[str, Any]] = []

    for idx, user in enumerate(users, start=1):
        user_id = user.get("user_id", f"user_{idx}")
        history = user.get("History", [])
        if len(history) < 4:
            user_details.append({
                "user_id": user_id,
                "skipped": True,
                "reason": "history length < 4"
            })
            continue

        n = len(history)
        split = max(3, int(n * 2 / 3))
        if split >= n:
            split = n - 1

        adapt_seq = history[:split]
        validation_seq = history[split:]

        initial_prefix_len = max(2, int(len(adapt_seq) / 3))
        if initial_prefix_len >= len(adapt_seq):
            initial_prefix_len = len(adapt_seq) - 1

        initial_prefix = adapt_seq[:initial_prefix_len]
        initial_cic_ex = get_cic_examples_for_last(cic_index, initial_prefix, max_cic_examples)

        pref_profile = divergent_thinking_preferences(
            initial_prefix,
            initial_cic_ex,
            temperature=pref_temperature,
            max_tokens=pref_max_tokens
        )
        if not pref_profile:
            pref_profile = {}

        step_profiles = []
        step_profiles.append({
            "step": "initial",
            "prefix_len": initial_prefix_len,
            "profile": pref_profile
        })

        for step_idx in range(initial_prefix_len, len(adapt_seq)):
            prefix = adapt_seq[:step_idx]
            true_next = adapt_seq[step_idx]
            cic_ex = get_cic_examples_for_last(cic_index, prefix, max_cic_examples)
            pref_profile = reflect_preferences_with_feedback(
                pref_profile,
                prefix,
                true_next,
                cic_ex,
                temperature=pref_temperature,
                max_tokens=pref_max_tokens
            )
            step_profiles.append({
                "step": step_idx,
                "prefix_len": len(prefix),
                "profile": pref_profile
            })

        final_cic_ex = get_cic_examples_for_last(cic_index, adapt_seq, max_cic_examples)
        recommendations = generate_recommendations_with_profile(
            adapt_seq,
            pref_profile,
            final_cic_ex,
            temperature=rec_temperature,
            template="{title(year)}",
            max_tokens=rec_max_tokens
        )

        validation_set = [
            {"title": m["title"], "rating": m.get("rating", 0)}
            for m in validation_seq
        ]
        hr, ndcg = calculate_hr_ndcg(recommendations, validation_set)
        hr_list.append(hr)
        ndcg_list.append(ndcg)

        user_details.append({
            "user_id": user_id,
            "skipped": False,
            "adapt_sequence": adapt_seq,
            "validation_sequence": validation_seq,
            "initial_prefix_len": initial_prefix_len,
            "step_profiles": step_profiles,
            "recommendations": recommendations,
            "hr_at_10": hr,
            "ndcg_at_10": ndcg,
            "num_cic_examples_final": len(final_cic_ex)
        })
        print(f"[DRDT-Style] user_id={user_id} HR@10={hr:.3f} NDCG@10={ndcg:.3f}")

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(user_details, outfile, ensure_ascii=False, indent=2)

    avg_hr = float(np.mean(hr_list)) if hr_list else 0.0
    avg_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    print(f"[DRDT-Style] Average HR@10: {avg_hr:.3f}, Average NDCG@10: {avg_ndcg:.3f}")
    print(f"[DRDT-Style] Valid users: {len(hr_list)} / {len(users)}")


if __name__ == "__main__":
    args = parse_args()

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_API_BASE") or ""

    if not api_key:
        print("[WARN] api_key is empty. Please pass --api_key or set DASHSCOPE_API_KEY / OPENAI_API_KEY.")
    if not base_url:
        print("[INFO] base_url is empty. Using default OpenAI-compatible base_url if required by your client.")

    MODEL_NAME = args.model_name
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    processBatch_drdt_style(
        file_path=args.data_path,
        output_path=args.output_path,
        max_cic_examples=args.max_cic_examples,
        pref_temperature=args.pref_temperature,
        rec_temperature=args.rec_temperature,
        pref_max_tokens=args.pref_max_tokens,
        rec_max_tokens=args.rec_max_tokens,
    )

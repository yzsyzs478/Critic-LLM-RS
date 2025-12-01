import json
import argparse
import numpy as np


def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.replace(".", "", 1).lstrip("+-").isdigit():
            try:
                return float(s)
            except ValueError:
                return None
    return None


def calculate_user_metrics(recommendations, k, relevance_threshold):
    precision_list, hr_list, ndcg_list = [], [], []

    for recs in recommendations:
        recs = [r for r in (to_float(v) for v in recs) if r is not None]

        if not recs:
            precision_list.append(0.0)
            hr_list.append(0.0)
            ndcg_list.append(0.0)
            continue

        k_eff = min(k, len(recs))
        top_k_recs = recs[:k_eff]

        rel = [1 if r >= relevance_threshold else 0 for r in top_k_recs]
        hits = sum(rel)

        precision = hits / k_eff
        hr = 1.0 if hits > 0 else 0.0

        dcg = sum((2**rel_i - 1) / np.log2(idx + 2) for idx, rel_i in enumerate(rel))

        ideal = sorted(recs, reverse=True)[:k_eff]
        ideal_rel = [1 if r >= relevance_threshold else 0 for r in ideal]
        idcg = sum((2**rel_i - 1) / np.log2(idx + 2) for idx, rel_i in enumerate(ideal_rel))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precision_list.append(precision)
        hr_list.append(hr)
        ndcg_list.append(ndcg)

    return precision_list, hr_list, ndcg_list


def calculate_metrics(recommendations, k, relevance_threshold):
    precision_list, hr_list, ndcg_list = calculate_user_metrics(recommendations, k, relevance_threshold)
    precision_mean, precision_std = float(np.mean(precision_list)), float(np.std(precision_list))
    hr_mean, hr_std = float(np.mean(hr_list)), float(np.std(hr_list))
    ndcg_mean, ndcg_std = float(np.mean(ndcg_list)), float(np.std(ndcg_list))
    return (precision_mean, precision_std), (hr_mean, hr_std), (ndcg_mean, ndcg_std)


def extract_recommendations(data, rec_type):
    recommendations = []
    for user_ratings in data.values():
        if rec_type in user_ratings:
            recs = []
            for v in user_ratings[rec_type].values():
                f = to_float(v)
                if f is not None:
                    recs.append(f)
            recommendations.append(recs)
    return recommendations


def validate_data(data, rec_types):
    for user, user_ratings in data.items():
        for rec_type in rec_types:
            if rec_type in user_ratings:
                for title, rating in user_ratings[rec_type].items():
                    if to_float(rating) is None:
                        print(f"[WARN] Non-numeric rating detected: user={user}, title={title}, rating={rating}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="real_rating_x_1.json")
    parser.add_argument("--output_user_metrics", default="x_metric.json")
    parser.add_argument("--relevance_threshold", type=float, default=4.0)
    parser.add_argument("--topk", default="10,5,3", help="Comma-separated list of K values, e.g. '10,5,3'")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        updated_ratings = json.load(file)

    rec_types = [
        "adjusted_recommendations",
        "second_adjusted_recommendations",
        "third_adjusted_recommendations",
    ]

    validate_data(updated_ratings, rec_types)

    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]

    metrics = {}
    user_metrics = {rec_type: {} for rec_type in rec_types}

    for rec_type in rec_types:
        recommendations = extract_recommendations(updated_ratings, rec_type)
        metrics[rec_type] = {}

        for k in topk_list:
            metrics[rec_type][f"top{k}"] = calculate_metrics(
                recommendations, k, args.relevance_threshold
            )

        for user, user_ratings in updated_ratings.items():
            if rec_type in user_ratings:
                recs = []
                for v in user_ratings[rec_type].values():
                    f = to_float(v)
                    if f is not None:
                        recs.append(f)

                per_user = {}
                for k in topk_list:
                    top_metrics = calculate_user_metrics([recs], k, args.relevance_threshold)
                    per_user[f"top{k}"] = {
                        "Precision": float(top_metrics[0][0]),
                        "HR": float(top_metrics[1][0]),
                        "NDCG": float(top_metrics[2][0]),
                    }

                user_metrics[rec_type][user] = per_user

    with open(args.output_user_metrics, "w", encoding="utf-8") as file:
        json.dump(user_metrics, file, indent=4, ensure_ascii=False)

    for rec_type, rec_metrics in metrics.items():
        print(f"{rec_type}:")
        for topk, ((precision_mean, precision_std), (hr_mean, hr_std), (ndcg_mean, ndcg_std)) in rec_metrics.items():
            print(f"  {topk}:")
            print(f"    Precision: {precision_mean:.4f} ± {precision_std:.4f}")
            print(f"    HR:        {hr_mean:.4f} ± {hr_std:.4f}")
            print(f"    NDCG:      {ndcg_mean:.4f} ± {ndcg_std:.4f}")


if __name__ == "__main__":
    main()

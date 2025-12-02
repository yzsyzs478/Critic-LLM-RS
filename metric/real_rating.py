import json
import re
import string
import argparse


def normalize_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    s = t.strip().lower()
    s = re.sub(r"\(\s*\d{4}\s*\)\s*$", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def to_float_if_numeric(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.replace(".", "", 1).lstrip("+-").isdigit():
            try:
                return float(s)
            except ValueError:
                return x
    return x


def build_validation_lookup(validation_data):
    lookup = {}
    for item in validation_data:
        user_id = item.get("user_id")
        vs = item.get("validation_set", {})
        if isinstance(vs, dict):
            val_map = {
                normalize_title(title): to_float_if_numeric(rating)
                for title, rating in vs.items()
            }
        elif isinstance(vs, list):
            val_map = {
                normalize_title(entry.get("title", "")): to_float_if_numeric(
                    entry.get("rating")
                )
                for entry in vs
            }
        else:
            val_map = {}
        lookup[user_id] = val_map
    return lookup


def update_predictions(user_predictions, validation_ratings, rec_type, changes_log, user_id):
    val_norm_to_rating = validation_ratings
    for raw_title in list(user_predictions.get(rec_type, {}).keys()):
        pred_rating = user_predictions[rec_type][raw_title]
        norm_pred = normalize_title(raw_title)

        if norm_pred in val_norm_to_rating:
            new_rating = val_norm_to_rating[norm_pred]
        else:
            new_rating = None
            for norm_val_title, vr in val_norm_to_rating.items():
                if (
                    norm_pred
                    and norm_val_title
                    and (norm_pred in norm_val_title or norm_val_title in norm_pred)
                ):
                    new_rating = vr
                    break

        if new_rating is not None and new_rating != pred_rating:
            user_predictions[rec_type][raw_title] = new_rating
            changes_log.append(
                f"Updated '{raw_title}' in {rec_type} for user {user_id} "
                f"from {pred_rating} to {new_rating}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted_file",
        default="predict_rating_x_1.json",
        help="Input JSON file with predicted ratings.",
    )
    parser.add_argument(
        "--validation_file",
        default="x_1.json",
        help="Input JSON file with validation ratings.",
    )
    parser.add_argument(
        "--output_file",
        default="real_rating_x_1.json",
        help="Output JSON file for updated ratings.",
    )
    parser.add_argument(
        "--log_file",
        default="changes_log.txt",
        help="Output text file for logging changes.",
    )
    args = parser.parse_args()

    with open(args.predicted_file, "r", encoding="utf-8") as file:
        predicted_ratings = json.load(file)

    with open(args.validation_file, "r", encoding="utf-8") as file:
        validation_data = json.load(file)

    validation_lookup = build_validation_lookup(validation_data)

    changes_log = []
    rec_types = [
        "adjusted_recommendations",
        "second_adjusted_recommendations",
        "third_adjusted_recommendations",
    ]

    for user_id, user_ratings in predicted_ratings.items():
        if user_id in validation_lookup:
            for rec_type in rec_types:
                if rec_type in user_ratings:
                    update_predictions(
                        user_ratings,
                        validation_lookup[user_id],
                        rec_type,
                        changes_log,
                        user_id,
                    )

    with open(args.log_file, "w", encoding="utf-8") as f:
        for line in changes_log:
            f.write(line + "\n")

    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(predicted_ratings, file, indent=4, ensure_ascii=False)

    print(
        f"Updated ratings saved to '{args.output_file}'. "
    )


if __name__ == "__main__":
    main()

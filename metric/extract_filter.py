import json
import re
import argparse


def extract_titles(recommendations: str):
    titles = []
    for line in recommendations.split("\n"):
        match = re.search(r"\d+\.\s*\(\d+\)\s*(.*? \(\d{4}\))", line)
        if not match:
            match = re.search(r"\d+\.\s*(.*? \(\d{4}\))\s*-\s*.*\(\d+/\d+\)", line)
        if not match:
            match = re.search(r"\d+\.\s*(.*? \(\d{4}\))", line)
        if match:
            title = match.group(1)
            title = re.sub(r"^\(TIE\)\s*", "", title)
            titles.append(title)
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="x.jsonl")
    parser.add_argument("--output", default="x_1.json")
    args = parser.parse_args()

    new_data_list = []

    with open(args.input, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            user_data = json.loads(line)
            user_id = user_data["user_id"]
            recommendation_subset = user_data["watched_movies_subset"]
            validation_set = user_data.get("validation_set", None)

            adjusted_titles = extract_titles(user_data["adjusted_recommendations"])
            second_adjusted_titles = extract_titles(user_data["second_adjusted_recommendations"])
            third_adjusted_titles = extract_titles(user_data["third_adjusted_recommendations"])

            new_user_data = {
                "user_id": user_id,
                "adjusted_recommendations": [
                    f"{i + 1}. {title}" for i, title in enumerate(adjusted_titles)
                ],
                "second_adjusted_recommendations": [
                    f"{i + 1}. {title}" for i, title in enumerate(second_adjusted_titles)
                ],
                "third_adjusted_recommendations": [
                    f"{i + 1}. {title}" for i, title in enumerate(third_adjusted_titles)
                ],
                "recommendation_subset": recommendation_subset,
                "validation_set": validation_set,
            }

            new_data_list.append(new_user_data)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(new_data_list, file, indent=4, ensure_ascii=False)

    print(f"Data extraction and saving completed. Output: {args.output}")


if __name__ == "__main__":
    main()

import openai
import json
import re
import numpy as np

openai.api_base = ""
openai.api_key = ""

def suggestMovie(watchedMoviesSubset, attempt=1, max_attempts=5):
    messages = [
        {"role": "system", "content": "You are a movie recommendation system. Recommend movies based on user preferences derived from previously watched movies. Focus on movies from 1880 to 2020 with ratings above 4."},
        {"role": "user", "content": "Can you recommend 10 movies for me? Rank them by how much I might like them, formatted as title(year)."}
    ]

    for index, movie in enumerate(watchedMoviesSubset):
        messages.append({"role": "system", "content": f"User watched: {movie['title']} directed by {movie['directedBy']} starring {movie['starring']} Rating {movie['rating']}."})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        recommendations = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSubset, attempt + 1)
        else:
            return ["Failed to get recommendations after several attempts."]

    return [line.strip() for line in recommendations if line.strip()]

def processBatch(file_path, output_path):
    with open(file_path, 'r') as file:
        users = [json.loads(line.strip()) for line in file]

    output_data = []

    for user in users:
        watched_movies_subset = user['History'][:len(user['History']) // 3]
        validation_set = [{"title": movie['title'], "rating": movie['rating']} for movie in user['History'][len(user['History']) // 3:]]
        recommendations = suggestMovie(watched_movies_subset)

        output_data.append({
            "user_id": user['user_id'],
            "watched_movies_subset": watched_movies_subset,
            "validation_set": validation_set,
            "recommendations": recommendations
        })

    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    file_path = 'user_movie_history_sample.jsonl'
    output_path = 'llm_movie_zero-shot.json'
    processBatch(file_path, output_path)

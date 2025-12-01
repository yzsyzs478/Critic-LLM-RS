import openai
import json
import re
import numpy as np

openai.api_base = ""
openai.api_key = ""

def suggestMovie(watchedMoviesSubset, attempt=1, max_attempts=5):
    messages = [
        {
            "role": "system",
            "content": "You are a movie recommendation system. Think step by step based on the title, directedBy, and starring of the movies that the user has watched in the viewing history, we obtain the user's preference, and then recommend movies for the user based on the user's preference. The recommended years for the movie are 1880 to 2020. And the recommended movie ratings should be greater than 4. Let's think step by step to find movies that the user might like, considering their watched history."
        },
        {
            "role": "user",
            "content": "Can you recommend 20 movies for me? And rank the movies in order of how much I might like them. The format of the recommended movies is title(year)."
        }
    ]

    for index, movie in enumerate(watchedMoviesSubset, start=1):
        messages.append({
            "role": "system",
            "content": f"User watched: title{index}: {movie.get('title', '')}. Directed by{index}: {movie.get('directedBy', '')}. Starring{index}: {movie.get('starring', '')}. Let's analyze what elements the user likes based on these movies."
        })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        gptSuggestion = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSubset, attempt + 1)
        else:
            return ["Failed to get recommendations after several attempts."]

    return [line for line in gptSuggestion if line.strip() and not line.startswith(
        "Considering the user's preferences, I recommend the following movies:")]

def clean_title(title):
    return re.sub(r'^\d+\.\s*', '', title).strip()

def processBatch(file_path, k_values=[10, 20, 50]):
    with open(file_path, 'r') as file:
        users = [json.loads(line.strip()) for line in file]

    results = []
    for user in users:
        watched_movies_subset = user['History'][:len(user['History']) // 3]
        validation_set = [{"title": movie['title'], "rating": movie['rating']} for movie in user['History'][len(user['History']) // 3:]]
        recommendations = suggestMovie(watched_movies_subset)
        user_data = {
            "user_id": user['user_id'],
            "watched_movies_subset": watched_movies_subset,
            "validation_set": validation_set,
            "recommendations": recommendations
        }
        results.append(user_data)

    with open('llm_movie_zero-shot_cot.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    file_path = 'user_movie_history_sample.jsonl'
    processBatch(file_path)

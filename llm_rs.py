import openai
import json
import re
import numpy as np

# OpenAI API setup
openai.api_base = ""
openai.api_key = ""
def suggestMovie(watchedMoviesSubset, attempt=1, max_attempts=5):
    """
    Simulates a call to the OpenAI API to generate movie recommendations
    based on a subset of watched movies.
    """
    template = "{title}"
    messages = [
        {"role": "system",
         "content": f"You are a movie recommendation system. Given a set of movies (each including the title, directedBy, starring) that a user watched, the user preference is based on the ratings that user give to these movies. Now, please recommend 10 movies based on the user's preferences, and the recommended years for the movie are 1880 to 2021. The format for the recommended movies should be {template}."},
        {"role": "user",
         "content": "Here are the movies that user watched and rated:\n" +
                    "\n".join([f"title: {movie['title']}, directedBy: {movie['directedBy']}, starring: {movie['starring']}, rating: {movie['rating']}" for movie in watchedMoviesSubset]) +
                    "\nPlease recommend to the user 10 movies and rank them according to how much the user might like them? Please base the ranking on a rating scale from 1 to 5, where 5 means I like them the most and 1 means I don't like them at all."}
    ]

    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1500)
        recommendations = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSubset, attempt + 1)
        else:
            return ["Failed to get recommendations after several attempts."]

    return [line.strip() for line in recommendations if line.strip()]

def processBatch(file_path, output_path):
    """
    Processes a batch of user data, generates recommendations, and saves the results.
    """
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
    file_path = 'user_movie_history.jsonl'  # Adjust the file path as necessary
    output_path = 'llm_nocritic.json'  # Output file path
    processBatch(file_path, output_path)

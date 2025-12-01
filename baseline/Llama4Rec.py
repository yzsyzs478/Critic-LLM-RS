import openai
import json
import re

openai.api_base = ""
openai.api_key = ""


def suggestMovie(watchedMoviesSubset, user_preferences, attempt=1, max_attempts=5):
    prompt = (
        "You are a movie recommendation system. "
        "Based on the movies the user has watched and the user's preferences in genres and directors, "
        "recommend 10 movies. The format of each recommendation should be: title (year). "
    )

    for movie in watchedMoviesSubset:
        prompt += (
            f"User watched: {movie['title']} directed by {movie['directedBy']} "
            f"starring {movie['starring']} rating {movie['rating']}. "
        )

    prompt += (
        f"User prefers genres: {user_preferences.get('genres')} "
        f"and movies from directors: {user_preferences.get('directors')}."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        gptSuggestion = response.choices[0].message.content.split('\n')
    except Exception as e:
        print(f"Error during API call: {e}")
        if attempt < max_attempts:
            return suggestMovie(watchedMoviesSubset, user_preferences, attempt + 1)
        else:
            return ["Failed to get recommendations after several attempts."]

    return [line.strip() for line in gptSuggestion if line.strip()]


def processBatch(file_path, output_path='llm_movie_llamarec.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        users = [json.loads(line.strip()) for line in file]

    results = []

    for user in users:
        user_results = {'user_id': user['user_id']}
        history = user['History']
        split_idx = len(history) // 3
        watched_movies_subset = history[:split_idx]
        validation_set = [movie['title'] for movie in history[split_idx:]]
        user_preferences = {
            'genres': user.get('preferred_genres'),
            'directors': user.get('preferred_directors')
        }
        recommendations = suggestMovie(watched_movies_subset, user_preferences)

        user_results.update({
            'watched_movies_subset': watched_movies_subset,
            'validation_set': validation_set,
            'recommendations': recommendations
        })
        results.append(user_results)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = 'user_movie_history_sample.jsonl'
    processBatch(file_path)

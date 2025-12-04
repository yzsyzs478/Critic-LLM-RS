# Critic-LLM-RS

## 1. Installation

We recommend creating a separate Conda environment before installing the dependencies:

```bash
# Create a new Conda environment
conda create -n critic-llm-rs python=3.10

# Activate the environment
conda activate critic-llm-rs

# Install required packages
python -m pip install -r requirements.txt
```

---

## 2. Repository Overview

This repository contains the core implementation of Critic-LLM-RS and its associated baselines and evaluation scripts. The main files and folders are organized as follows:

- `baseline/`  
  Contains the baseline recommendation models used for comparison with Critic-LLM-RS.

- `metric/`  
  Contains evaluation scripts for computing standard recommendation metrics (e.g., HR@K, NDCG@K, Precision@K).

- `critic_data_sample.json`  
  A small sample of training data for the Recommendation Critic, used as an example of the input format.

- `user_movie_history_sample.jsonl`  
  A small sample of evaluation data for Critic-LLM-RS, containing user–movie interaction histories.

- `movie.json`  
  A metadata file containing rich attribute information for a large collection of movies.
  
- `critic.py`  
  Training and optimization code for the Recommendation Critic model.

- `critic_llm_rs.py`  
  The main pipeline of Critic-LLM-RS, which uses the Recommendation Critic to provide feedback and adjust LLM-based recommendations.

- `requirements.txt`  
  The list of Python dependencies and their versions required to run the project.

---

## 3. Data Collection and Processing

We use two types of user–item interaction datasets for our experiments:

- **Movie domain:** MovieLens (Tag Genome 2021), downloaded from the official GroupLens website:  
  <https://grouplens.org/datasets/movielens/tag-genome-2021>
- **Book domain:** Book-Crossing (2022), obtained from:  
  <https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem>

### 3.1 Raw Fields

- **Movie dataset (MovieLens 2021)**  
  For each interaction, we extract:
  - `user_id`
  - `title`
  - `director`
  - `main actors`
  - `user rating`

- **Book dataset (Book-Crossing 2022)**  
  For each interaction, we extract:
  - `user_id`
  - `title`
  - `URL`
  - `authors`
  - `language`
  - `year published`
  - `book description`
  - `user rating`

### 3.2 Dataset Partitioning

After preprocessing, each domain-specific dataset is split into **three non-overlapping subsets**, with **no user overlap** among them:


#### (1) Recommendation Critic Training Set

This subset is used to train the **Recommendation Critic** model, and contains **30,000 users**, each with at least **10 item interactions**.

For each user, we first take their full interaction history and:

- Randomly select **one movie** as the **target item** to be predicted;
- Treat all remaining movies as the user’s **historical interactions**.

Each user is thus represented by **three components** in the training data:

1. **User viewing history**  
   A list of movies the user has watched, where each movie record includes  
   `title`, `directedBy`, `starring`, and `rating`.

2. **Target movie information**  
   The movie to be predicted, including its `title`, `directedBy`, and `starring`.

3. **Target movie rating**  
   The ground-truth `rating` of the target movie.

This design allows the Recommendation Critic to learn how to predict the rating of a target movie **conditioned on** the user’s viewing history, thereby modeling personalized preferences.

For model training, this subset is further split into:

- **Training / Validation / Test** = **7 : 2 : 1**

These splits are used for model fitting, hyperparameter tuning, and performance evaluation of the Recommendation Critic.

For illustration, we also provide a small de-identified sample file, critic_data_sample.json, which contains a subset of users from this Recommendation Critic training set and follows exactly the three-component format described above (history, target item, and target rating).

#### (2) Critic-LLM-RS Evaluation Set

This subset is used to evaluate the **Critic-LLM-RS** framework.

- From all users **excluding** those used in the Recommendation Critic training set,  
  we randomly sample **1,000 users** to form the evaluation set.
- This evaluation set is **strictly disjoint** from the Recommendation Critic training data at the **user level**, ensuring:
  - No user overlap  
  - Fair and independent evaluation of Critic-LLM-RS

In addition, we release a sample file, user_movie_history_sample.jsonl, which contains example user–item interaction histories from the Critic-LLM-RS evaluation pool and illustrates the exact input format used by our framework.

#### (3) Remaining Dataset

The **remaining users and interactions**, after removing those used in:

- the Recommendation Critic training set, and  
- the Critic-LLM-RS evaluation set,

are grouped into a **residual dataset**. This subset is **not** directly used in the main experiments, but can be reserved for:

- **Extended studies**, or  
- **External validation** in future work.


This partitioning strategy enforces a **strict separation** between:

- the Recommendation Critic training stage, and  
- the Critic-LLM-RS evaluation stage,

effectively preventing **information leakage** and improving the **fairness** and **reliability** of the evaluation for Critic-based recommendation models.

---


## 4. How to Run

Both our main script and the associated baseline scripts support using --help to list all available command-line arguments. For example:

```bash
python critic_llm_rs.py --help
```

Example (truncated) output:

```text
usage: critic-llm-rs.py [-h]
                        [--input INPUT]
                        [--movie_json MOVIE_JSON]
                        [--critic_ckpt CRITIC_CKPT]
                        [--output_dir OUTPUT_DIR]
                        [...]

Run Critic-LLM-RS with an LLM backend and a trained rating predictor.

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the user–movie interaction file (JSONL), e.g.,
                        user_movie_history_sample.jsonl.
  --movie_json MOVIE_JSON
                        Path to the movie metadata file (JSON or JSONL)
                        containing title/director/cast info.
  --critic_ckpt CRITIC_CKPT
                        Path to the trained rating predictor checkpoint (.pth).
  --output_dir OUTPUT_DIR
                        Directory where results, checkpoints, and logs will be saved.
  --api_key API_KEY     API key for the OpenAI-compatible LLM endpoint.
  --base_url BASE_URL   Base URL of the OpenAI-compatible LLM endpoint.
  [output omitted]
```

### Step 1: Train the Recommendation Critic

Run the following script to train the Recommendation Critic model:

```bash
python critic.py
```

This script loads the **Recommendation Critic training set**, splits it into train/validation/test (7:2:1), and trains the critic to predict user ratings.


### Step 2: Generate Critic-LLM-RS Recommendations

First, deploy an LLM locally **or** configure the corresponding API endpoint. In critic_llm_rs.py, we define three command-line arguments via parser.add_argument—--api_key, --base_url, and --llm_model_name—to configure the OpenAI-compatible LLM endpoint and model. Users can either modify the default values of these arguments directly in the code or explicitly pass them via the command line when running the script (e.g., python critic_llm_rs.py --api_key ...). All available arguments and their descriptions can be viewed using the --help flag.
Then run:

```bash
python critic_llm_rs.py
```

This script:

- Calls the LLM to generate **candidate recommendations** for each user.
- Optionally integrates the **Recommendation Critic** scores into the recommendation pipeline (depending on your configuration).
- Performs a minimal validation step to check whether each recommended title actually exists in the local database `movie.json`.

> **Note:**  
> - `movie.json` contains all movie entries collected in our experiments.  
> - If you prefer a more authoritative or comprehensive validator, you can replace this component with a third-party API (e.g., Google’s API or any online movie database API).


### Step 3: Evaluate Critic-LLM-RS Results

All evaluation scripts are located in the `metric/` directory and support two evaluation settings.


#### Option A: Real Rating–Only  
(*From candidate item list to recommendation*)

1. **Extract recommended titles from LLM outputs:**
   ```bash
   python extract_filter.py
   ```
   - Parses the LLM’s raw outputs  
   - Extracts recommended titles  
   - Cleans and deduplicates them  
   - Optionally retains only Top-K items per user

2. **Attach ground-truth ratings:**
   ```bash
   python real_rating.py
   ```
   - For items with available ground-truth ratings  
   - Records their true ratings for subsequent evaluation

3. **Compute Top-K metrics:**
   ```bash
   python metric.py
   ```
   - Produces **Top-10 / Top-5 / Top-3** recommendation lists  
   - Evaluates them with standard metrics (e.g., **HR@K**, **NDCG@K**, **Precision@K**, etc.)


#### Option B: Critic + Real  
(*Critic Rating + Real Rating for LLM-generated recommendations*)

1. **Extract & clean recommended titles:**
   ```bash
   python extract_filter.py
   ```

2. **Predict scores with Recommendation Critic:**
   ```bash
   python predict_rating.py
   ```
   - Completes related metadata based on the title (e.g., director, starring, etc.)  
   - Uses the **Recommendation Critic** to assign a predicted rating/score to each item

3. **Replace with real ratings when available:**
   ```bash
   python real_rating.py
   ```
   - For items with known ground-truth ratings  
   - Replaces the Critic’s predicted scores with their true ratings

4. **Evaluate with Top-K metrics:**
   ```bash
   python metric.py
   ```
   - Generates Top-10 / Top-5 / Top-3 lists based on the final scores  
   - Computes HR@K, NDCG@K, Precision@K, etc.

---

> **Note:**  
> Depending on your dataset format and experimental setting, some file paths or parameter configurations in these scripts may need to be adapted to your local environment.

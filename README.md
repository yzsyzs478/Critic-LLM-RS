Data Collection and Processing
## Data Collection and Processing

We collected two types of raw datasets from **Website https://grouplens.org/datasets/movielens/tag-genome-2021** and **Website https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem**, corresponding to **movies: MovieLens (2021)** and **books: Book-Crossing (2022) **, respectively. 

- For the **movie dataset**, the extracted fields include: `user_id`, `title`, `director`, `main actors`, and the corresponding `user rating`.
- For the **book dataset**, the extracted fields include: `user_id`, `title`, `URL`, `authors`, `language`, `year published`, `book description`, and `user rating`.

After preprocessing, each dataset was divided into **three non-overlapping subsets**, ensuring that there was **no user overlap** among them.

---

### (1) Recommendation Critic Training Set

This subset contains **30,000 users**, each with at least **10 item interactions**.  
For each user, **one item** is randomly selected from their interaction history as the **target item to be predicted**, while the remaining items are used as **historical interactions**.  
The model learns the user's personalized preference by predicting the rating of the target item based on their interaction history.

During model training, the data is further split into **training, validation, and test sets** with a ratio of **7:2:1** for learning, parameter tuning, and performance evaluation.

---

### (2) Critic-LLM-RS Evaluation Set

From all users **excluding** those used in the Recommendation Critic training data, **1,000 users** are randomly sampled to construct the evaluation dataset for **Critic-LLM-RS**.  
This evaluation set is **strictly separated** from the Recommendation Critic training data to ensure **no user overlap**, thereby maintaining the fairness and independence of model evaluation.

---

### (3) Remaining Dataset

The **remaining data**, excluding those used for Recommendation Critic training and Critic-LLM-RS evaluation, forms the **residual dataset**.  
This subset is not directly used for model training or evaluation but may serve for **future extended experiments** or **external validation**.

---

This data partitioning strategy ensures a **strict separation** between the Recommendation Critic training and Critic-LLM-RS evaluation stages, effectively preventing **information leakage**, and guaranteeing **fairness** and **reliability** in evaluating the Critic-based recommendation models.



## Environments

- python 3.9
- pytorch-2.0.1
- openai=1.55.0



## How to run
1) Train the Critic
python critic.py

2) Generate Critic-LLM-RS Recommendations
Deploy an LLM locally or configure the API, then run:
python critic-llm-rs.py

critic-llm-rs.py includes a minimal validation step that checks whether each recommended title actually exists by looking it up in an external database, movie.json. The file contains all movie entries we collected. If you prefer a more authoritative or comprehensive validator, you can replace this component with Google’s API (or any third-party movie database API).

3) Evaluate Critic-LLM-RS Results
Evaluation scripts are under the metric directory and support two settings:

Option A: Real Rating-Only (From Candidate item list to recommendation)
python tiqu_shaixuan.py  #Extracts each recommended Title from the LLM output and performs necessary cleaning/deduplication (optionally retaining Top-K).
python real_rating.py    #For items with ground-truth ratings, record (add) their true ratings so they can be used in subsequent evaluation.
python metric.py         #Produces Top-10/Top-5/Top-3 lists based on the final scores and evaluates them (e.g., HR@K, NDCG@K, Precision@K, etc.).

Option B: Critic + Real （Critic Rating + Real Rating (LLM generate recommendation)）
python tiqu_shaixuan.py  #Extracts each recommended Title from the LLM output and performs necessary cleaning/deduplication (optionally retaining Top-K).
python predict_rating.py #Completes related metadata based on the Title (e.g., director,starring, etc.) and generates a predicted score using the Recommendation Critic.
python real_rating.py    #For items with ground-truth ratings, replaces the Critic’s predicted score with the true rating.
python metric.py         #Produces Top-10/Top-5/Top-3 lists based on the final scores and evaluates them (e.g., HR@K, NDCG@K, Precision@K, etc.).


Note: Some parts of the code may need to be adjusted for your specific use case.

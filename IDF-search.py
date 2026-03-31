import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups # Keep for context if needed
from sklearn.utils import shuffle # <<< ADD THIS LINE
import os
import warnings

sns.set_theme(style="whitegrid")
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Libraries imported.")


print("Loading verified components for search and evaluation...")

# Paths to VERIFIED components
VECTORIZER_PATH = 'tfidf_vectorizer_verified.joblib'
TFIDF_MATRIX_PATH = 'tfidf_matrix_verified.joblib'
#TF_IGM_MATRIX_PATH = 'tf_igm_matrix_verified.joblib'
ALIGNED_TARGET_PATH = 'target_aligned.joblib'
ALIGNED_NAMES_PATH = 'target_names_aligned.joblib'
DATA_CACHE_PATH = "dataset" # For reloading texts

# Initialize variables
vectorizer = None
tfidf_matrix = None
#tf_igm_matrix = None
target = None
target_names = None
texts = None

# Load components
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    #tf_igm_matrix = joblib.load(TF_IGM_MATRIX_PATH)
    target = joblib.load(ALIGNED_TARGET_PATH) # Load ALIGNED targets
    target_names = joblib.load(ALIGNED_NAMES_PATH) # Load ALIGNED names
    print("Loaded vectorizer, matrices, targets, and names successfully.")
    LOAD_SUCCESS = True
except FileNotFoundError:
    print("Error: Could not find all necessary verified joblib files.")
    LOAD_SUCCESS = False
except Exception as e:
    print(f"An error occurred loading joblib files: {e}")
    LOAD_SUCCESS = False

# Reload texts and verify alignment ONE LAST TIME
if LOAD_SUCCESS:
     try:
        print(f"\nReloading and shuffling texts from cache '{DATA_CACHE_PATH}' for context...")
        newsgroups_data = fetch_20newsgroups(
            subset='all', remove=('headers', 'footers', 'quotes'),
            shuffle=True, random_state=42, data_home=DATA_CACHE_PATH
        )
        texts_shuffled, target_check = shuffle(
            newsgroups_data.data, newsgroups_data.target, random_state=42
        )
        texts = texts_shuffled # Use correctly shuffled texts

        if len(texts) != len(target) or not np.array_equal(target_check, target):
             print(f"CRITICAL ERROR: Reloaded texts/targets do not align with saved targets.")
             LOAD_SUCCESS = False
        else:
             print("Texts reloaded and aligned successfully.")
     except Exception as e:
        print(f"An error occurred reloading texts: {e}")
        LOAD_SUCCESS = False

if not LOAD_SUCCESS:
    print("\nFailed to load/align components. Cannot proceed with search/evaluation.")
else:
    print("\nComponents ready for search and evaluation.")


def search_tfidf(query, top_n=10):
    """
    Performs TF-IDF search on the loaded matrix.

    Args:
        query (str): The search query string.
        top_n (int): The number of top results to return.

    Returns:
        list: A list of tuples, where each tuple contains
              (document_index, similarity_score, document_text_snippet).
              Returns an empty list if components are not loaded.
    """
    if vectorizer is None or tfidf_matrix is None or texts is None:
        print("Error: TF-IDF components not loaded.")
        return []

    try:
        # 1. Transform the query using the loaded vectorizer
        query_vector = vectorizer.transform([query]) # Input must be iterable

        # 2. Calculate cosine similarity
        # Shape: (1, num_documents)
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        # 3. Get similarity scores for all documents
        # Flatten to shape: (num_documents,)
        scores = cosine_similarities.flatten()

        # 4. Get indices of top N documents (highest scores)
        # argsort sorts ascending, so we get the last N indices and reverse them
        # or sort negated scores
        # related_doc_indices_sorted = np.argsort(scores)[::-1] # Sort descending
        related_doc_indices_sorted = np.argsort(-scores) # Sort descending via negation

        # 5. Format results
        results = []
        for i in range(min(top_n, len(scores))): # Ensure we don't exceed available docs
            doc_index = related_doc_indices_sorted[i]
            score = scores[doc_index]
            # Provide a snippet of the document text
            text_snippet = texts[doc_index][:500] + "..." if len(texts[doc_index]) > 500 else texts[doc_index]
            results.append((doc_index, score, text_snippet))

        return results

    except Exception as e:
        print(f"An error occurred during search: {e}")
        return []

# Example usage (optional - can be run interactively later)
# test_query = "computer graphics card performance"
# search_results = search_tfidf(test_query, top_n=5)
# print(f"\nExample Search Results for: '{test_query}'")
# for idx, score, snippet in search_results:
#    print(f"  Index: {idx}, Score: {score:.4f}, Category: {target_names[target[idx]]}\n    Snippet: {snippet}\n")


print("\n--- Interactive Search ---")
print("Type your query or 'quit' to exit.")

while True:
    user_query = input("Enter search query: ")
    if user_query.lower() == 'quit':
        break
    if not user_query.strip():
        print("Please enter a query.")
        continue

    search_results = search_tfidf(user_query, top_n=5)

    if not search_results:
        print("Search could not be performed.")
        continue

    print(f"\n--- Top 5 Results for '{user_query}' ---")
    for i, (idx, score, snippet) in enumerate(search_results):
         # Check if target info is available
         category_name = "N/A"
         if target is not None and target_names is not None and idx < len(target):
              category_name = target_names[target[idx]]

         print(f"\n{i+1}. Index: {idx}, Score: {score:.4f}, Category: {category_name}")
         print(f"   Snippet: {snippet}")
    print("-" * 25)

print("Exiting interactive search.")


# --- Evaluation Setup ---

# Define test queries and their expected target newsgroup(s)
# This is a small sample, you might want to expand this list
EVALUATION_QUERIES = {
    "graphics card drivers algorithm performance": ["comp.graphics", "comp.sys.ibm.pc.hardware", "comp.os.ms-windows.misc"],
    "religious freedom versus atheism beliefs discussion": ["alt.atheism", "talk.religion.misc", "soc.religion.christian"],
    "NHL hockey playoffs scores results pens islanders": ["rec.sport.hockey"],
    "space shuttle mission nasa exploration launch": ["sci.space"],
    "President Clinton policy budget government": ["talk.politics.misc", "talk.politics.mideast", "talk.politics.guns"],
    "forsale encryption privacy security keys": ["sci.crypt", "misc.forsale"],
    "medical research cancer treatment study": ["sci.med"],
    "macintosh computer apple hardware software": ["comp.sys.mac.hardware"]
}

N_EVAL = 10 # Number of documents to retrieve for evaluation (Precision@10)

def calculate_precision_at_n(retrieved_indices, expected_categories):
    """
    Calculates Precision@N based on expected categories.

    Args:
        retrieved_indices (list): List of document indices retrieved.
        expected_categories (list): List of target category names expected for the query.

    Returns:
        float: Precision@N score. Returns 0.0 if components are missing.
    """
    if target is None or target_names is None:
        print("Error: Target labels/names not loaded. Cannot calculate P@N.")
        return 0.0
    if not retrieved_indices: # Handle empty results case
        return 0.0

    num_relevant = 0
    num_retrieved = len(retrieved_indices)

    for doc_index in retrieved_indices:
        if doc_index < len(target): # Ensure index is valid
            actual_category = target_names[target[doc_index]]
            if actual_category in expected_categories:
                num_relevant += 1
        else:
            print(f"Warning: Retrieved index {doc_index} out of bounds for target array.")

    return num_relevant / num_retrieved

print(f"Defined {len(EVALUATION_QUERIES)} evaluation queries.")
print(f"Evaluation will use Precision@{N_EVAL}.")


print("\n--- Running Evaluation ---")

all_precisions = []
query_results = {} # To store results per query

if vectorizer is None or tfidf_matrix is None or texts is None or target is None:
    print("Cannot run evaluation due to missing components.")
else:
    for query, expected_cats in EVALUATION_QUERIES.items():
        print(f"\nEvaluating query: '{query}' (Expected: {expected_cats})")

        # Perform search, retrieve only indices for evaluation
        search_results_eval = search_tfidf(query, top_n=N_EVAL)
        retrieved_indices = [idx for idx, score, snippet in search_results_eval]

        # Calculate P@N
        precision = calculate_precision_at_n(retrieved_indices, expected_cats)
        all_precisions.append(precision)
        query_results[query] = precision

        print(f"Retrieved indices: {retrieved_indices}")
        print(f"Precision@{N_EVAL}: {precision:.4f}")

    # Calculate Average Precision@N
    if all_precisions:
        average_precision = np.mean(all_precisions)
        print(f"\n---\nOverall Average Precision@{N_EVAL} (TF-IDF): {average_precision:.4f}")
    else:
        print("\nNo evaluation results obtained.")

# Optional: Print detailed results per query
# print("\nDetailed P@N Results per Query:")
# for q, p in query_results.items():
#    print(f"  {p:.4f} - {q}")



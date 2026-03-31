import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
print("Libraries imported.")


print("Fetching/Loading 20 Newsgroups data...")
DATA_CACHE_PATH = "dataset"
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

try:
    # Fetch the data - ensure these parameters are consistent everywhere
    newsgroups_data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True, # Shuffle initially is fine, but we'll re-shuffle below too
        random_state=42, # Consistent random state
        data_home=DATA_CACHE_PATH
    )

    # --- Critical Alignment Step ---
    # Re-shuffle the loaded data immediately to ensure consistent order
    # for both vectorization and target alignment. Use the *same* objects hereafter.
    texts_shuffled, target_shuffled = shuffle(
        newsgroups_data.data, newsgroups_data.target, random_state=42
    )
    target_names = newsgroups_data.target_names
    num_docs = len(texts_shuffled)
    num_classes = len(target_names)

    print(f"Dataset loaded and shuffled consistently.")
    print(f"Number of documents: {num_docs}")
    print(f"Number of classes: {num_classes}")
    LOAD_SUCCESS = True

except Exception as e:
    print(f"An error occurred loading data: {e}")
    texts_shuffled, target_shuffled, target_names = None, None, None
    LOAD_SUCCESS = False


vectorizer = None
tfidf_matrix = None

if LOAD_SUCCESS:
    print("\nCreating TF-IDF Vectorizer and Matrix...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 1)
    )

    # Fit and transform using the consistently shuffled texts
    tfidf_matrix = vectorizer.fit_transform(texts_shuffled)

    print("TF-IDF matrix created.")
    print(f"Shape: {tfidf_matrix.shape}")

    # --- Basic Check ---
    if tfidf_matrix.shape[0] != len(target_shuffled):
        print(f"CRITICAL ERROR: Matrix rows ({tfidf_matrix.shape[0]}) != Target length ({len(target_shuffled)})! Alignment failed.")
        tfidf_matrix = None # Invalidate matrix
    else:
        print("Matrix shape matches target length.")
else:
    print("Skipping TF-IDF creation due to loading errors.")


# --- Sanity Check ---
MIN_EXPECTED_ACCURACY = 0.75 # Set your threshold

if tfidf_matrix is not None and target_shuffled is not None:
    print(f"\nPerforming immediate classification sanity check (expecting >= {MIN_EXPECTED_ACCURACY*100:.1f}% accuracy)...")

    # Split the matrix and the *aligned* targets
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix,
        target_shuffled, # Use the target aligned with the matrix
        test_size=0.2,
        random_state=42,
        stratify=target_shuffled
    )

    # Train simple model
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nSanity Check Accuracy: {accuracy:.4f}")

    if accuracy >= MIN_EXPECTED_ACCURACY:
        print("Sanity Check PASSED: TF-IDF matrix appears reasonable for classification.")
        SANITY_CHECK_PASSED = True
    else:
        print(f"SANITY CHECK FAILED: Accuracy is below the expected threshold.")
        print("There might still be an issue with data loading, preprocessing, or alignment.")
        print("Do NOT proceed with using this matrix/vectorizer until resolved.")
        SANITY_CHECK_PASSED = False

    print("\nClassification Report (Sanity Check):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

else:
    print("Cannot perform sanity check due to missing matrix or targets.")
    SANITY_CHECK_PASSED = False


TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer_verified.joblib'
TFIDF_MATRIX_PATH = 'tfidf_matrix_verified.joblib'
ALIGNED_TARGET_PATH = 'target_aligned.joblib' # Save targets aligned with matrix
ALIGNED_NAMES_PATH = 'target_names_aligned.joblib' # Save target names

if SANITY_CHECK_PASSED:
    print("\nSaving verified TF-IDF components and aligned targets...")
    try:
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
        joblib.dump(target_shuffled, ALIGNED_TARGET_PATH) # Save the aligned target array
        joblib.dump(target_names, ALIGNED_NAMES_PATH) # Save the names list
        print("Components saved successfully:")
        print(f"  - {TFIDF_VECTORIZER_PATH}")
        print(f"  - {TFIDF_MATRIX_PATH}")
        print(f"  - {ALIGNED_TARGET_PATH}")
        print(f"  - {ALIGNED_NAMES_PATH}")
    except Exception as e:
        print(f"An error occurred saving components: {e}")
else:
    print("\nSanity check failed. Components were NOT saved.")



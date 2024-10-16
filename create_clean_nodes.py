import json
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.sparse import csr_matrix
import pandas as pd
import pickle
import argparse
import os

# Load the small English model for spaCy
nlp = spacy.load("en_core_web_sm")

def remove_mistakes(phrases, stop_words):
    """
    Remove invalid phrases based on spaCy's lemmatizer, stop words, and numerical values.
    
    Args:
        phrases (set): A set of phrases to process.
        stop_words (set): A set of stopwords to filter out.

    Returns:
        set: Cleaned set of phrases.
        dict: Dictionary mapping original phrases to their cleaned versions.
    """
    clean_phrases = set()
    phrase_map = {}

    for phrase in tqdm(phrases):
        if phrase.isalpha():  # Process only alphabetic phrases
            doc = nlp(phrase)
            lemmatized_word = doc[0].lemma_  # Get the lemma of the first word
            if lemmatized_word in stop_words or is_number(lemmatized_word) or len(lemmatized_word) < 2 or doc[0].pos_ == "VERB":
                continue
            clean_phrases.add(lemmatized_word)
            phrase_map[phrase] = lemmatized_word.strip() 
        else:
            clean_phrases.add(phrase.strip())
    
    return clean_phrases, phrase_map

def is_number(n):
    """
    Check if a string is numeric.

    Args:
        n (str): Input string to check.

    Returns:
        bool: True if the string represents a number, False otherwise.
    """
    try:
        float(n)
        return True
    except ValueError:
        return False

def clean_phrase(phrase, stop_words):
    """
    Cleans up phrases by removing leading stopwords or numbers, but keeps valid components
    in the middle of the phrase (e.g., "22 of these").
    
    Args:
        phrase (str): The original phrase to clean.
        stop_words (set): A set of stopwords to filter out.

    Returns:
        str: The cleaned phrase.
    """
    words = phrase.split(" ")
    cleaned_words = []
    leading_removed = False  # Flag to track if leading words have been removed

    for word in words:
        if not leading_removed:
            if word in stop_words or is_number(word):
                continue  # Skip leading stopwords or numbers
            else:
                leading_removed = True  # Start adding words after the first valid word
        cleaned_words.append(word)  # Keep the rest of the phrase as it is

    return " ".join(cleaned_words)

def prep(file_name):
    """
    Main function to prepare and clean phrases from a file. It removes stopwords, irrelevant words, 
    and parser mistakes.

    Returns:
        list: List of cleaned phrases.
        list: List of lines with valid phrase IDs.
        dict: Dictionary mapping original phrases to their cleaned version.
    """
    # Load stop words
    stop_words = set(stopwords.words('english'))

    # Initialize sets and dictionaries
    all_phrases = set()
    phrase_to_clean = {}

    # Read the file and clean phrases
    with open(file_name, encoding="utf8") as fin:
        for line in tqdm(fin):
            phrases = []
            for w in line.strip().split('\t'):
                origin_w = w.lower()  # Convert word to lowercase
                
                # Clean the phrase by removing leading stopwords/numbers
                cleaned_w = clean_phrase(origin_w, stop_words)

                if cleaned_w and len(cleaned_w) > 1:
                    phrases.append(cleaned_w)
                phrase_to_clean[w] = cleaned_w

            all_phrases.update(set(phrases))
    
    # Clean phrases by removing stop words, parser mistakes, and irrelevant words
    all_phrases, phrase_map = remove_mistakes(all_phrases, stop_words)

    # Create phrase-to-ID mapping
    id2phrase = list(all_phrases)
    phrase2id = {phrase: idx for idx, phrase in enumerate(id2phrase)}

    # Process file lines again to create lines with valid phrase IDs
    all_lines = []
    with open(file_name, encoding="utf8") as fin:
        for line in tqdm(fin):
            phrase_ids = []
            for phrase in line.strip().split('\t'):
                phrase_lower = phrase.lower()
                
                cleaned_phrase = clean_phrase(phrase_lower, stop_words)

                if cleaned_phrase in all_phrases:
                    phrase_ids.append(phrase2id[cleaned_phrase])
                elif cleaned_phrase in phrase_map and phrase_map[cleaned_phrase] in phrase2id:
                    phrase_ids.append(phrase2id[phrase_map[cleaned_phrase]])

            if len(set(phrase_ids)) > 1:
                all_lines.append(phrase_ids)

    return id2phrase, all_lines, phrase_to_clean

def sparse_and_id2phrase(id2phrase, all_lines):
    """
    Constructs a weight matrix based on co-occurrence of phrases in the given lines.

    Parameters:
    - id2phrase: List of phrase identifiers.
    - all_lines: List of lines, where each line contains a sequence of phrases (represented by ids).

    Returns:
    - data: List of weights corresponding to each (row, col) pair in the matrix.
    - row: List of row indices corresponding to the first element of each phrase pair.
    - col: List of column indices corresponding to the second element of each phrase pair.
    - n: The total number of phrases (length of id2phrase).
    - id2phrase: List of phrases in lowercase.
    """

    # Iterate through each line and create co-occurrence pairs
    weight_matrix = defaultdict(int)
    for line in all_lines:
        for a, b in combinations(line, 2):
            weight_matrix[(a,b)] += 1
            weight_matrix[(b,a)] += 1

    # Convert all phrases to lowercase
    id2phrase = [i.lower() for i in id2phrase]

    # Unpack the dictionary keys and values into row, col, and data lists
    row, col = zip(*weight_matrix.keys())
    data = list(weight_matrix.values())


    n = len(id2phrase)
    return data, row, col, n, id2phrase

def sparse_pow( data, row, col, n, p):
    """
    Generates a sparse matrix with powered data values.

    Parameters:
    - data: List of weights for each non-zero element in the matrix.
    - row: List of row indices.
    - col: List of column indices.
    - n: The size of the square matrix.
    - p: The power to which the data values should be raised.

    Returns:
    - X: A compressed sparse row (CSR) matrix where data values are raised to the power of p.
    """

    X = csr_matrix((list(np.power(data,p)), (row, col)), shape=(n, n))
    return X


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, required=True)

args = parser.parse_args()
file_name: str = args.file_path

# Run the prep function
id2phrase, all_lines, phrase2clean = prep(file_name)

data_path = "data"
if not os.path.exists(data_path):
    os.makedirs(data_path)    
    
# Save the phrase2clean dictionary to a pickle file for later use
with open(os.path.join(data_path, 'phrase2clean.pkl'), 'wb') as fp:
    pickle.dump(phrase2clean, fp)

# Generate data and matrix
data, row, col, n, id2phrase = sparse_and_id2phrase(id2phrase, all_lines)

# Set the power value
Pow = 0.5

# Create the sparse matrix
X = sparse_pow( data, row, col, n, Pow)

# Extract the non-zero nodes from the sparse matrix
Node1 = [id2phrase[id] for id in X.nonzero()[0]] # List of nodes from the first dimension
Node2 = [id2phrase[id] for id in X.nonzero()[1]] # List of nodes from the second dimension

# Extract the weights from the sparse matrix and convert them to integers for better readability
weights = [float(int(weight)) for weight in X.data]
weighted_pairs = pd.DataFrame({"Node1":Node1,"Node2":Node2,"Weight":weights})

# Create a DataFrame of node pairs and their associated weights
weighted_pairs.to_csv(os.path.join(data_path, "weighted_pairs.csv"), index=False)
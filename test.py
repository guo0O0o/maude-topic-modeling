'''from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_biobert_embeddings(text, tokenizer, model):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # The embeddings are in `outputs.last_hidden_state`
    # [batch_size, seq_len, hidden_dim]
    return outputs.last_hidden_state


def get_word_embedding(text, word, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.tokenize(text)
    word_tokens = tokenizer.tokenize(word)
    word_indices = [i for i, token in enumerate(tokens) if token in word_tokens]

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.squeeze(0)
    # Aggregate embeddings for the word (e.g., if split into subwords)
    word_embedding = torch.mean(embeddings[word_indices], dim=0)
    return word_embedding



def calculate_similarity(embedding1, embedding2):
    # Convert embeddings to numpy
    emb1 = embedding1.numpy().reshape(1, -1)
    emb2 = embedding2.numpy().reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]'''
'''
import gensim
# https://github.com/ncbi-nlp/BioSentVec
# Embeddings using PubMed and the clinical notes from MIMIC-III Clinical Database

word2vec = gensim.models.KeyedVectors.load_word2vec_format(
     'BioWordVec_PubMed_MIMICIII_d200.vec.bin',
      binary=True,
      # limit=None, # this fuckin thing has 4 billion tokens (4E9)
      limit=int(4E7) # faster load if you limit to most frequent terms
)

# Define words to compare
word1 = "severe bleeding"
word2 = "intermenstrual bleeding"

# Check if both words are in the vocabulary
if word1 in word2vec and word2 in word2vec:
    similarity = word2vec.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One or both words not in vocabulary: '{word1}', '{word2}'")

word1 = "severe bleeding"
word2 = "hemorrhage/blood loss/bleeding"
if word1 in word2vec and word2 in word2vec:
    similarity = word2vec.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One or both words not in vocabulary: '{word1}', '{word2}'")'''

import numpy as np
def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

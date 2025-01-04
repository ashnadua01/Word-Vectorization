import pandas as pd
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch
import numpy as np
from scipy.sparse.linalg import svds
from collections import defaultdict
from scipy.sparse import lil_matrix
import argparse
from model import preprocess_text, build_vocab

# nltk.download('punkt')

class SVD_Embeddings:
    def __init__(self, corpus, vocab_index, window_size=1):
        self.corpus = corpus
        self.vocab_index = vocab_index
        self.window_size = window_size

    def build_co_occurrence_matrix(self):
        self.vocab_size = len(self.vocab_index)
        co_occurrence_counts = defaultdict(float)

        for doc in self.corpus:
            doc_len = len(doc)
            for i, word in enumerate(doc):
                for j in range(i + 1, min(i + self.window_size + 1, doc_len)):
                    co_occurrence_counts[(self.vocab_index[word], self.vocab_index[doc[j]])] += 1

        self.co_occurrence_matrix = lil_matrix((self.vocab_size, self.vocab_size), dtype=np.float32)
        for (word_i, word_j), count in co_occurrence_counts.items():
            self.co_occurrence_matrix[word_i, word_j] = count

        return self.co_occurrence_matrix

    def svd_process(self, embedding_dims, saveFlag):
        self.u, self.s, self.vt = svds(self.co_occurrence_matrix, k=100)
        self.word_vectors = self.u

        norms = np.linalg.norm(self.word_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8

        word_vectors_normalized = self.word_vectors / norms
        self.word_vectors_dict = {word: vector.tolist() for word, vector in zip(self.vocab_index, word_vectors_normalized)}

        if saveFlag == 1:
            self.save_embeddings()

        return self.word_vectors_dict

    def save_embeddings(self):
        torch.save(self.word_vectors_dict, './models/svd-word-vectors.pt')
        
    def load_embeddings(self, embeddings_path):
        self.word_vectors_dict = torch.load(embeddings_path)
        return self.word_vectors_dict

def create_embeddings(train_data_description, vocab_index, embedding_dim=100, window_size=5, save_flag=0):
    svd = SVD_Embeddings(train_data_description, vocab_index, window_size)
    co_matrix = svd.build_co_occurrence_matrix()
    print("\n----------Co-occurrence matrix created----------")
    word_vectors_dict_svd = svd.svd_process(embedding_dim, save_flag)
    print("----------SVD Process finished----------")
    return word_vectors_dict_svd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVD Embeddings Training')
    parser.add_argument('--e', type=int, default=100, help='Embedding dimension (default: 100)')
    parser.add_argument('--w', type=int, default=5, help='Window size (default: 5)')
    parser.add_argument('--s', type=int, default=0, help='Flag to save embeddings (default: 0)')
    args = parser.parse_args()

    embedding_dim = args.e
    window_size = args.w
    save_flag = args.s

    train_data = 'dataset/train.csv'
    test_data = 'dataset/test.csv'

    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)

    train_data_description = df_train["Description"].tolist()[:20000]
    train_data_description = [preprocess_text(desc) for desc in train_data_description]

    test_data_description = df_test["Description"].tolist()
    test_data_description = [preprocess_text(desc) for desc in test_data_description]

    vocab_index = build_vocab(train_data_description)

    labels = labels = df_train["Class Index"].tolist()[:20000]
    labels_test = df_test["Class Index"].tolist()

    # torch.save(vocab_index, "vocab.pt")
    word_vectors_dict_svd = create_embeddings(train_data_description, vocab_index, embedding_dim, window_size, save_flag)
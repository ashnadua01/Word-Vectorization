import pandas as pd
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from scipy.sparse.linalg import svds
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from scipy.sparse import lil_matrix
from model import EmbeddingTaskDataset, train_model, evaluate_model, LSTMClassifier
from svd import create_embeddings, preprocess_text, build_vocab
import torch.nn.functional as F
import argparse

# nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data():
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
    
    return train_data_description, test_data_description, vocab_index, labels, labels_test

if __name__ == "__main__":
    train_data_description, test_data_description, vocab_index, labels, labels_test = prepare_data()
    parser = argparse.ArgumentParser(description='SVD Classification')

    parser.add_argument('--use_model', type=int, default=0, help="Use pre-trained model")
    parser.add_argument('--use_pretrained', type=int, default=0, help="Use pre-trained embeddings")
    parser.add_argument('--e', type=int, default=100, help='Embedding dimension (default: 100)')
    parser.add_argument('--w', type=int, default=5, help='Window size (default: 5)')
    parser.add_argument('--s', type=int, default=0, help='Flag to save embeddings (default: 0)')
    parser.add_argument('--save_model', type=int, default=0, help='Flag to save model (default: 0)')

    args = parser.parse_args()

    if args.use_model == 1:
        input_dim = len(vocab_index)
        embedding_dim = 100
        hidden_dim = 128
        output_dim = 4
        batch_size = 128
        num_epochs = 10
        typeFlag = "svd"
        
        word_vectors_dict_svd = torch.load("./models/svd-word-vectors.pt")
        word_vectors_svd = np.array(list(word_vectors_dict_svd.values()))
        word_vectors_svd_tensor = torch.tensor(word_vectors_svd, dtype=torch.float32)

        test_dataset = EmbeddingTaskDataset(test_data_description, vocab_index, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

        model = LSTMClassifier(input_dim, embedding_dim, hidden_dim, output_dim, word_vectors_svd_tensor)
        model.load_state_dict(torch.load("./models/svd-classification-model.pt", map_location=torch.device('cpu')))

        evaluate_model(model, test_loader)
        exit()

    if args.use_pretrained == 1:
        word_vectors_dict_svd = torch.load("./models/svd-word-vectors.pt")
        print("----------Embeddings Loaded----------")
    else:
        embedding_dim = args.e
        window_size = args.w
        save_flag = args.s

        word_vectors_dict_svd = create_embeddings(train_data_description, vocab_index, embedding_dim, window_size, save_flag)
        print("----------Embeddings Created----------")
    
    saveModel = args.save_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_vectors_svd = np.array(list(word_vectors_dict_svd.values()))
    word_vectors_svd_tensor = torch.tensor(word_vectors_svd, dtype=torch.float32)

    input_dim = len(vocab_index)
    embedding_dim = len(word_vectors_svd[0])
    hidden_dim = 128
    output_dim = 4
    batch_size = 128
    num_epochs = 10
    typeFlag = "svd"

    train_dataset = EmbeddingTaskDataset(train_data_description, vocab_index, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    test_dataset = EmbeddingTaskDataset(test_data_description, vocab_index, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

    model = train_model(typeFlag, train_loader, word_vectors_svd_tensor, input_dim, output_dim, embedding_dim, hidden_dim, batch_size, num_epochs, saveModel)
    evaluate_model(model, test_loader)

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
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import EmbeddingTaskDataset, LSTMClassifier
import torch.nn.functional as F
import argparse
from model import preprocess_text, build_vocab, train_model, evaluate_model
import argparse
from skip_gram import create_embeddings_skipgram, Word2Vec_Embeddings

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

    train_data_indices = [[vocab_index[word] for word in doc] for doc in train_data_description]
    
    return train_data_description, test_data_description, train_data_indices, vocab_index, labels, labels_test

if __name__ == "__main__":
    train_data_description, test_data_description, train_data_indices, vocab_index, labels, labels_test = prepare_data()
    parser = argparse.ArgumentParser(description='Skip Gram Classification')

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
        
        word_vectors_dict = torch.load("./models/skip-gram-word-vectors.pt")
        word_vectors_sg = np.array(list(word_vectors_dict.values()))
        word_vectors_sg_tensor = torch.tensor(word_vectors_sg, dtype=torch.float32)

        test_dataset = EmbeddingTaskDataset(test_data_description, vocab_index, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

        model = LSTMClassifier(input_dim, embedding_dim, hidden_dim, output_dim, word_vectors_sg_tensor)
        model.load_state_dict(torch.load("./models/skip-gram-classification-model.pt", map_location=torch.device('cpu')))

        evaluate_model(model, test_loader)
        exit()

    if args.use_pretrained:
        word_vectors_dict_w2v = torch.load("./models/skip-gram-word-vectors.pt")
        print("----------Embeddings Loaded----------")
    else:
        embedding_dim = args.e
        window_size = args.w
        save_flag = args.s

        word_vectors_dict_w2v = create_embeddings_skipgram(train_data_indices, vocab_index, embedding_dim, window_size, save_flag)
        print("----------Embeddings Created----------")
    
    saveModel = args.save_model

    word_vectors_skipgram = np.array(list(word_vectors_dict_w2v.values()))
    word_vectors_sg_tensor = torch.tensor(word_vectors_skipgram, dtype=torch.float32)

    input_dim = len(vocab_index)
    embedding_dim = len(word_vectors_skipgram[0])
    hidden_dim = 128
    output_dim = 4
    batch_size = 128
    num_epochs = 10
    typeFlag = "w2v"

    train_dataset = EmbeddingTaskDataset(train_data_description, vocab_index, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    test_dataset = EmbeddingTaskDataset(test_data_description, vocab_index, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

    model = train_model(typeFlag, train_loader, word_vectors_sg_tensor, input_dim, output_dim, embedding_dim, hidden_dim, batch_size, num_epochs, saveModel)
    evaluate_model(model, test_loader)

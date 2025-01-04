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
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    return ['<START>'] + tokens + ['<END>']

def build_vocab(corpus):
    vocab = set()
    for doc in corpus:
        for word in doc:
            vocab.add(word)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    return vocab_index

class EmbeddingTaskDataset(Dataset):
    def __init__(self, data, vocab_index, labels):
        self.data = data
        self.vocab_index = vocab_index
        self.labels = [label - 1 for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def collate_fn(self, batch):
        batch_x = []
        for x, _ in batch:
            indices = [self.vocab_index[word] for word in x if word in self.vocab_index]
            batch_x.append(torch.tensor(indices))
        padded_batch_x = pad_sequence(batch_x, batch_first=True, padding_value=0)

        batch_y = torch.tensor([y for _, y in batch])
        return padded_batch_x, batch_y
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, embeddings):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)

        mask = (text != 0).unsqueeze(2)
        masked_output = output * mask

        sum_hidden = torch.sum(masked_output, dim=1)
        lengths = (text != 0).sum(dim=1).unsqueeze(1).float()
        mean_hidden = sum_hidden / lengths

        output = F.tanh(self.fc(mean_hidden))
        return output
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    @classmethod
    def load_model(cls, path, *args):
        model = cls(*args)
        model.load_state_dict(torch.load(path))
        return model

    def predict(self, sentence):
        self.eval()
        with torch.no_grad():
            outputs = self(sentence)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_tags = torch.max(probabilities, dim=1)
        return predicted_tags
    
def train_model(typeFlag, train_loader, tensor, input_dim, output_dim=4, embedding_dim=100, hidden_dim=128, batch=128, num_epochs=1, saveFlag=0):
    model = LSTMClassifier(input_dim, embedding_dim, hidden_dim, output_dim, tensor)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    model.train()
    print("\n----------Starting Training----------")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    print("----------Finished Training----------")

    if saveFlag == 1:
        if typeFlag == "svd":
            model.save_model("./models/svd-classification-model.pt")
        elif typeFlag == "w2v":
            model.save_model("./models/skip-gram-classification-model.pt")

    return model

def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(device)
            labels = labels.to(device)

            predicted = model.predict(texts)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    f1_macro = f1_score(all_labels, all_predictions, average='micro')

    print("\n----------Evaluation----------")
    print(f'Accuracy on test data: {accuracy * 100}')
    print(f'Precision on test data: {precision * 100}')
    print(f'Recall on test data: {recall * 100}')
    print(f'F1 micro score on test data: {f1_micro * 100}')
    print(f'F1 macro score on test data: {f1_macro * 100}')
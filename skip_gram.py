import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import argparse
from model import preprocess_text, build_vocab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# nltk.download('punkt')

def generate_skipgrams(sentence, window_size):
    pairs = []
    for i, target_word in enumerate(sentence):
        context_indices = list(range(max(0, i - window_size), i)) + list(range(i+1, min(len(sentence), i + window_size + 1)))
        for context_word in context_indices:
            pairs.append((target_word, sentence[context_word]))

    return pairs

class SkipGramDataset(Dataset):
    def __init__(self, data_indices, window_size=5):
        self.data = []
        for sentence in data_indices:
            self.data.extend(generate_skipgrams(sentence, window_size))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
class Word2Vec_Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec_Embeddings, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, target_word, context_word, negative_words):
        target_embed = self.embeddings(target_word)
        context_embed = self.out_embeddings(context_word)
        neg_embed = self.out_embeddings(negative_words)
        positive_score = torch.sum(torch.mul(target_embed, context_embed), dim=1)
        negative_score = torch.sum(torch.mul(target_embed.unsqueeze(1), neg_embed), dim=2)
        return positive_score, negative_score

    def train(self, train_data_indices, num_epochs=10, batch_size=128, learning_rate=0.001, negative_samples=5, window = 5):
        print("----------Training Skip Gram Model----------")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_dataset = SkipGramDataset(train_data_indices, window)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0
            for target_word, context_word in train_loader:
                optimizer.zero_grad()
                target_word, context_word = target_word.to(self.device), context_word.to(self.device)
                negative_words = torch.randint(0, self.vocab_size, (context_word.size(0), negative_samples), device=self.device)
                positive_score, negative_score = self(target_word, context_word, negative_words)

                positive_score = positive_score.view(-1, 1)
                negative_score = negative_score.view(-1, 1)

                targets = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)], dim=0)
                outputs = torch.cat([positive_score, negative_score], dim=0)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

        print("----------Finished Training----------")

    def get_embeddings(self, saveFlag, vocab_index):
        embeddings = self.embeddings.weight.data.cpu().numpy()
        embeddings = self.normalize_embeddings(embeddings)
        self.word_vectors_dict = {word: embeddings[index] for word, index in vocab_index.items()}

        if saveFlag == 1:
            self.save_embeddings()

        return self.word_vectors_dict

    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings

    def save_embeddings(self):
        torch.save(self.word_vectors_dict, "./models/skip-gram-word-vectors.pt")

    def load_embeddings(self, embeddings_path):
        self.word_vectors_dict = torch.load(embeddings_path)

def create_embeddings_skipgram(train_data_indices, vocab_index, embedding_dim=100, window=5, save_flag=0):
    w2v = Word2Vec_Embeddings(len(vocab_index), embedding_dim)
    w2v.train(train_data_indices, num_epochs=10, batch_size=128, learning_rate=0.001, negative_samples=5, window=window)
    print("\n----------Creating embeddings----------")
    word_vectors_dict_w2v = w2v.get_embeddings(save_flag, vocab_index)
    print("----------Skip Gram Process finished----------")
    if save_flag == 1:
        w2v.save_embeddings()

    return word_vectors_dict_w2v

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skip Gram Embeddings Training')
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

    train_data_indices = [[vocab_index[word] for word in doc] for doc in train_data_description]

    # torch.save(vocab_index, "vocab.pt")
    word_vectors_dict_skipgram = create_embeddings_skipgram(train_data_indices, vocab_index, embedding_dim, window_size, save_flag)
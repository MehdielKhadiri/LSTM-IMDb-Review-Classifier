import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import vocab
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(torch.tensor(text))
        # Explicitly convert the label to a float
        label_list.append(torch.tensor([float(label)], dtype=torch.float32))
    return pad_sequence(text_list, batch_first=True, padding_value=final_vocab['<pad>']), torch.tensor(label_list).view(-1)


epochs = 10
batch_size = 64
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
# Set the vocab_size
vocab_size = 25000

# Create tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load the IMDb dataset
train_data, test_data = IMDB(split=('train', 'test'))
train_data = to_map_style_dataset(train_data)
test_data = to_map_style_dataset(test_data)

# Split the data into train and validation sets
train_len = int(len(train_data) * 0.8)
valid_len = len(train_data) - train_len
train_data, valid_data = random_split(train_data, [train_len, valid_len])

# Build the vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

from collections import Counter
from itertools import chain

counter = Counter()
for tokens in yield_tokens(train_data):
    counter.update(tokens)

most_common_tokens = [token for token, _ in counter.most_common(vocab_size)]
sorted_counter = counter.most_common(vocab_size - 2)
sorted_tokens = ['<unk>', '<pad>'] + [token for token, _ in sorted_counter]
final_vocab = vocab(OrderedDict([(token, 1) for token in sorted_tokens]))

def encode(example):
    text, label = example
    stoi = final_vocab.get_stoi()
    # Convert string labels to numerical values
    label = 1.0 if label == 'pos' else 0.0
    return [stoi[token] if token in stoi else stoi['<unk>'] for token in tokenizer(str(text))], label





# Encode datasets
train_data = [encode(example) for example in train_data]
valid_data = [encode(example) for example in valid_data]
test_data = [encode(example) for example in test_data]

# Create data iterators
train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_iterator = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_batch)
test_iterator = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_batch)
# Instantiate the model
input_size = vocab_size
hidden_size = 256
num_layers = 2
output_size = 1
embedding_dim = 100  # Add this line
model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout_rate=0.5).to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(epochs):
    # Training
    train_losses = []
    train_accuracies = []
    model.train()
    for batch in train_iterator:
        optimizer.zero_grad()
        text, labels = batch  # Remove text_lengths
        text = text.to(device)
        labels = labels.to(device)
        predictions = model(text).squeeze(1)  # Remove text_lengths
        loss = criterion(predictions, labels)  # Replace batch.label with labels
        acc = binary_accuracy(predictions, labels)  # Replace batch.label with labels
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accuracies.append(acc.item())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accuracies) / len(train_accuracies)

    # Validation
    valid_losses = []
    valid_accuracies = []
    model.eval()
    with torch.no_grad():
        for batch in valid_iterator:
            text, labels = batch  # Remove text_lengths
            text = text.to(device)
            labels = labels.to(device)
            predictions = model(text).squeeze(1)  # Remove text_lengths
            loss = criterion(predictions, labels)  # Replace batch.label with labels
            acc = binary_accuracy(predictions, labels)  # Replace batch.label with labels
            valid_losses.append(loss.item())
            valid_accuracies.append(acc.item())

    valid_loss = sum(valid_losses) / len(valid_losses)
    valid_acc = sum(valid_accuracies) / len(valid_accuracies)

    print(f'Epoch: {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
# Save the model
model_path = 'textclass.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
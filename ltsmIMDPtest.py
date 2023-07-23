import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import vocab
from collections import OrderedDict

# Load the IMDb dataset
train_data, test_data = IMDB(split=('train', 'test'))
train_data = to_map_style_dataset(train_data)
test_data = to_map_style_dataset(test_data)

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

# Instantiate the model
input_size = vocab_size
hidden_size = 256
num_layers = 2
output_size = 1
embedding_dim = 100  # Add this line
model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout_rate=0.5)

# Load the saved model
model_path = 'textclass.pth'
model.load_state_dict(torch.load(model_path))

# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def predict_sentiment(model, tokenizer, vocab, sentence):
    model.eval()
    tokens = tokenizer(sentence)
    indexed = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

# Example usage
sentence = """
While the Wick franchise has already solidified itself one of the best in the action genre, this massive film is the most spectacular entry of the genre in 30 years.
Stuntman turned writer/director Chad Stahelski struck gold with his 2014 surprise hit John Wick. It was somewhat of a comeback for the legendary Keanu Reeves and reinvigorated the action genre. Since then it's become THE action juggernaut franchise. Now we are on number 4 and while usually things get redone with that amount of sequels, this film innovates and thrills to new heights in an absolute epic.

One of the most surprising aspects of this particular entry, is the story being written as well as it is. Something that is hard to come by in the genre at times. Not saying the others in the franchise weren't but this is easily the best story diving deeper in the high table aspects as well as John Wick's true emotions . The rich characterization is also at its best with outstanding additions like Skaarsgard, and even more screen time for supporting greats like Fishburne to compliment the magnificent Reeves. RIP Lance Reddick. It's nice to have such substance amongst the endless high octane ballistic visuals.

Speaking of ballistic, this movie goes more all out than any I've seen for insanely well crafted choreographed shootout and fight scenes. Just when you thought you've seen it all, Stahelski/Reeves prove their action minds are ever evolving. These guys were born to make action movies together.

The extremely vibrant colors and plethora of locations is also a feast for the eyes. The sharp atmospheric imagery creates the ultimate backdrop for not only the action but also just the dialogue. The sound is also top notch and perfectly compliments the intensity. It's just an absolute sensory journey that you don't get in too many action films aside from the Wick genre.

Overall the limitless and outrageous action alone is nothing like we've seen before for the genre or just cinema in general . But also this rich unique story may only build Wick's timeless character for more films in the future which at this momentum will be welcomed."""
prediction = predict_sentiment(model, tokenizer, final_vocab, sentence)
sentiment = "Positive" if prediction >= 0.5 else "Negative"
print(f"Review: {sentence}\nSentiment: {sentiment}")
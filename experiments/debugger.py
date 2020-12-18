import torch
import optuna
import numpy as np
from mlearn import base
from tqdm import tqdm, trange
from mlearn.utils.metrics import Metrics
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_singletask_model as run_model
from mlearn.modeling.embedding import MLPClassifier, CNNClassifier
from mlearn.utils.pipeline import process_and_batch, param_selection
from mlearn.data.batching import TorchtextExtractor
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator

# Initialize experiment
datadir =  'data/json'
torch.random.manual_seed( 42)
np.random.seed(42)
encoding =  'index'
tokenizer = 'ekphrasis'
metrics =  ['f1-score', 'precision', 'recall', 'accuracy']
cleaners = ['username', 'url', 'hashtag', 'lower']
display_metric = stop_metric = 'f1-score'
batch_size =  64
epochs =  200
learning_rate =  0.01
dropout =  0.1
embedding =  200
hidden =  200
nonlinearity =  'relu'
filters =  128
window_sizes =  [2,3,4]
gpu = True
hyperopt = False
save_path = None
train_metrics = Metrics(metrics, display_metric, stop_metric)
dev_metrics = Metrics(metrics, display_metric, stop_metric)
exp = 'liwc'

c = Cleaner(cleaners) # Cleaner(['url', 'hashtag', 'username', 'lower'])
experiment = Preprocessors('data/').select_experiment(exp)
onehot = True if encoding == 'onehot' else False

if tokenizer == 'spacy':
    selected_tok  = c.tokenize
elif tokenizer == 'bpe':
    selected_tok = c.bpe_tokenize
elif tokenizer == 'ekphrasis' and exp == 'word':
    selected_tok = c.ekphrasis_tokenize
    annotate = {'elongated', 'emphasis'}
    flters = [f"<{filtr}>" for filtr in annotate]
    c._load_ekphrasis(annotate, flters)
elif tokenizer == 'ekphrasis' and exp == 'liwc':
    ekphr = c.ekphrasis_tokenize
    annotate = {'elongated', 'emphasis'}
    flters = [f"<{filtr}>" for filtr in annotate]
    c._load_ekphrasis(annotate, flters)
    def liwc_toks(doc):
        tokens = ekphr(doc)
        tokens = experiment(tokens)
        return tokens
    selected_tok = liwc_toks
tokenizer = selected_tok

text = Field(tokenize = tokenizer, lower = True, batch_first = True)
label = LabelField()
fields = {'text': ('text', text), 'label': ('label', label)}  # Because we load from json we just need this.
train, dev, test = TabularDataset.splits(datadir, train = 'davidson_binary_train.json',
                                         validation = 'davidson_binary_dev.json', test = 'davidson_binary_test.json', 
                                         format = 'json', skip_header = True, fields = fields)
text.build_vocab(train)
label.build_vocab(train)

model = MLPClassifier(len(text.vocab.stoi), embedding, hidden, len(label.vocab.stoi), dropout, True, nonlinearity)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

train_ds = BucketIterator(dataset = train, batch_size = batch_size)
dev_ds = BucketIterator(dataset = dev, batch_size = batch_size)
batched_train = TorchtextExtractor('text', 'label', 'davidson_binary_train', train_ds)
batched_dev = TorchtextExtractor('text', 'label', 'davidson_binary_dev', dev_ds)

train_singletask_model(model, save_path, epochs, batched_train, loss, optimizer, train_metrics, dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = True, clip = 1.0)

max_train = np.argmax(train_metrics.scores['f1-score'])
max_dev = np.argmax(dev_metrics.scores['f1-score'])
print(max_train, train_metrics.scores['f1-score'][max_train])
print(max_dev, dev_metrics.scores['f1-score'][max_dev])

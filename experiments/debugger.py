import torch
import optuna
import numpy as np
from mlearn import base
from tqdm import tqdm, trange
from mlearn.utils.metrics import Metrics
from mlearn.modeling.embedding import MLPClassifier
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_singletask_model as run_model
from mlearn.utils.train import train_singletask_model
from mlearn.utils.pipeline import process_and_batch, param_selection
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator

class TorchTextDefaultExtractor:
    """A class to get index-tensor batches from torchtext data object."""

    def __init__(self, datafield: str, labelfield: str, dataloader: base.DataType):
       """Initialize batch generator for torchtext."""
       self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
       """Get length of the batches."""
       return len(self.data)

    def __iter__(self):
       """Iterate over batches in the data."""
       for batch in self.data:
          X = getattr(batch, self.df)
          y = getattr(batch, self.lf)
          yield (X, y)

# Initialize experiment
datadir = 'data/'
torch.random.manual_seed(42)
np.random.seed(42)
encoding = 'index'
tokenizer = 'bpe'
metrics = ['f1-score', 'precision', 'recall', 'accuracy']
display_metric = stop_metric = 'f1-score'
batch_size=  64
epochs = 50
learning_rate = 0.001
dropout = 0.0
embedding = 100
hidden = 100
nonlinearity = 'relu'
gpu = False
hyperopt = False
save_path = None
train_metrics = Metrics(metrics, display_metric, stop_metric)
dev_metrics = Metrics(metrics, display_metric, stop_metric)

c = Cleaner(['url', 'hashtag', 'username', 'lower'])
experiment = Preprocessors(datadir).select_experiment('word')
onehot = True if encoding == 'onehot' else False

if tokenizer == 'spacy':
    tokenizer = c.tokenize
elif tokenizer == 'bpe':
    tokenizer = c.bpe_tokenize
elif tokenizer == 'ekphrasis':
    tokenizer = c.ekphrasis_tokenize
    annotate = {'elongated', 'emphasis'}
    filters = [f"<{filtr}>" for filtr in annotate]
    c._load_ekphrasis(annotate, filters)

text = Field(tokenize = tokenizer, lower = True, batch_first = True)
label = LabelField()
fields = [('ignore', None), ('text', text), ('label', label), ('ignore', None)]
train, dev, test = TabularDataset.splits('/home/zeerakw/projects/MTL/data/', train = 'wulczyn_train.tsv',
                                         validation = 'wulczyn_dev.tsv', test = 'wulczyn_test.tsv', 
                                         format = 'tsv', skip_header = True, fields = fields)
text.build_vocab(train)
label.build_vocab(train)

model = MLPClassifier(len(text.vocab.stoi), embedding, hidden, len(label.vocab.stoi), False, nonlinearity)
loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

train_ds = BucketIterator(dataset = train, batch_size = batch_size)
dev_ds = BucketIterator(dataset = dev, batch_size = batch_size)
batched_train = TorchTextDefaultExtractor('text', 'label', train_ds)
batched_dev = TorchTextDefaultExtractor('text', 'label', dev_ds)

train_singletask_model(model, save_path, epochs, batched_train, loss, optimizer, train_metrics, dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = True)

import torch
import numpy as np
from mlearn.data import loaders
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import process_and_batch
from mlearn.modeling.embedding import MLPClassifier
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model


# Initialize experiment
datadir = 'data/'
torch.random.manual_seed(42)
np.random.seed(42)
encoding = 'index'
tokenizer = 'bpe'
metrics = ['f1-score', 'precision', 'recall', 'accuracy']
display_metric = stop_metric = 'f1-score'
batch_size = 64
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

main = loaders.wulczyn(tokenizer, datadir, preprocessor = experiment, label_processor = None,
                       tratify = 'label', skip_header = True)

main.build_token_vocab(main.data)
main.build_label_vocab(main.data)
breakpoint()
batched_train = process_and_batch(main, main.data, batch_size, False)
batched_dev = process_and_batch(main, main.dev, batch_size, False)

model = MLPClassifier(main.vocab_size(), embedding, hidden, main.label_count(), False, nonlinearity)
loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

train_singletask_model(model, save_path, epochs, batched_train, loss, optimizer, train_metrics, dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = True)


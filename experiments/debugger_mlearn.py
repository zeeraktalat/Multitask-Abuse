import torch
import numpy as np
from mlearn.data import loaders
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import process_and_batch
from mlearn.modeling.embedding import MLPClassifier
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model


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

c = Cleaner(['url', 'hashtag', 'username', 'lower'])
experiment = Preprocessors('data/').select_experiment(exp)
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

main = loaders.davidson_json(tokenizer, datadir, preprocessor = experiment, label_processor = None,
                            stratify = 'label', skip_header = True)

main.build_token_vocab(main.data)
main.build_label_vocab(main.data)

model = MLPClassifier(main.vocab_size(), embedding, hidden, main.label_count(), dropout, True, nonlinearity)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

batched_train = process_and_batch(main, main.data, batch_size, False)
batched_dev = process_and_batch(main, main.dev, batch_size, False)

train_singletask_model(model, save_path, epochs, batched_train, loss, optimizer, train_metrics, dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = True, clip = 1.0)

max_train = np.argmax(train_metrics.scores['f1-score'])
max_dev = np.argmax(dev_metrics.scores['f1-score'])
print(max_train, train_metrics.scores['f1-score'][max_train])
print(max_dev, dev_metrics.scores['f1-score'][max_dev])

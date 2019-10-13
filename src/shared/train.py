import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from src.shared.prep import Dataset, BatchGenerator
import src.shared.types as t


def multilabel_processing(label, field):
    labels = [(lab, field) for lab in label.split(',')]
    return labels


def select_batch(dist: t.List[int], batch_gens: t.List[t.BatchGenerator]):
    return np.random(batch_gens, size = 1, p = dist)


def sort_func(x):
    return len(x.data)


def create_batches(data_dir, splits, ftype, fields, cleaners, batch_sizes, shuffle, sep, skip_header,
                   repeat_in_batches, device):

    data = Dataset(data_dir = data_dir, splits = splits, ftype = ftype, fields = fields, cleaners = cleaners,
                   batch_sizes = batch_sizes, shuffle = shuffle, sep = sep, skip_header = skip_header,
                   repeat_in_batches = repeat_in_batches, device = device)

    loaded = data.load_data()
    if len(splits.keys()) == 1:
        train, test = data.split(split_ratio = 0.8, stratified = True, strata_field = 'label')
        loaded = (train, None, test)

    return loaded


def setup_data():
    """Train the model.
    :param epochs: The number of epochs to run.
    """
    device = 'cpu'
    data_dir = '/Users/zeerakw/Documents/PhD/projects/Multitask-abuse/data/'

    # MFTC
    mftc = Dataset(data_dir = data_dir,
                   splits = {'train': 'MFTC_V4_text_parsed.tsv', 'validation': None, 'test': None},
                   ftype = 'tsv', fields = None, cleaners = ['username', 'hashtag', 'url', 'lower'],
                   batch_sizes = (64,), shuffle = True, sep = '\t', skip_header = True,
                   repeat_in_batches = False, device = device)

    mftc_text = t.text_data
    mftc_label = t.text_label
    mftc.set_field_attribute(mftc_text, 'tokenize', mftc.tokenize)
    mftc.set_field_attribute(mftc_label, 'preprocessing', multilabel_processing)

    fields = [('tweet_id', None), ('data', mftc_text),
              ('annotator_1', None), ('annotator_2', None), ('annotator_3', None), ('annotator_4', None),
              ('annotator_5', None), ('annotator_6', None), ('annotator_7', None), ('annotator_8', None),
              ('label1', None), ('label2', None), ('label3', None), ('label4', None),
              ('label5', None), ('label6', None), ('label7', None), ('label8', None),
              ('label', mftc_label), ('corpus', None)]

    mftc.fields = fields

    mftc_data, _, _ = mftc.load_data()
    mftc_train, mftc_test = mftc.split(split_ration = 0.8, stratified = True, strata_field = 'label')
    mftc_text.build_vocab(mftc_train)
    mftc_label.build_vocab(mftc_train)

    train_iter, test_iter = mftc.generate_batches(sort_func, (mftc_train, mftc_test))
    mftc_train_batch = BatchGenerator(train_iter, 'data', 'label')
    mftc_test_batch = BatchGenerator(test_iter, 'data', 'label')
    mftc_loaded = (mftc_train_batch, mftc_test_batch)

    # Sentiment analysis
    sent = Dataset(data_dir = data_dir,
                   splits = {'train': 'semeval_sentiment_train.tsv', 'test': 'semeval_sentiment_test.tsv', },
                   ftype = 'tsv', fields = None, cleaners = ['username', 'hashtag', 'url', 'lower'],
                   batch_sizes = (64, 64), shuffle = True, sep = '\t', skip_header = True,
                   repeat_in_batches = False, device = device)

    sent_text = t.text_data
    sent_label = t.text_label
    sent.set_field_attribute(sent_text, 'tokenize', mftc.tokenize)

    fields = [('tweet_id', None), ('label', sent_label), ('data', sent_text)]
    sent.fields = fields

    sent_train, _, sent_test = sent.load_data()
    sent_text.build_vocab(sent_train)
    sent_label.build_vocab(sent_label)

    train_iter, _, test_iter = sent.generate_batches(sort_func = sort_func)
    sent_train_batch = BatchGenerator(train_iter, 'data', 'label')
    sent_test_batch = BatchGenerator(test_iter, 'data', 'label')
    sent_loaded = (sent_train_batch, sent_test_batch)

    return mftc_loaded, sent_loaded


def train(epochs):
    # TODO Load and batch data
    # TODO Create hard parameter sharing???
    # TODO Define loss for model.
    return

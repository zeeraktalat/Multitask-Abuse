import torch
import torch.nn.functional as F
import torch.optim as optim
from src.shared.prep import Dataset, BatchGenerator
import src.shared.types as t


def multilabel_processing(label, field):
    labels = [(lab, field) for lab in label.split(',')]
    return labels


def train(epochs: int = 200):
    """Train the model.
    :param epochs: The number of epochs to run.
    """
    mftc = Dataset(data_dir = '/Users/zeerakw/Documents/PhD/projects/Multitask-abuse',
                   splits = {'train': 'MFTC_V4_text_parsed.tsv', 'validation': None, 'test': None},
                   ftype = 'tsv', fields = None, cleaners = ['username', 'hashtag', 'url', 'lower'],
                   batch_sizes = (64,), shuffle = True, sep = None, skip_header = True,
                   repeat_in_batches = False, device = 'cpu')

    text = t.text_data
    label = t.text_label
    mftc.set_field_attribute(text, 'tokenize', mftc.tokenize)
    mftc.set_field_attribute(label, 'preprocessing', multilabel_processing)

    fields = [('tweet_id', None), ('data', text),
              ('annotator_1', None), ('annotator_2', None), ('annotator_3', None), ('annotator_4', None),
              ('annotator_5', None), ('annotator_6', None), ('annotator_7', None), ('annotator_8', None),
              ('label1', None), ('label2', None), ('label3', None), ('label4', None),
              ('label5', None), ('label6', None), ('label7', None), ('label8', None),
              ('label', label), ('corpus', None)]

    mftc.fields = fields


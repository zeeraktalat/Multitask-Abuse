import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import src.shared.types as t
import torch.nn.functional as F
from src.shared.prep import Dataset, BatchGenerator
from src.shared.clean import Cleaner


def multilabel_processing(label, field):
    labels = [(lab, field) for lab in label.split(',')]
    return labels


def select_batch(dist: t.List[int], batch_gens: t.List[t.BatchGenerator]):
    return np.random(batch_gens, size = 1, p = dist)


def sort_func(x):
    return len(x.data)


def store_fields(obj, data_field, label_field, **kwargs):
    """Store fields in the dataset object. Final two fields always label, train.
    :param data (t.FieldType): The field instance for the data.
    :param label (t.FieldType): The field instance for the label.
    :param kwargs: Will search for any fields in this.
    """
    if kwargs:
        field = []
        for k in kwargs:
            if 'field' in k:
                field.append(kwargs[k])
    field.extend([label_field, data_field])
    obj.field_instance = tuple(field)


def create_batches(data_dir: str, splits: t.Dict[str, t.Union[str, None]], ftype: str, fields: t.Union[dict, list],
                   cleaners: t.List[str], batch_sizes: t.Tuple[int, ...], shuffle: bool, sep: str, skip_header: bool,
                   repeat_in_batches: bool, device: t.Union[str, int],
                   data_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]],
                   label_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]], **kwargs):

    # Initiate the dataset object
    data = Dataset(data_dir = data_dir, splits = splits, ftype = ftype, fields = fields, cleaners = cleaners,
                   shuffle = shuffle, sep = sep, skip_header = skip_header, repeat_in_batches = repeat_in_batches,
                   device = device)

    # If the fields need new attributes set: set them.
    # TODO assumes only data and field labels need modification.
    if data_field[1]:
        data.set_field_attribute(data_field[0], data_field[1]['attribute'], data_field[1]['value'])

    if label_field[1]:
        data.set_field_attribute(label_field[0], label_field[1]['attribute'], label_field[1]['value'])

    # Store our Field instances so we can later access them.
    store_fields(data, data_field, label_field, **kwargs)

    data.fields = fields  # Update the fields in the class

    loaded = data.load_data()  # Data paths exist in the class

    if len([v for v in splits.values() if v is not None]) == 1:  # If only one dataset is given
        train, test = data.split(split_ratio = kwargs['split_ratio'], stratified = True, strata_field = kwargs['label'])
        loaded = (train, None, test)

    data_field.build_vocab()
    label_field.build_vocab()

    train, dev, test = data.generate_batches(sort_func, loaded, batch_sizes)
    train_batch = BatchGenerator(train, 'data', 'label')
    dev_batch = BatchGenerator(dev, 'data', 'label') if dev is not None else None
    test_batch = BatchGenerator(test, 'data', 'label')

    batches = (train_batch, dev_batch, test_batch)
    return data, batches


def setup_data():
    """Train the model.
    :param epochs: The number of epochs to run.
    """
    device = 'cpu'
    data_dir = '/Users/zeerakw/Documents/PhD/projects/Multitask-abuse/data/'
    clean = Cleaner()

    # MFTC
    mftc_text = (t.text_data, {'attribute': 'tokenize', 'value': clean.tokenize})
    # TODO Move Multilable procoessing
    mftc_label = (t.text_label, {'attribute': 'preprocessing', 'value': multilabel_processing})

    fields = [('tweet_id', None), ('data', mftc_text),
              ('annotator_1', None), ('annotator_2', None), ('annotator_3', None), ('annotator_4', None),
              ('annotator_5', None), ('annotator_6', None), ('annotator_7', None), ('annotator_8', None),
              ('label1', None), ('label2', None), ('label3', None), ('label4', None),
              ('label5', None), ('label6', None), ('label7', None), ('label8', None),
              ('label', mftc_label), ('corpus', None)]

    mftc_opts = {'splits': {'train': 'MFTC_V4_text_parsed.tsv'}, 'ftype': 'tsv', 'data_field': mftc_text,
                 'label_field': mftc_label, 'batch_sizes': (64, 64), 'shuffle': True, 'sep': '\t', 'skip_header': True,
                 'repeat_in_batches': False}

    mftc = create_batches(data_dir = data_dir, device = device, **mftc_opts)

    # Sentiment analysis
    sent_text = (t.text_data, {'attribute': 'tokenize', 'value': clean.tokenize})
    sent_label = (t.text_label, None)

    fields = [('tweet_id', None), ('label', sent_label), ('data', sent_text)]

    sent_opts = {'splits': {'train': 'semeval_sentiment_train.tsv', 'test': 'semeval_sentiment_test.tsv'},
                 'ftype': 'tsv', 'fields': fields, 'shuffle': True, 'sep': '\t', 'skip_header': True,
                 'repeat_in_batches': False}
    sent = create_batches(data_dir = data_dir, device = device, **sent_opts)

    return mftc, sent


def train(epochs):

    for ep in tqdm(range(epochs)):
        # TODO Load and batch data
        # TODO Create hard parameter sharing???
        # TODO Define loss for model.
    return

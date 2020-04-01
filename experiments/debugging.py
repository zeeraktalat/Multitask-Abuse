import torch

from mlearn.base import Field
from mlearn.data_processing import clean
# from mlearn.data_processing import loaders
from mlearn.modeling.train import train_mtl_model
from mlearn.modeling.neural import MTLLSTMClassifier
from mlearn.data_processing.data import GeneralDataset


if __name__ == "__main__":

    # Set up cleaners
    cl = clean.Cleaner(processes = ['lower', 'url', 'hashtag'])
    pr = clean.Preprocessors(liwc_path = '~/PhD/projects/active/MTL_abuse/data/liwc-2015.csv')

    # Load Davidson - Slow
    # davidson = loaders.davidson(cleaners = cl, data_path = '~/PhD/projects/active/MTL_abuse/data/', length = 200,
    #                             label_processor = None)

    # Load Davidson - Fast
    text_field = Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
    label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)
    ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

    davidson_fields = [ignore_field, ignore_field, ignore_field, ignore_field, ignore_field, label_field, text_field]

    davidson = GeneralDataset(data_dir = '~/PhD/projects/active/MTL_abuse/data/',
                              ftype = 'csv', fields = davidson_fields, train = 'davidson_offensive.csv', dev = None,
                              test = None, train_labels = None, tokenizer = lambda x: x.split(),
                              lower = True, preprocessor = None, transformations = None,
                              label_processor = None, sep = ',', name = 'Davidson et al.')
    davidson.load('train')

    # Load Hoover - Slow
    # hoover = loaders.hoover(cleaners = cl, data_path = '~/PhD/projects/active/MTL_abuse/data/', length = 200,
    #                         preprocessor = pr.word_token, label_processor = lambda x: x.split()[0])

    # Load Hoover - Fast
    text_field = Field('text', train = True, label = False, ignore = False, ix = 1, cname = 'text')
    label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 18)
    ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

    hoover_fields = [ignore_field, text_field] + 16 * [ignore_field] + [label_field, ignore_field]

    hoover = GeneralDataset(data_dir = '~/PhD/projects/active/MTL_abuse/data/',
                            ftype = 'tsv', fields = hoover_fields, train = 'MFTC_V4_text_parsed.tsv', dev = None,
                            test = None, train_labels = None, tokenizer = lambda x: x.split(),
                            lower = True, preprocessor = None, transformations = None,
                            label_processor = None, sep = '\t', name = 'Hoover et al.')
    hoover.load('train')

    # Process data
    davidson.build_token_vocab(davidson.data)
    davidson.build_label_vocab(davidson.data)

    hoover.build_token_vocab(hoover.data)
    hoover.build_label_vocab(hoover.data)

    model = MTLLSTMClassifier(input_dims = [int(hoover.vocab_size()), int(davidson.vocab_size())], shared_dim = 150,
                              hidden_dims = [128, 128], output_dims = [hoover.label_count(), davidson.label_count()],
                              no_layers = 1, dropout = 0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = torch.nn.CrossEntropyLoss()
    train_mtl_model(model, [hoover, davidson], 'results/', optimizer, dev_data = hoover.dev, loss_func = loss)

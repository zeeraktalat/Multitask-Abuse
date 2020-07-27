import os
import csv
import torch
import numpy as np
from tqdm import tqdm
import mlearn.data.loaders as loaders
from mlearn.utils.metrics import Metrics
import mlearn.modeling.multitask as mod_lib
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_mtl_model as run_model
from mlearn.utils.pipeline import process_and_batch, hyperparam_space


if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                        type = str.lower, default = 'Davidson')
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", nargs = '+',
                        default = ['mlp'], type = str.lower)
    parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/models/')
    parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                        type = str.lower)
    parser.add_argument("--stop_metric", help = "Set the metric to be used for early stopping", default = "loss")
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                        default = None, type = int)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'f1-score')
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')

    # Model architecture
    parser.add_argument("--embedding_dim", help = "Set the embedding dimension.", default = [300], type = int,
                        nargs = '+')
    parser.add_argument("--hidden_dim", help = "Set the hidden dimension.", default = [128], type = int, nargs = '+')
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Embedding, Tfidf, Count",
                        default = 'embedding', type = str.lower)

    # Model (hyper) parameters
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int, nargs = '+')
    parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int, nargs = '+')
    parser.add_argument("--dropout", help = "Set value for dropout.", default = [0.0], type = float, nargs = '+')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = [0.01],
                        type = float, nargs = '+')
    parser.add_argument("--nonlinearity", help = "Set nonlinearity function for neural nets.", default = ['tanh'],
                        type = str.lower, nargs = '+')
    parser.add_argument("--hyperparams", help = "List of names of the hyper parameters to be searched.",
                        default = ['epochs'], type = str.lower, nargs = '+')

    # Experiment parameters
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = bool, default = False)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--slur_window", help = "Set window size for slur replacement.", default = None, type = int,
                        nargs = '+')
    parser.add_argument('--cfg', action = ActionConfigFile, default = None)

    args = parser.parse_args()

    if 'f1' in args.metrics + [args.display, args.stop_metric]:
        for i, m in enumerate(args.metrics):
            if 'f1' in m:
                args.metrics[i] = 'f1-score'
        if args.display == 'f1':
            args.display = 'f1-score'
        if args.stop_metric == 'f1':
            args.display = 'f1-score'

    if args.encoding == 'onehot':
        onehot = True
    elif args.encoding == 'embedding':
        onehot = False

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    c = Cleaner(args.cleaners)
    p = Preprocessors(args.datadir)

    experiment = p.word_token

    # Define arg dictionaries
    train_args = {}
    model_args = {}

    # Load datasets
    # TODO write loaders for all other datasets.
    if 'waseem' in args.main:  # Waseem is the main task
        main = loaders.waseem(c, args.datadir, preprocessor = experiment,
                               label_processor = loaders.waseem_to_binary, stratify = 'label'),
        other = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label')
        main.data.extend(other.data)
        main.dev.extend(other.dev)
        main.test.extend(other.test)

        aux = [loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True),
               loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary, stratify = 'label',
                                skip_header = True),
               ]

    if args.main == 'davidson':
        main = loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary,
                                stratify = 'label', skip_header = True)

        waseem = loaders.waseem(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.waseem_to_binary, stratify = 'label'),
        w_hovy = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
        waseem.data.extend(w_hovy.data)
        waseem.dev.extend(w_hovy.dev)
        waseem.test.extend(w_hovy.test)

        aux = [loaders.wulczyn(c, args.datadir, preprocessor = experiment,
                               stratify = 'label', skip_header = True),
               waseem,
               ]

    elif args.main == 'wulczyn':
        main = loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True)

        waseem = loaders.waseem(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.waseem_to_binary, stratify = 'label'),
        w_hovy = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
        waseem.data.extend(w_hovy.data)
        waseem.dev.extend(w_hovy.dev)
        waseem.test.extend(w_hovy.test)

        aux = [loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary, stratify = 'label',
                                skip_header = True),
               waseem,
               ]

    datasets = [main] + aux
    dev = main.dev
    test = main.test

    # Build token and label vocabularies for datasets
    for dataset in datasets:
        dataset.build_token_vocab(dataset.data)
        dataset.build_label_vocab(dataset.data)

    # Batch dev and test
    dev_batcher = process_and_batch(main, main.dev, 64, onehot)
    test_batcher = process_and_batch(main, main.test, 64, onehot)

    # Set input and ouput dims
    train_args['input_dim'] = [dataset.vocab_size() for dataset in datasets]
    train_args['output_dim'] = [dataset.label_count() for dataset in dataset]
    train_args['main_name'] = main.name

    # Set models to iterate over
    models = []
    for m in args.model:
        if m == 'mlp':
            if onehot:
                models.append(mod_lib.OnehotMLPClassifier)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # Select optimizer
    if args.optimizer == 'adam':
        model_args['optimizer'] = torch.optim.Adam
    elif args.optimizer == 'sgd':
        model_args['optimizer'] = torch.optim.SGD
    elif args.optimizer == 'asgd':
        model_args['optimizer'] = torch.optim.ASGD
    elif args.optimizer == 'adamw':
        model_args['optimizer'] = torch.optim.AdamW

    if args.loss == 'nlll':
        model_args['loss_func'] = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        model_args['loss_func'] = torch.nn.CrossEntropyLoss

    # Set models
    if args.model == 'lstm':
        models = [LSTMClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output dim', 'num layers',
                        'learning rate']
        model_info = {'lstm': ['lstm', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                               train_args['output_dim'], train_args['num_layers'], args.learning_rate]}

    train_args['batches'] = process_and_batch(main, main.data, args.batch_size, args.onehot)

    if main.dev is not None:  # As the dataloaders always create a dev set, this condition will always be True
        train_args['dev_batches'] = process_and_batch(main, main.dev, args.batch_size, args.onehot)

    test_sets = [process_and_batch(main, data.test, args.batch_size, args.onehot) for data in evals]

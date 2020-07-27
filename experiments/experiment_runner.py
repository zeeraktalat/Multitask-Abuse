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

    base_params = {'gpu': args.gpu,
                   'batch_first': True,
                   'model': MTLLSTMClassifier,
                   'metrics': select_metrics(args.metrics)
                   }

    hyper_params = {'dropout': args.dropout,
                    'shuffle': args.shuffle,
                    'epochs': args.epochs
                    }

    cnn_args = {'num_filters': args.filters,
                'max_feats': args.max_feats,
                'window_sizes': args.window_sizes
                }

    rnn_args = {'hidden_dim': args.hidden,
                'embedding_dim': args.embedding,
                'num_layers': 1,
                }

    eval_args = {'model': None,
                 'iterator': None,
                 'loss_func': None,
                 'metrics': train_args['metrics'],
                 'gpu': args.gpu
                 }

    train_args = {} + base_params + hyper_params

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up preprocessing
    c = Cleaner(args.cleaners)
    p = Preprocessors()

    args.features = args.features.lower()
    args.train = args.train.lower()
    args.loss = args.loss.lower()
    args.optimizer = args.optimizer.lower()
    args.model = args.model.lower()

    # Set features to run
    if args.features == 'word':
        features = p.word_token

    # if args.main == 'davidson':
    #     main = loaders.davidson(c, features)
    #     evals = [main, loaders.wulczyn(c, features), loaders.garcia(c, features), loaders.waseem(c, features),
    #              loaders.waseem_hovy(c, features), ]
    #
    # main.build_token_vocab(main.data)
    # main.build_label_vocab(main.data)

    train_args['input_dims'] = [main.vocab_size()] + [a.vocab_size() for a in aux]
    train_args['hidden_dims'] = args.hidden
    train_args['output_dim'] = [main.label_count()] + [a.label_count() for a in aux]

    model_args = {}

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

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
    parser = argparse.ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument('--gpu', help = "Set to run on GPU", action = 'store_true', default = False)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--train", help = "Choose train data: davidson, Waseem, Waseem and Hovy, wulczyn, and garcia.")
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = "mlp")
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"], type = str)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.")
    parser.add_argument("--results", help = "Set file to output results to.")
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 64, type = int)
    parser.add_argument("--save_model", help = "Directory to store models in.")

    # Model (hyper) parameters
    parser.add_argument("--loss", help = "Loss to use.", default = 'NLLL')
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 200, type = int)
    parser.add_argument("--dropout", help = "Set value for dropout.", default = 0.0, type = float)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam')
    parser.add_argument("--learning_rate", help = "Set the learning rate for the model.", default = 0.01, type = float)

    # All
    parser.add_argument('--onehot', help = "Use one-hot tensors.", action = 'store_true', default = False)
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", action = 'store_true', default = True)

    # RNN / LSTM
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [128, 128], type = int, nargs = '+')
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 300, type = int)

    # CNN
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128, type = int)
    parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100, type = int)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [2, 3, 4], type = int)

    # Experiment parameters
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--features", help = "Decide which features to train on.", default = "word_token")

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

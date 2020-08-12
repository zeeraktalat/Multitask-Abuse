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


csv.field_size_limit(1000000)


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
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = [[100, 100, 100]], type = list,
                        nargs = '+')
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [[128, 128, 128]], type = list,
                        nargs = '+')
    parser.add_argument("--shared", help = "Set the shared dimension", default = [256], type = int, nargs = '+')
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
    parser.add_argument("--batches_epoch", help = "Set the number of batches per epoch", type = int, default = None)
    parser.add_argument("--loss_weights", help = "Set the weight of each task", type = int, default = None,
                        nargs = '+')
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = bool, default = False)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)
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
                               label_processor = None, stratify = 'label')
        other = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                    label_processor = None,
                                    stratify = 'label')
        main.data.extend(other.data)
        main.dev.extend(other.dev)
        main.test.extend(other.test)

        aux = [loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True),
               loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = None, stratify = 'label',
                                skip_header = True),
               # loaders.preotiuc_user(c, args.datadir, preprocessor = experiment, label_processor = None,
               #                       stratify = 'label'),
               loaders.oraby_sarcasm(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True),
               loaders.oraby_fact_feel(c, args.datadir, preprocessor = experiment, skip_header = True),
               loaders.hoover(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True)
               ]

    if args.main == 'davidson':
        main = loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = None,
                                stratify = 'label', skip_header = True)

        waseem = loaders.waseem(c, args.datadir, preprocessor = experiment,
                                label_processor = None, stratify = 'label')
        w_hovy = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = None,
                                     stratify = 'label')
        waseem.data.extend(w_hovy.data)
        waseem.dev.extend(w_hovy.dev)
        waseem.test.extend(w_hovy.test)

        aux = [loaders.wulczyn(c, args.datadir, preprocessor = experiment,
                               stratify = 'label', skip_header = True),
               waseem, loaders.preotiuc_user(c, args.datadir, preprocessor = experiment, label_processor = None,
                                             stratify = 'label'),
               loaders.oraby_sarcasm(c, args.datadir, preprocessor = experiment, stratify = 'label'),
               loaders.oraby_fact_feel(c, args.datadir, preprocessor = experiment),
               loaders.hoover(c, args.datadir, preprocessor = experiment, stratify = 'label')
               ]

    elif args.main == 'wulczyn':
        main = loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True)

        waseem = loaders.waseem(c, args.datadir, preprocessor = experiment,
                                label_processor = None, stratify = 'label')
        w_hovy = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = None,
                                     stratify = 'label')
        waseem.data.extend(w_hovy.data)
        waseem.dev.extend(w_hovy.dev)
        waseem.test.extend(w_hovy.test)

        aux = [loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = None, stratify = 'label',
                                skip_header = True),
               waseem, loaders.preotiuc_user(c, args.datadir, preprocessor = experiment, label_processor = None,
                                             stratify = 'label'),
               loaders.oraby_sarcasm(c, args.datadir, preprocessor = experiment, stratify = 'label'),
               loaders.oraby_fact_feel(c, args.datadir, preprocessor = experiment),
               loaders.hoover(c, args.datadir, preprocessor = experiment, stratify = 'label')
               ]

    datasets = [main] + aux
    dev = main.dev
    test = main.test

    # Build token and label vocabularies for datasets
    for dataset in datasets:
        dataset.build_token_vocab(dataset.data)
        dataset.build_label_vocab(dataset.data)

    # Batch dev and test
    train_args['dev'] = process_and_batch(main, main.dev, 64, onehot)
    test_batcher = process_and_batch(main, main.test, 64, onehot)

    # Set input and ouput dims
    train_args['input_dims'] = [dataset.vocab_size() for dataset in datasets]
    train_args['output_dims'] = [dataset.label_count() for dataset in datasets]
    train_args['main_name'] = main.name

    # Set number of batches per epoch and weight of each task
    train_args['batches_per_epoch'] = args.batches_epoch
    train_args['loss_weights'] = args.loss_weights

    # Set models to iterate over
    models = []
    for m in args.model:
        if m == 'mlp':
            if onehot:
                models.append(mod_lib.OnehotMLPClassifier)
            else:
                models.append(mod_lib.EmbeddingMLPClassifier)
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

    # Explains losses:
    # https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
    # Set loss
    if args.loss == 'nlll':
        model_args['loss'] = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        model_args['loss'] = torch.nn.CrossEntropyLoss

    # Open output files
    base = f'{args.results}{args.main}_{args.encoding}_{args.experiment}'
    enc = 'a' if os.path.isfile(f'{base}_train.tsv') else 'w'
    pred_enc = 'a' if os.path.isfile(f'{base}_preds.tsv') else 'w'

    train_writer = csv.writer(open(f"{base}_train.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{base}_test.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{base}_preds.tsv", pred_enc, encoding = 'utf-8'), delimiter = '\t')
    batch_writer = csv.writer(open(f"{base}_batch.tsv", enc, encoding = 'utf-8'), delimiter = '\t')

    model_hdr = ['Model', 'Input dim', 'Embedding dim', 'Hidden dim', 'Output dim', 'Dropout', 'nonlinearity']
    train_args.update({'model_hdr': model_hdr, 'metric_hdr': args.metrics + ['loss'], 'batch_writer': batch_writer})

    if enc == 'w':
        metric_hdr = args.metrics + ['loss']
        hdr = ['Timestamp', 'Main task', 'Tasks', 'Batch size', '# Epochs', 'Learning rate'] + model_hdr
        hdr += metric_hdr
        test_writer.writerow(hdr)  # Don't include dev columns when writing test
        train_writer.writerow(hdr)

        # Batch hdr
        batch_hdr = ['Timestamp', 'Epoch', 'Batch', 'Task name', 'Main task', 'Batch size', '# Epochs', 'Learning rate']
        batch_hdr += model_hdr + metric_hdr
        batch_writer.writerow(batch_hdr)

    pred_metric_hdr = args.metrics + ['loss']
    if pred_enc == 'w':
        hdr = ['Timestamp', 'Main task', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += ['Label', 'Prediction']
        pred_writer.writerow(hdr)

    # Get hyper-parameter combinations
    base_param = args.hyperparams.pop()
    search_space = [{base_param: val} for val in getattr(args, base_param)]
    hyper_parameters = [(param, getattr(args, param)) for param in args.hyperparams]

    train_args.update({'num_layers': 1,
                       'shuffle': args.shuffle,
                       'batch_first': True,
                       'gpu': args.gpu,
                       'save_path': f"{args.save_model}{args.main}_{args.experiment}_best",
                       'early_stopping': args.patience,
                       'low': True if args.stop_metric == 'loss' else False,
                       'data_name': "_".join([data.name.split()[0] for data in datasets])
                       })

    with tqdm(args.batch_size, desc = "Batch Size Iterator") as b_loop,\
         tqdm(models, desc = "Model Iterator") as m_loop:
        for batch_size in b_loop:
            b_loop.set_postfix(batches = batch_size)
            train_args['batchers'] = [process_and_batch(dataset, dataset.data, batch_size, onehot)
                                      for dataset in datasets]

            for parameters in tqdm(hyperparam_space(search_space, hyper_parameters), desc = "Hyper-parameter Iterator"):
                train_args.update(parameters)
                if 'hidden' in train_args:
                    train_args['hidden_dims'] = train_args['hidden']
                    del train_args['hidden']
                elif 'embedding' in train_args:
                    train_args['embedding_dims'] = train_args['embedding']
                    del train_args['embedding']
                train_args['shared_dim'] = train_args['shared']

                # hyper_info = ['Batch size', '# Epochs', 'Learning Rate']
                train_args['hyper_info'] = [batch_size, train_args['epochs'], train_args['learning_rate']]
                for model in m_loop:
                    # Intialize model, loss, optimizer, and metrics
                    train_args['model'] = model(**train_args)
                    train_args['loss'] = model_args['loss']()
                    train_args['optimizer'] = model_args['optimizer'](train_args['model'].parameters(),
                                                                      train_args['learning_rate'])

                    train_args['metrics'] = Metrics(args.metrics, args.display, args.stop_metric)
                    train_args['dev_metrics'] = Metrics(args.metrics, args.display, args.stop_metric)
                    m_loop.set_postfix(model = train_args['model'].name)  # Cur model name

                    run_model(train = True, writer = train_writer, **train_args)

                    eval_args = {'model': train_args['model'],
                                 'batchers': test_batcher,
                                 'loss': train_args['loss'],
                                 'metrics': Metrics(args.metrics, args.display, args.stop_metric),
                                 'gpu': args.gpu,
                                 'data': main.test,
                                 'dataset': main,
                                 'hyper_info': train_args['hyper_info'],
                                 'model_hdr': train_args['model_hdr'],
                                 'metric_hdr': train_args['metric_hdr'],
                                 'main_name': train_args['main_name'],
                                 'data_name': main.name,
                                 'train_field': 'text',
                                 'label_field': 'label',
                                 'store': True,
                                 'mtl': 0
                                 }
                    run_model(train = False, writer = test_writer, pred_writer = pred_writer, **eval_args)

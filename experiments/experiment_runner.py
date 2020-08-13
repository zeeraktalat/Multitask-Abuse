import os
import csv
import torch
import optuna
import numpy as np
from tqdm import tqdm
import mlearn.data.loaders as loaders
from mlearn.utils.metrics import Metrics
import mlearn.modeling.multitask as mod_lib
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_mtl_model as run_model
from mlearn.utils.pipeline import process_and_batch, param_selection


def sweeper(trial, training: dict, datasets: list, params: dict, model, modeling: dict, direction: str):
    """
    The function that contains all loading and setting of values and running the sweeps.

    :trial: The Optuna trial.
    :training (dict): Dictionary containing training modeling.
    :datasets (list): List of datasets objects.
    :params (dict): A dictionary of the different tunable parameters and their values.
    :model: The model to train.
    :modeling (dict): The arguments for the model and metrics objects.
    """
    optimisable = param_selection(trial, params)

    # TODO Think of a way to not hardcode this.
    training.update(dict(
        batchers = [process_and_batch(dataset, dataset.data, optimisable['batch_size'], onehot)
                    for dataset in datasets],
        hidden_dims = optimisable['hidden'] if 'hidden' in optimisable else None,
        embedding_dims = optimisable['embedding'] if 'embedding' in optimisable else None,
        shared_dim = optimisable['shared'],
        hyper_info = [optimisable['batch_size'], optimisable['epochs'], optimisable['learning_rate']],
        dropout = optimisable['dropout'],
        nonlinearity = optimisable['nonlinearity'],
        epochs = optimisable['epochs'],
        hyperopt = trial
    ))
    training['model'] = model(**training)
    training.update(dict(
        loss = modeling['loss'](),
        optimizer = modeling['optimizer'](training['model'].parameters(), optimisable['learning_rate']),
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        dev_metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop'])
    ))

    run_model(train = True, writer = modeling['train_writer'], **training)

    if direction == 'minimize':
        metric = training['dev_metrics'].loss
    else:
        metric = np.mean(training['dev_metrics'].scores[modeling['display']])

    eval = dict(
        model = training['model'],
        batchers = modeling['test_batcher'],
        loss = training['loss'],
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        gpu = training['gpu'],
        data = modeling['main'].test,
        dataset = modeling['main'],
        hyper_info = training['hyper_info'],
        model_hdr = training['model_hdr'],
        metric_hdr = training['metric_hdr'],
        main_name = training['main_name'],
        data_name = training['main_name'],
        train_field = 'text',
        label_field = 'label',
        store = False,
        mtl = 0
    )

    run_model(train = False, writer = modeling['test_writer'], pred_writer = None, **eval)

    return metric


def limiter(counts_dict: dict, limit: int, limit_type: str = 'freq'):
    """
    Limit the vocabulary of a dataset.

    :counts_dict (dict): The counts object to limit.
    :limit (int): The frequency or max number of tokens.
    :type (str, default = freq): Limit based on frequency or max vocab size.
    """
    remaining, deleted = [], []
    if limit_type == 'size':
        most_common = dict(counts_dict.most_common(limit))

    for token, frequency in counts_dict.items():
        if limit_type == 'freq':
            if frequency >= limit:
                remaining.append(token)
            else:
                deleted.append(token)
        elif limit_type == 'size':
            if token in most_common:
                remaining.append(token)
            else:
                deleted.append(token)
    return remaining, deleted


if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    # For all modesl
    parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                        type = str.lower, default = 'Davidson')
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", nargs = '+',
                        default = ['mlp'], type = str.lower)
    parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/models/')
    parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                        type = str.lower)
    parser.add_argument("--stop_metric", help = "Set the metric to be used for early stopping", default = "loss")
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'f1-score')
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                        default = None, type = int)

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
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, BPE", default = 'spacy',
                        type = str.lower)

    # Model (hyper) parameters
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int, nargs = '+')
    parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int, nargs = '+')
    parser.add_argument("--dropout", help = "Set value for dropout.", default = [0.0, 0.0], type = float, nargs = '+')
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

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    csv.field_size_limit(1000000)

    # Initialize experiment
    c = Cleaner(args.cleaners)
    p = Preprocessors(args.datadir)
    experiment = p.word_token
    onehot = True if args.encoding == 'onehot' else False

    tokenizer = c.tokenize if args.tokenizer == 'spacy' else c.bpe_tokenize

    # Load datasets
    if 'waseem' in args.main:  # Waseem is the main task
        main = loaders.waseem(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                              stratify = 'label')

        aux = [loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True),
               loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                   stratify = 'label'),
               loaders.davidson(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                stratify = 'label', skip_header = True),
               loaders.preotiuc_user(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                     stratify = 'label'),
               loaders.oraby_sarcasm(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
               loaders.oraby_fact_feel(tokenizer, args.datadir, preprocessor = experiment, skip_header = True),
               loaders.hoover(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                              skip_header = True)
               ]

    if args.main == 'davidson':
        main = loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                label_processor = None,
                                stratify = 'label', skip_header = True)

        aux = [loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True),
               loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                   stratify = 'label'),
               loaders.waseem(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                              stratify = 'label'),
               loaders.preotiuc_user(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                     stratify = 'label'),
               loaders.oraby_sarcasm(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label'),
               loaders.oraby_fact_feel(tokenizer, args.datadir, preprocessor = experiment),
               loaders.hoover(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label')
               ]

    elif args.main == 'wulczyn':
        main = loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True)

        aux = [loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                   stratify = 'label'),
               loaders.waseem(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                              stratify = 'label'),
               loaders.davidson(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                stratify = 'label', skip_header = True),
               loaders.preotiuc_user(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                     stratify = 'label'),
               loaders.oraby_sarcasm(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label'),
               loaders.oraby_fact_feel(tokenizer, args.datadir, preprocessor = experiment),
               loaders.hoover(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label')
               ]

    datasets = [main] + aux
    dev = main.dev
    test = main.test

    # Build token and label vocabularies for datasets
    for dataset in datasets:
        dataset.build_token_vocab(dataset.data)
        dataset.build_label_vocab(dataset.data)

    # Limit the Wulczyn and Preotiuc vocabularies to at least 3 instances.
    preotiuc = aux[3]
    preotiuc.limit_vocab(limiter, limit = 3, limit_type = 'freq')

    wulczyn = aux[0] if args.main != 'wylczyn' else main
    wulczyn.limit_vocab(limiter, limit = 3, limit_type = 'freq')

    # Open output files
    base = f'{args.results}{args.main}_{args.encoding}_{args.experiment}'
    enc = 'a' if os.path.isfile(f'{base}_train.tsv') else 'w'
    pred_enc = 'a' if os.path.isfile(f'{base}_preds.tsv') else 'w'

    train_writer = csv.writer(open(f"{base}_train.trial.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{base}_test.trial.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{base}_preds.trial.tsv", pred_enc, encoding = 'utf-8'), delimiter = '\t')
    batch_writer = csv.writer(open(f"{base}_batch.trial.tsv", enc, encoding = 'utf-8'), delimiter = '\t')

    model_hdr = ['Model', 'Input dim', 'Embedding dim', 'Hidden dim', 'Output dim', 'Dropout', 'nonlinearity']

    if enc == 'w':
        metric_hdr = args.metrics + ['loss']
        hdr = ['Timestamp', 'Main task', 'Tasks', 'Batch size', '# Epochs', 'Learning rate'] + model_hdr
        hdr += metric_hdr
        test_writer.writerow(hdr)  # Don't include dev columns when writing test
        hdr += [f"dev {m}" for m in metric_hdr]
        train_writer.writerow(hdr)

        # Batch hdr
        batch_hdr = ['Timestamp', 'Epoch', 'Batch', 'Task name', 'Main task', 'Batch size', '# Epochs', 'Learning rate']
        batch_hdr += model_hdr + metric_hdr
        batch_writer.writerow(batch_hdr)

    # Define arguments
    train_args = dict(
        # For writers
        model_hdr = model_hdr,
        metric_hdr = args.metrics + ["loss"],
        batch_writer = batch_writer,

        # Batch dev
        dev = process_and_batch(main, main.dev, 64, onehot),

        # Set model dimensionality
        input_dims = [dataset.vocab_size() for dataset in datasets],
        output_dims = [dataset.label_count() for dataset in datasets],
        num_layers = 1,  # LSTM
        batch_first = True,
        early_stopping = args.patience,

        # Name of main task
        main_name = main.name,
        batches_per_epoch = args.batches_epoch,  # Set batches per epoch
        loss_weights = args.loss_weights,  # Set weight of each task

        # Meta information
        shuffle = args.shuffle,
        gpu = args.gpu,
        save_path = f"{args.save_model}{args.main}_{args.experiment}_best",
        low = True if args.stop_metric == "loss" else False,
        data_name = "_".join([data.name.split()[0] for data in datasets])
    )

    # Select optimizer and losses
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD
    elif args.optimizer == 'asgd':
        optimizer = torch.optim.ASGD
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW

    # Info about losses: https://bit.ly/3irxvYK
    if args.loss == 'nlll':
        loss = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        loss = torch.nn.CrossEntropyLoss

    modeling = dict(
        loss = loss,
        optimizer = optimizer,
        metrics = args.metrics,
        display = args.display,
        stop = args.stop_metric,
        test_batcher = process_and_batch(main, main.test, 64, onehot),
        main = main,
        batch_writer = batch_writer,
        train_writer = train_writer,
        test_writer = test_writer,
        pred_writer = None,
    )

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

    pred_metric_hdr = args.metrics + ['loss']
    if pred_enc == 'w':
        hdr = ['Timestamp', 'Main task', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += ['Label', 'Prediction']
        pred_writer.writerow(hdr)

    with tqdm(models, desc = "Model Iterator") as m_loop:
        params = {param: getattr(args, param) for param in args.hyperparams}  # Get hyper-parameters to search
        direction = 'minimize' if args.display == 'loss' else 'maximize'
        study = optuna.create_study(study_name = 'MTL-abuse', direction = direction)
        trial_file = open(f"{base}.trials", 'a', encoding = 'utf-8')

        for m in models:
            study.optimize(lambda trial: sweeper(trial, train_args, datasets, params, m, modeling, direction),
                           n_trials = 100, gc_after_trial = True, n_jobs = 1, show_progress_bar = True)

            print(f"Model: {m}", file = trial_file)
            print(f"Best parameters: {study.best_params}", file = trial_file)
            print(f"Best trial: {study.best_trial}", file = trial_file)
            print(f"All trials: {study.trials}")

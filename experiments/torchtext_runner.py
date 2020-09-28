import torch
import optuna
import numpy as np
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import param_selection
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_mtl_model as run_model
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator


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
        # TODO Update from process_and_batch
        # batchers = [process_and_batch(dataset, dataset.data, optimisable['batch_size'], modeling['onehot'])
        #             for dataset in datasets],
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
    parser.add_argument("--aux", help = "Specify the auxiliary datasets to be loaded.", type = str, nargs = '+')

    # Model architecture
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = [(100)], type = tuple,
                        nargs = '+')
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [(128, 128, 128)], type = tuple,
                        nargs = '+')
    parser.add_argument("--shared", help = "Set the shared dimension", default = [256], type = int, nargs = '+')
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Embedding, Tfidf, Count",
                        default = 'embedding', type = str.lower)
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, Ekphrasis, BPE",
                        default = 'ekphrasis', type = str.lower)

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
    # torch.cuda.set_device(0)

    # Initialize experiment
    c = Cleaner(args.cleaners)
    experiment = Preprocessors(args.datadir).select_experiment(args.experiment)
    onehot = True if args.encoding == 'onehot' else False

    if args.tokenizer == 'spacy':
        tokenizer = c.tokenize
    elif args.tokenizer == 'bpe':
        tokenizer = c.bpe_tokenize
    elif args.tokenizer == 'ekphrasis':
        tokenizer = c.ekphrasis_tokenize
        # Set annotations, corrections and filters.
        annotate = {'elongated', 'emphasis'}
        filters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, filters)

    if args.main == 'waseem':
        text = Field(tokenize = tokenizer, lower = True, batch_first = True)
        label = LabelField()
        fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
        train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                 validation = 'waseem_dev.json', test = 'waseem_test.json',
                                                 format = 'json', fields = fields)
        text.build_vocab(train)
        label.build_vocab(train)
    if args.main == 'waseem-hovy':
        text = Field(tokenize = tokenizer, lower = True, batch_first = True)
        label = LabelField()
        fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
        train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem-hovy_train.json',
                                                 validation = 'waseem-hovy_dev.json', test = 'waseem-hovy_test.json',
                                                 format = 'json', fields = fields)
        text.build_vocab(train)
        label.build_vocab(train)
    if args.main == 'wulczyn':
        text = Field(tokenize = tokenizer, lower = True, batch_first = True)
        label = LabelField()
        fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
        train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
                                                 validation = 'wulczyn_dev.json', test = 'wulczyn_test.json',
                                                 format = 'json', fields = fields)
        text.build_vocab(train)
        label.build_vocab(train)
    if args.main == 'davidson':
        text = Field(tokenize = tokenizer, lower = True, batch_first = True)
        label = LabelField()
        fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
        train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_train.json',
                                                 validation = 'davidson_dev.json', test = 'davidson_test.json',
                                                 format = 'json', fields = fields)
        text.build_vocab(train)
        label.build_vocab(train)
    main = {'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label}

    aux = []
    for auxiliary in args.aux:
        if args.main == 'waseem':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                     validation = 'waseem_dev.json', test = 'waseem_test.json',
                                                     format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if args.main == 'waseem-hovy':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem-hovy_train.json',
                                                     validation = 'waseem-hovy_dev.json',
                                                     test = 'waseem-hovy_test.json',
                                                     format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if args.main == 'wulczyn':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
                                                     validation = 'wulczyn_dev.json', test = 'wulczyn_test.json',
                                                     format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if args.main == 'davidson':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_train.json',
                                                     validation = 'davidson_dev.json', test = 'davidson_test.json',
                                                     format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if auxiliary == 'hoover':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'hoover_train.json',
                                                     validation = 'hoover_dev.json', test = 'hoover_test.json',
                                                     format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if auxiliary == 'oraby_sarcasm':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'oraby_sarcasm_train.json',
                                                     validation = 'oraby_sarcasm_dev.json',
                                                     test = 'oraby_sarcasm_test.json', format = 'json', fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if auxiliary == 'oraby_factfeel':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'oraby_factfeel_train.json',
                                                     validation = 'oraby_factfeel_dev.json',
                                                     test = 'oraby_factfeel_test.json', format = 'json',
                                                     fields = fields)
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})
        if auxiliary == 'preotiuc':
            text = Field(tokenize = tokenizer, lower = True, batch_first = True)
            label = LabelField()
            fields = [('text', text), ('label', label)]  # Because we load from json we just need this.
            train, dev, test = TabularDataset.splits(args.datadir, train = 'preotiuc_train.json',
                                                     validation = 'preotiuc_dev.json',
                                                     test = 'preotiuc_test.json', format = 'json', fields = fields)
            aux.append({'train': train, 'dev': dev, 'test': test})
            text.build_vocab(train)
            label.build_vocab(train)
            aux.append({'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label})

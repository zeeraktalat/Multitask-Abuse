import os
import csv
import json
import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import mlearn.modeling.multitask as mtl
from mlearn.utils.metrics import Metrics
from mlearn.data.batching import TorchtextExtractor
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import run_mtl_model as run_model
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator

if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    # Data inputs and outputs
    parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                        type = str.lower, default = 'Davidson')
    parser.add_argument("--aux", help = "Specify the auxiliary datasets to be loaded.", type = str, nargs = '+',
                        default = ['hoover', 'waseem'])
    parser.add_argument("--binary", dest='binary',help = "Load binary versions of main task dataset?", default = False, type = bool)
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/json/')
    parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
    parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/models/')

    # Cleaning and metrics
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = ['lower'])
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                        type = str.lower)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'f1-score')
    parser.add_argument("--stop_metric", help = "Set the metric to be used for early stopping", default = "f1-score")

    # Experiment
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, Ekphrasis, BPE",
                        default = 'bpe', type = str.lower)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)

    # Modelling
    # All models
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = 'mlp',
                        type = str.lower)
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best", default = -1,
                        type = int)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Embedding, Tfidf, Count",
                        default = 'embedding', type = str.lower)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = int, default = 0)
    # LSTM
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    # CNN
    parser.add_argument('--window_sizes', help = "Set CNN window sizes.", default = "2,3,4", type = str)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128, type = int)

    # Hyper Parameters
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 64, type = int)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = "64,128,50", type = str)
    parser.add_argument("--shared", help = "Set the shared dimension", default = 64, type = int)  # TODO Fix in MTL code
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 5, type = int)
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 64, type = int)
    parser.add_argument("--nonlinearity", help = "Set nonlinearity function for neural nets.", default = 'tanh',
                        type = str.lower)
    parser.add_argument('--learning_rate', help = "Set the upper limit for the learning rate.", default = 1.0,
                        type = float)
    parser.add_argument("--dropout", help = "Set upper limit for dropout.", default = 1.0, type = float)

    # MTL specific
    parser.add_argument("--batches_epoch", help = "Set the number of batches per epoch", type = int, default = 20)
    parser.add_argument("--loss_weights", help = "Set the weight of each task", type = str, default = "1.0,0.5")
    parser.add_argument("--dataset_weights", help = "Set the probability for each task to be chosen.", type = str,
                        default = None)
    args = parser.parse_args()

    if 'f1' in args.metrics:
        args.metrics[args.metrics.index('f1')] = 'f1-score'
    if args.display == 'f1':
        args.display = 'f1-score'
    if args.stop_metric == 'f1':
        args.stop_metric = 'f1-score'

    # Set up WandB logging
    # wandb.init(settings=wandb.Settings(start_method='thread'))
    # wandb.init(settings=wandb.Settings(start_method='fork'))
    wandb.init(config = args)
    config = wandb.config

    if args.patience == -1: # If patience is not given then set the model to run all epochs.
        args.patience = config.epochs

    # Initialise random seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu != -1:
        torch.cuda.set_device(0)

    # Set up experiment and cleaner
    c = Cleaner(processes = args.cleaners)
    exp = 'word' if args.experiment != 'liwc' else args.experiment
    exp = Preprocessors('data/').select_experiment(exp)
    onehot = True if args.encoding == 'onehot' else False

    # Load tokenizers
    if args.tokenizer == 'spacy':
        selected_tok  = c.tokenize
    elif args.tokenizer == 'bpe':
        selected_tok = c.bpe_tokenize
    elif args.tokenizer == 'ekphrasis':
        ekphr = c.ekphrasis_tokenize
        annotate = {'elongated', 'emphasis'}
        flters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, flters)
        selected_tok = ekphr

        if args.experiment == 'liwc':
            def liwc_toks(doc):
                tokens = ekphr(doc)
                tokens = exp(tokens)
                return tokens
            selected_tok = liwc_toks
    tokenizer = selected_tok

    # Set up fields
    text = Field(tokenize = tokenizer, lower = True, batch_first = True)
    label = LabelField()
    fields = {'text': ('text', text), 'label': ('label', label)}  # Because we load from json we just need this.

    # Load main task training data
    if args.binary:
        if args.main == 'davidson':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_binary_train.json',
                                                     validation = 'davidson_binary_dev.json',
                                                     test = 'davidson_binary_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'wulczyn':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_binary_train.json',
                                                     validation = 'wulczyn_binary_dev.json',
                                                     test = 'wulczyn_binary_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'waseem':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_binary_train.json',
                                                     validation = 'waseem_binary_dev.json',
                                                     test = 'waseem_binary_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'waseem_hovy':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_hovy_binary_train.json',
                                                     validation = 'waseem_hovy_binary_dev.json',
                                                     test = 'waseem_hovy_binary_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
    else:
        if args.main == 'davidson':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_train.json',
                                                     validation = 'davidson_dev.json',
                                                     test = 'davidson_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'wulczyn':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
                                                     validation = 'wulczyn_dev.json',
                                                     test = 'wulczyn_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'waseem':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                     validation = 'waseem_dev.json',
                                                     test = 'waseem_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif args.main == 'waseem_hovy':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_hovy_train.json',
                                                     validation = 'waseem_hovy_dev.json',
                                                     test = 'waseem_hovy_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
    text.build_vocab(train)  # TODO This is where max_size should be set.
    label.build_vocab(train)
    main = {'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label, 'name': args.main, 'vocab': {key: item for key, item in text.vocab.stoi.items()}}

    # Dump Vocabulary files
    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.vocab', 'w',
              encoding = 'utf-8') as vocab_file:
        vocab_file.write(json.dumps(text.vocab.stoi))

    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.label', 'w',
              encoding = 'utf-8') as label_file:
        label_file.write(json.dumps(label.vocab.stoi))

    # Load aux tasks
    auxillary = []
    for i, aux in enumerate(args.aux):
        print(f"Loading dataset #{i}: {aux}")
        # Set up fields
        text = Field(tokenize = tokenizer, lower = True, batch_first = True)
        label = LabelField()
        fields = {'text': ('text', text), 'label': ('label', label)}  # Because we load from json we just need this.

        if aux == 'davidson':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_train.json',
                                                     validation = 'davidson_dev.json',
                                                     test = 'davidson_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'waseem':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                     validation = 'waseem_dev.json',
                                                     test = 'waseem_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'waseem_hovy':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_hovy_train.json',
                                                     validation = 'waseem_hovy_dev.json',
                                                     test = 'waseem_hovy_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'wulczyn':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
                                                     validation = 'wulczyn_dev.json',
                                                     test = 'wulczyn_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'hoover':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'hoover_train.json',
                                                     validation = 'hoover_dev.json',
                                                     test = 'hoover_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'oraby_factfeel':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'oraby_fact_feel_train.json',
                                                     validation = 'oraby_fact_feel_dev.json',
                                                     test = 'oraby_fact_feel_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        elif aux == 'oraby_sarcasm':
            train, dev, test = TabularDataset.splits(args.datadir, train = 'oraby_sarcasm_train.json',
                                                     validation = 'oraby_sarcasm_dev.json',
                                                     test = 'oraby_sarcasm_test.json',
                                                     format = 'json', skip_header = True, fields = fields)
        text.build_vocab(train)
        label.build_vocab(train)
        auxillary.append({'train': train,
                          'dev': dev,
                          'test': test,
                          'text': text,
                          'labels': label,
                          'name': aux,
                          'task_id': i + 1  # to prevent i == 0 == main task
                          })

        # Dump vocabs
        with open(f'{args.results}/vocabs/{aux}_{args.encoding}_{args.experiment}.vocab', 'w',
                  encoding = 'utf-8') as vocab_file:
            vocab_file.write(json.dumps(text.vocab.stoi))

        # Dump label vocabs
        with open(f'{args.results}/vocabs/{aux}_{args.encoding}_{args.experiment}.label', 'w',
                  encoding = 'utf-8') as label_file:
            label_file.write(json.dumps(label.vocab.stoi))

        if len(auxillary) == len(args.aux):
            break  # All datasets have been loaded.

    # Hyper parameters
    # Not in model info
    epochs = config.epochs
    batch_size = config.batch_size
    batch_epoch = args.batches_epoch
    loss_weights = [float(w) for w in config.loss_weights.split(',')]

    # In model info
    hidden = config.hidden
    shared = config.shared
    learning_rate = config.learning_rate
    dropout = config.dropout
    nonlinearity = config.nonlinearity if args.model != 'lstm' else 'tanh'

    params = dict(hidden_dims = [int(hid) for hid in hidden.split(',')],
                  shared_dim = shared,
                  dropout = dropout,
                  nonlinearity = nonlinearity,
                  batch_first = True,
                  input_dims = [len(main['vocab'])] + [len(aux['text'].vocab.stoi) for aux in auxillary],
                  output_dims = [len(main['labels'].vocab.stoi)] + [len(aux['labels'].vocab.stoi) for aux in auxillary],
                  )

    if not onehot:
        params.update({'embedding_dims': config.embedding})
    if args.model == 'lstm':
        params.update({'no_layers': args.layers})
        model = mtl.OnehotLSTMClassifier if onehot else mtl.EmbeddingLSTMClassifier
    else:
        params.update({'non-linearity': config.nonlinearity})

        if args.model == 'cnn':
            params.update({'window_sizes': [int(win) for win in config.window_sizes.split(',')],
                           'num_filters': config.filters})
            model = mtl.OnehotCNNClassifier if onehot else mtl.EmbeddingCNNClassifier
        elif args.model == 'mlp':
            model = mtl.OnehotMLPClassifier if onehot else mtl.EmbeddingMLPClassifier
    model = model(**params)

    # Info about losses: https://bit.ly/3irxvYK
    if args.loss == 'nlll':
        loss = torch.nn.NLLLoss()
    elif args.loss == 'crossentropy':
        loss = torch.nn.CrossEntropyLoss()

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum = 0.9)
    elif config.optimizer == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), learning_rate)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    # Batch data
    batchers = []
    test_batchers = []
    if not onehot:
        train_buckets = BucketIterator(dataset = main['train'], batch_size = batch_size, sort_key = lambda x: len(x))
        main_train = TorchtextExtractor('text', 'label', main['name'], train_buckets)
        batchers.append(main_train)

        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets)

        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, train = False, shuffle = False, sort_key = lambda x: len(x.text), sort = False)
        main_test = TorchtextExtractor('text', 'label', main['name'], test_buckets)

        for aux in auxillary:
            train_buckets = BucketIterator(dataset = aux['train'], batch_size = batch_size, sort_key = lambda x: len(x))
            train = TorchtextExtractor('text', 'label', aux['name'], train_buckets)
            batchers.append(train)

            test_buckets = BucketIterator(dataset = aux['test'], batch_size = 64, train = False, shuffle = False, sort_key = lambda x: len(x.text), sort = False)
            test = TorchtextExtractor('text', 'label', aux['name'], test_buckets)
            test_batchers.append(test)
    else:
        train_buckets = BucketIterator(dataset = main['train'], batch_size = batch_size, sort_key = lambda x: len(x))
        train = TorchtextExtractor('text', 'label', main['name'], train_buckets, len(main['text'].vocab.stoi))
        batchers.append(train)

        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets, len(main['text'].vocab.stoi))

        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, train = False, shuffle = False, sort_key = lambda x: len(x.text), sort = False)
        main_test = TorchtextExtractor('text', 'label', main['name'], test_buckets)

        for aux in auxillary:
            train_buckets = BucketIterator(dataset = aux['train'], batch_size = batch_size, sort_key = lambda x: len(x))
            train = TorchtextExtractor('text', 'label', aux['name'], train_buckets, len(aux['text'].vocab.stoi))
            batchers.append(train)

            test_buckets = BucketIterator(dataset = aux['test'], batch_size = 64, sort_key = lambda x: len(x.text), train = False, shuffle = False, sort = False)
            test = TorchtextExtractor('text', 'label', aux['name'], test_buckets)
            test_batchers.append(test)

    # Open output files
    base = f'{args.results}{args.main}_{args.encoding}_{args.experiment}_{args.tokenizer}'

    train_writer = csv.writer(open(f"{base}_train.trial.tsv", 'w', encoding = 'utf-8'), delimiter = '\t')
    dev_writer = csv.writer(open(f"{base}_dev.trial.tsv", 'w', encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{base}_test.trial.tsv", 'w', encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{base}_preds.trial.tsv", 'w', encoding = 'utf-8'), delimiter = '\t')
    batch_writer = csv.writer(open(f"{base}_batch.trial.tsv", 'w', encoding = 'utf-8'), delimiter = '\t')

    model_hdr = ['Model',
                 'Input dim',
                 'Embedding dim',
                 'Hidden dim',
                 'Output dim',
                 'Shared dim',
                 'Dropout',
                 'nonlinearity']

    metric_hdr = args.metrics + ['loss']
    hdr = ['Timestamp',
           'Main task',
           'Tasks',
           'Batch size',
           '# Epochs',
           'Learning rate',
           'Batches / epoch'] + model_hdr
    hdr += metric_hdr
    test_writer.writerow(hdr)  # Don't include dev columns when writing test
    hdr += [f"dev {m}" for m in metric_hdr]
    train_writer.writerow(hdr)

    # Batch hdr
    batch_hdr = ['Timestamp',
                 'Epoch',
                 'Batch',
                 'Task name',
                 'Main task',
                 'Batch size',
                 '# Epochs',
                 'Learning rate',
                 'Batches / epoch']
    batch_hdr += model_hdr + metric_hdr
    batch_writer.writerow(batch_hdr)

    # pred_metric_hdr = args.metrics + ['loss']
    # hdr = ['Timestamp', 'Main task', 'Batch size', '# Epochs', 'Learning Rate', 'Batches per epoch'] + model_hdr
    # hdr += ['Label', 'Prediction']
    hdr = ['Timestamp', 'Dataset', 'Prediction', 'Label']
    pred_writer.writerow(hdr)

    # Set args
    gpu = True if args.gpu != -1 else False

    train_dict = dict(save_path = f"{args.save_model}{args.experiment}_{args.tokenizer}_{args.main}_best",
                      hyperopt = True,
                      gpu = gpu,
                      shuffle = False,

                      # Hyper-parameters
                      clip = 1.0,
                      epochs = config.epochs,
                      early_stopping = args.patience,
                      low = True if args.stop_metric == 'loss' else False,
                      loss_weights = loss_weights,
                      dataset_weights = [float(w) for w in config.dataset_weights.split(',')],
                      batches_per_epoch = batch_epoch,

                      # Model definitions
                      model = model,
                      loss = loss,
                      optimizer = optimizer,

                      # Dataset
                      batchers = batchers,
                      metrics = Metrics(args.metrics, args.display, args.stop_metric),
                      dev = dev,
                      dev_metrics = Metrics(args.metrics, args.display, args.stop_metric)
                      )

    # Writing
    write_dict = dict(batch_writer = batch_writer,
                      writer = train_writer,
                      test_writer = dev_writer,
                      pred_writer = None,
                      main_name = main['name'],
                      data_name = "_".join([main['name']] + [aux['name'].split()[0] for aux in auxillary]),
                      model_hdr = model_hdr,
                      metric_hdr = args.metrics + ['loss'],
                      hyper_info = [batch_size, epochs, learning_rate, batch_epoch],
                      )
    start = time.time()
    run_model(train = True, **train_dict, **write_dict)
    end = time.time()
    wandb.log({'Training time (s)': end - start, 'Training time (m)': (end - start) / 60})
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(config.save_model, f'{args.model}.pt'))
    wandb.save(os.path.join(args.save_model, f'{args.model}.pt'))

    # Do tests
    test_metrics = Metrics(args.metrics, args.display, args.stop_metric)
    main_task_eval = dict(model = model,
                          batchers = main_test,
                          loss = loss,
                          metrics = test_metrics,
                          gpu = gpu,
                          mtl = 0,
                          store = True,
                          data = None,
                          writer = test_writer,
                          main_name = main['name'],
                          data_name = main['name'],
                          metric_hdr = args.metrics,
                          model_hdr = model_hdr,
                          hyper_info = [batch_size, epochs, learning_rate, batch_epoch],

                          # Torchtext specific things
                          pred_writer = pred_writer,
                          pred_path = f"{base}_preds.trial.tsv",
                          labels = main['labels']
                          )
    run_model(train = False, **main_task_eval)
    test_scores = {f"test/{main['name']}_{m}": value[-1] for m, value in test_metrics.scores.items() if m != 'loss'}
    wandb.log(test_scores)

    for task_ix, aux in enumerate(test_batchers):
        aux_metrics = Metrics(args.metrics, args.display, args.stop_metric)
        aux_dict = dict(model = model,
                        batchers = aux,
                        metrics = aux_metrics,
                        gpu = gpu,
                        mtl = auxillary[task_ix]['task_id'],
                        store = True,
                        data = None,
                        loss = loss,
                        writer = test_writer,
                        main_name = main['name'],
                        data_name = auxillary[task_ix]['name'],
                        metric_hdr = args.metrics,
                        model_hdr = model_hdr,
                        hyper_info = [batch_size, epochs, learning_rate, batch_epoch],

                        # Torchtext specific things
                        pred_writer = pred_writer,
                        pred_path = f"{base}_preds.trial.tsv",
                        labels = auxillary[task_ix]['labels']
                        )
        run_model(train = False, **aux_dict)
        aux_metrics = {f"test/{auxillary[task_ix]['name']}_{m}": value[-1] for m, value in aux_metrics.scores.items() if m != 'loss'}
        wandb.log(aux_metrics)

    if args.binary:
        # Add tests for runninng on all abuse datasets (that aren't in main or aux)
        abuse_datasets = ['wulczyn', 'waseem', 'waseem_hovy', 'garcia', 'davidson']
        out_of_domain_abuse = [ds for ds in abuse_datasets if ds not in [args.main] + args.aux]

        with torch.no_grad():  # Do evaluations
            predictions = defaultdict(lambda: defaultdict(list))
            eval_loop = tqdm(out_of_domain_abuse, desc = "Evaluation")

            for aux in eval_loop:
                eval_loop.set_postfix(dataset = aux)
                test_scores = Metrics(args.metrics, args.display, args.stop_metric)

                # Load AUX data
                aux_fp = open(os.path.join(args.datadir, f'{aux}_binary_test.json'), 'r', encoding = 'utf-8')
                aux_test, aux_labels, lens = [], [], []

                for line in tqdm(aux_fp, desc = "Loading data", leave = False):
                    line = json.loads(line)
                    aux_test.append(tokenizer(line['text']))
                    aux_labels.append(line['label'].strip('\r\n'))
                    lens.append(len(aux_test[-1]))

                # Tensorise
                max_len = max(lens)
                pretensors = []
                for label, doc in tqdm(zip(aux_labels, aux_test), desc = "Encoding data", leave = False):
                    # Tensorize data
                    indices = [main['vocab'].get(tok.lower(), main['vocab']['<pad>']) for tok in doc]
                    if len(indices) < max_len:
                        indices += (max_len - len(indices)) * [main['vocab'].get('<pad>', main['vocab']['<pad>'])]
                    pretensors.append(torch.tensor(indices, device = 'cpu').long())

                # Make batches
                test_batches = []
                preds = []
                for start_ix in tqdm(range(0, len(pretensors), 64), desc = "Run inference", leave = False):
                    test_batches = pretensors[start_ix:start_ix + 64]
                    batch_labels = aux_labels[start_ix:start_ix + 64]
                    tensor = torch.stack(test_batches, dim = 0)

                    if gpu:
                        tensor = tensor.cuda()
                    try:
                        pred = model(tensor, task_id=0)
                        pred = torch.argmax(pred, dim = 1)
                    except RuntimeError as e: # Catching this to prevent failing due to bigger kernel size than document.
                        if onehot:
                            tensor = onehot_encoder(pretensors, model_params['input_dim']).type(torch.long)
                        if gpu:
                            tensor = tensor.cuda()
                        result = model(tensor, task_id=0)
                        pred = torch.argmax(result, dim = 1)
                    preds.extend([main['labels'].vocab.itos[p] for p in pred])

                # Store predictions
                predictions[aux]['preds'] = preds
                predictions[aux]['true'] = aux_labels
                predictions[aux]['data'] = aux_test

                # Compute & store metrics
                predictions[aux]['scores'] = test_scores.compute(aux_labels, preds)
                wandb.log({f'test/{aux}_{score_n}': scores[-1] for score_n, scores in predictions[aux]['scores'].items() if score_n != 'loss'})

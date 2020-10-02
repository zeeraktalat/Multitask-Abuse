from tqdm import tqdm
import mlearn.data.loaders as loaders
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile

if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, Ekphrasis, BPE",
                        default = 'ekphrasis', type = str.lower)
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)
    parser.add_argument('--cfg', action = ActionConfigFile, default = None)

    args = parser.parse_args()

    # Initialize experiment
    c = Cleaner(args.cleaners)
    experiment = Preprocessors(args.datadir).select_experiment(args.experiment)

    if args.tokenizer == 'spacy':
        tokenizer = c.tokenize
    elif args.tokenizer == 'bpe':
        tokenizer = c.bpe_tokenize
    elif args.tokenizer == 'ekphrasis':
        tokenizer = c.ekphrasis_tokenize
        annotate = {'elongated', 'emphasis'}
        filters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, filters)

    datasets = []
    datasets.append(loaders.waseem(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                   stratify = 'label'))
    datasets.append(loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                        stratify = 'label'))
    datasets.append(loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                    stratify = 'label', skip_header = True))
    datasets.append(loaders.davidson(tokenizer, args.datadir, preprocessor = experiment, label_processor = None,
                                     stratify = 'label', skip_header = True))
    datasets.append(loaders.hoover(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                   skip_header = True))
    datasets.append(loaders.oraby_sarcasm(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                          skip_header = True))
    datasets.append(loaders.oraby_fact_feel(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                            skip_header = True))

    for dataset in tqdm(datasets, desc = "Dump datasets"):
        dataset.dump('train', args.datadir, 'json')
        dataset.dump('test', args.datadir, 'json')
        dataset.dump('dev', args.datadir, 'json')

import re
import spacy
from torchtext import data
import src.shared.types as types
from typing import List, Tuple, Dict, Union, Callable, Any


class BatchGenerator:
    """A class to get the information from the batches."""

    def __init__(self, dataloader, datafield, labelfield):
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = getattr(batch, self.df)
            y = getattr(batch, self.lf)
            yield (X, y)


class Dataset(data.TabularDataset):

    def __init__(self, data_dir: str, splits: Dict[str, str], ftype: str,
                 fields: List[Tuple[types.FieldType, ...]] = None, cleaners: List[str] = None,
                 batch_sizes: Tuple[int, ...] = (32), shuffle: bool = True, sep: str = 'tab', skip_header: bool = True,
                 repeat_in_batches = False, device: str = 'cpu'):
        """Initialise data class.
        :param data_dir (str): Directory containing dataset.
        :param fields (Dict[str, str]): The data fields in the file.
        :param cleaners List[str]: Cleaning operations to be taken.
        :param splits (str): Dictionary containing filenames type of data.
        :param ftype: File type of the data file.
        :param batch_sizes (Tuple[int]): Tuple of batch size for each dataset.
        :param shuffle (bool, default: True): Shuffle the data between each epoch.
        :param sep (str): Seperator (if csv/tsv file).
        :param repeat_in_batches (bool, default: False): Repeat data within batches
        :param device (str, default: 'cpu'): Device to process on
        """
        self.tagger = spacy.load('en', disable = ['ner'])
        num_files = len(splits.keys())  # Get the number of datasets.

        self.load_params = {'path': data_dir,
                            'format': ftype,
                            'fields': fields,
                            'skip_header': skip_header,
                            'num_files': num_files}
        self.load_params.update(splits)
        self.dfields = fields
        self.batch_sizes = batch_sizes
        self.repeat = repeat_in_batches
        self.cleaners = cleaners
        self.device = device

    @property
    def fields(self):
        return self.dfields

    @fields.setter
    def fields(self, fields):
        self.dfields = fields
        self.load_params.update({'fields': fields})

    @property
    def data_params(self):
        return self.load_params

    @data_params.setter
    def data_params(self, params):
        self.load_params.update(params)

    def load_data(self) -> Tuple[types.DataType, ...]:
        """Load the dataset and return the data.
        :return data: Return loaded data.
        """
        if self.load_params['num_files'] == 1:
            train = self._data(**self.load_params)
            self.data = (train, None, None)
        elif self.load_params['num_files'] == 2:
            train, test = self._data(**self.load_params)
            self.data = (train, None, test)
        elif self.load_params['num_files'] == 3:
            train, dev, test = self._data(**self.load_params)
            self.data = (train, dev, test)
        return self.data

    @classmethod
    def _data(cls, path: str, format: str, fields: Union[List[Tuple[types.FieldType, ...]], Dict[str, tuple]],
              train: str, validation: str = None, test: str = None, skip_header: bool = True,
              num_files: int = 3) -> Tuple[types.DataType, ...]:
        """Use the loader in torchtext.
        :param path (str): Directory the data is stored in.
        :param format (str): Format of the data.
        :param fields (Union[List[types.FieldType], Dict[str tuple]]): Initialised fields.
        :param train (str): Filename of the training data.
        :param validation (str, default: None): Filename of the development data.
        :param test (str, default: None): Filename of the test data.
        :param skip_header (bool, default: True): Skip first line.
        :param num_files (int, default: 3): Number of files/datasets to load.
        :return data: Return loaded data.
        """
        splitted = data.TabularDataset.splits(path = path, format = format, fields = fields, train = train,
                                              validation = validation, test = test, skip_header = skip_header)
        return splitted

    def clean_document(self, text: types.DocType, processes: List[str] = None):
        """Data cleaning method.
        :param text (types.DocType): The document to be cleaned.
        :param processes (List[str]): The cleaning processes to be undertaken.
        :return cleaned: Return the cleaned text.
        """
        cleaned = str(text)
        if 'lower' in self.cleaners or 'lower' in processes:
            cleaned = cleaned.lower()
        if 'url' in self.cleaners or 'url' in processes:
            cleaned = re.sub(r'https?:/\/\S+', 'URL', cleaned)
        if 'hashtag' in self.cleaners or 'hashtag' in processes:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', 'HASHTAG', cleaned)
        if 'username' in self.cleaners or 'username' in processes:
            cleaned = re.sub(r'@\S+', 'AT_USER', cleaned)

        return cleaned

    def tokenize(self, document: types.DocType, processes: List[str] = None):
        """Tokenize the document using SpaCy and clean it as it is processed.
        :param document: Document to be parsed.
        :param processes: The cleaning processes to engage in.
        :return toks: Document that has been passed through spacy's tagger.
        """
        if processes:
            toks = [tok.text for tok in self.tagger(self.clean_document(document, processes = processes))]
        else:
            toks = [tok.text for tok in self.tagger(self.clean_document(document))]
        return toks

    def generate_batches(self, sort_func: Callable, datasets: Tuple[types.DataType, ...] = None):
        """Create the minibatching here.
        :param train (types.DataType, optional): Provide a processed train dataset.
        :param test (types.DataType, optional): Provide a processed test dataset.
        :param dev (types.DataType, optional): Provide a processed test dataset.
        :return ret: Return the batched data.
        """
        if datasets:
            batches = data.BucketIterator.splits(datasets,
                                                 self.batch_sizes,
                                                 sort_key = sort_func,
                                                 device = self.device,
                                                 sort_within_batch = True, repeat = self.repeat)
        else:
            batches = data.BucketIterator.splits(self.data,
                                                 self.batch_sizes,
                                                 sort_key = sort_func,
                                                 device = self.device,
                                                 sort_within_batch = True, repeat = self.repeat)
        return batches

    def tag(self, document: types.DocType):
        """Tag document using spacy.
        :param document: Document to be parsed.
        :return doc: Document that has been passed through spacy's tagger.
        """
        doc = self.tagger(self.clean_document(document))
        return doc

    def get_spacy_annotations(self, document: types.DocType, processes: List[str]) -> Tuple:
        """Get annotations from SpaCy requested.
        :param document: The document to process.
        :param processes: The annotation processes to get.
        :return res (tuple): Tuple containing annotations requested.
        """
        res = [(tok.text, tok.pos_, tok.dep_, (tok.dep_, tok.head.dep_)) for tok in document]
        token, pos, dep, head = zip(*res)

        res = [None, None, None, None]

        if 'token' in processes:
            res[0] = token
        if 'pos' in processes:
            res[1] = pos
        if 'dep' in processes:
            res[2] = dep
        if 'head' in processes:
            res[3] = head
        if 'children' in processes:
            raise NotImplementedError

        return tuple(res)

    def set_field_attribute(self, field: Union[types.FieldType, List[types.FieldType]],
                            attribute: Union[str, List[str]],
                            val: Union[Any, List[Any]]):
        """Take an initialised field and an attribute.
        :param field (types.FieldType): The field to be modified.
        :param attribute (str): The attribute to modify.
        :param val (types.AllBuiltin): The new value of the attribute.
        """
        if isinstance(field, List):
            for f, a, v in zip(field, attribute, val):
                setattr(f, a, v)
        else:
            setattr(field, attribute, val)

import re
import src.shared.types as t


class Cleaner:
    """A class for methods for cleaning."""

    def __init__(self, cleaners: t.List[str] = None):
        """Initialise cleaner class.
        :param cleaners t.List[str]: Cleaning operations to be taken.
        """
        self.processes = cleaners

    def clean_document(self, text: t.DocType, processes: t.List[str] = None):
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

    def tokenize(self, document: t.DocType, processes: t.List[str] = None):
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

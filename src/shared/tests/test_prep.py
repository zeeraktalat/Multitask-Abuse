import unittest
from src.shared import prep
from src.shared import types


class DataPrepTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = prep.Dataset(data_dir = '~/Documents/PhD/projects/Generalisable_abuse/data',
                              fields = None,
                              splits = {'train': 'cyberbullying_dataset.csv',
                                        'test': 'test.csv',
                                        'validation': 'dev.csv'},
                              cleaners = ['lower', 'url', 'hashtag', 'username'],
                              batch_sizes = (32, 32, 32),
                              ftype = 'csv',
                              shuffle = True,
                              sep = ',',
                              repeat_in_batches = True)

    @classmethod
    def tearDownClass(cls):
        cls.ds = None

    def test_field_property(self):
        """Test the field property."""

        self.assertEqual(self.ds.fields, None)  # Test that it is initalised empty.

        # Set up fields
        text = types.text_data
        label = types.text_label
        self.ds.set_field_attribute(text, 'tokenize', self.ds.tokenize)

        fields = [('id', None),
                  ('bad_word', text),
                  ('question', text),
                  ('question_sentiment_gold', label),
                  ('answer', text),
                  ('answer_sentiment_gold', label),
                  ('username', text)]
        self.ds.fields = fields

        out = len([('id', None),
                   ('bad_word', '<torchtext.data.field.Field object at 0x11781ed50>'),
                   ('question', '<torchtext.data.field.Field object at 0x11781ed50>'),
                   ('question_sentiment_gold', '<torchtext.data.field.Field object at 0x117790d50>'),
                   ('answer', '<torchtext.data.field.Field object at 0x11781ed50>'),
                   ('answer_sentiment_gold', '<torchtext.data.field.Field object at 0x117790d50>'),
                   ('username', '<torchtext.data.field.Field object at 0x11781ed50>')])

        self.assertEqual(len(self.ds.fields), out)  # Test that fields have been set.

    def test_set_field_attribute(self):
        """Test that the field attribute can be set."""
        raise NotImplementedError

    def test_load_data(self):
        """Test the data loading."""
        raise NotImplementedError

    def test_clean_document(self):
        """Test that the document cleaning works."""
        raise NotImplementedError

    def test_generate_batches(self):
        """Test that batches are generated and accessible."""
        raise NotImplementedError

    def test_tag(self):
        """Test that the tagger works."""
        raise NotImplementedError

    def test_spacy_annotations(self):
        """Test that that extractions of annotations work."""
        raise NotImplementedError

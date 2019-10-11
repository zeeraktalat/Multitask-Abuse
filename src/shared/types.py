import torch
import spacy
import numpy as np
from torch.nn import Module
from torchtext.data import Field
from sklearn.base import ClassifierMixin, TransformerMixin
from typing import *


# Data types
FieldType = Field
DataType = Union[list, np.ndarray, torch.LongTensor]
DocType = Union[str, list, spacy.tokens.doc.Doc]

# Model/Vectorizer Type
ModelType = Union[ClassifierMixin, Module]
VectType = TransformerMixin


# Set up label types for data processing
text_label = Field(sequential = False,
                   include_lengths = False,
                   use_vocab = True,
                   pad_token = None,
                   unk_token = None)

int_label = Field(sequential = False,
                  include_lengths = False,
                  use_vocab = True,
                  pad_token = None,
                  unk_token = None)

text_data = Field(sequential = True,
                  include_lengths=True,
                  use_vocab=True)

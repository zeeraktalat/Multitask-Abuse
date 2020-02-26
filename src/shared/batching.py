import torch
import numpy as np

from scipy.sparse import issparse
from torch.autograd import Variable


class Batcher:

    def __init__(self, total_size, size):
        self.total_size = total_size
        self.size = size
        self.pointer = 0

        # if isinstance(self.data, coo_matrix):
        #     self.data = self.data.tocsr()

    def next_loop(self):
        if self.pointer == self.total_size:
            self.pointer = 0
        return self.__next__()

    def __next__(self):
        if self.pointer == self.total_size:
            self.pointer = 0
            raise StopIteration

        next_pointer = min(self.total_size, self.pointer+self.size)

        start, end = self.pointer, next_pointer

        self.pointer = next_pointer
        return end-start, start, end

    def __iter__(self):
        return self


def splen(data):
    try:
        return data.shape[0]
    except:
        return len(data)


def prepare_with_labels(data, labels, binary=True):
    # Note, we should just be passing in a sparse minibatch here!
    # Doing todense on the entire datset is silly
    if issparse(data):
        data = data.todense()

    v = torch.FloatTensor(np.array(data))
    # if gpu():
    #     return Variable(v.cuda()), Variable(torch.LongTensor(labels).cuda())
    # print(labels)
    # print(Variable(torch.FloatTensor(labels)))
    if binary:
        return Variable(v), Variable(torch.FloatTensor(np.array(labels)))

        # return Variable(v), Variable(torch.LongTensor(np.array(labels)))
        # return Variable(v), Variable(labels)
    else:
        return Variable(v), Variable(torch.FloatTensor(np.array(labels)))

def prepare(data):
    # Note, we should just be passing in a sparse minibatch here!
    # Doing todense on the entire datset is silly
    if issparse(data):
        data = data.todense()
    v = torch.FloatTensor(np.array(data))
    # if gpu():
    #     return Variable(v.cuda())
    return Variable(v)

import torch.nn as nn
import src.shared.types as t


class MTLLSTMClassifier(nn.Module):

    def __init(self, hidden_dims: t.List[int], input_dims: t.List[int], embedding_dim: t.List[int],
               output_dims: t.List[int], no_layers: int, dropout: int = 0.2):
        """Initialise the LSTM.
        :param hidden_dim (t.List[int]): The dimensionality of the hidden dimensions for each task.
        :param input_dim: The dimensionality of the input to the embedding generation.
        :param embedding_dim: The dimensionality of the the produced embeddings.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        :param dropout: Value fo dropout
        """
        super(MTLLSTMClassifier, self).__init__()

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        # Input layer (not shared) [Linear]
        # Hidden layer (shared) [LSTM]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Linear(input_dim, hidden_dims[task_id])
            self.inputs[task_id] = layer
            self.all_parameters.append(layer.weight)

        self.shared = []
        for i in range(len(hidden_dims) - 1):
            all_layers, layer = nn.LSTM(hidden_dims[i], hidden_dims[i + 1])
            self.shared.append(layer)
            self.all_parameters.append(layer.weight)

        self.outputs = {}
        for task_id, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dim, output_dims[task_id])
            self.outputs[task_id] = layer
            self.all_parameters.append(layer.weight)

        # Define layers of the network
        # self.i2e = nn.Embedding(input_dim, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # self.h2o = nn.Linear(hidden_dim, no_classes)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim = 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, task_id):
        """The forward step in the classifier.
        :param sequence: The sequence to pass through the network.
        :param task_id: The task on which to perform forward pass.
        :return scores: The "probability" distribution for the classes.
        """

        res = self.dropout(self.inputs[task_id](sequence))

        for layer in self.shared:
            res = self.dropout(layer(res))

        output = self.outputs[task_id](res)

        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist

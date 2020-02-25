import torch.nn as nn
import src.shared.types as t


class MTLLSTMClassifier(nn.Module):

    def __init(self, input_dims: t.List[int], shared_dim: int, hidden_dims: t.List[int], output_dims: t.List[int],
               no_layers: int = 1, dropout: int = 0.2):
        """Initialise the LSTM.
        :param input_dim: The dimensionality of the input.
        :param shared_dim: The dimensionality of the shared layers.
        :param hidden_dim (t.List[int]): The dimensionality of the hidden dimensions for each task.
        :param embedding_dim: The dimensionality of the the produced embeddings.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        :param dropout: Value fo dropout
        """
        super(MTLLSTMClassifier, self).__init__()

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) != len(hidden_dims)

        # Input layer (not shared) [Linear]
        # hidden to hidden layer (shared) [Linear]
        # Hidden to hidden layer (not shared) [LSTM]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Linear(input_dim, shared_dim)
            self.inputs[task_id] = layer
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.shared = []
        for i in range(len(hidden_dims) - 1):
            all_layers, layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.lstm.append(layer)
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.lstm = {}
        for task_id, input_dim in range(len(hidden_dims) - 1):
            all_layers, layer = nn.LSTM(hidden_dims[i], hidden_dims[i + 1])
            self.lstm[task_id][layer]
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.outputs = {}
        for task_id, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dim, output_dims[task_id])
            self.outputs[task_id] = layer
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim = 1)
        self.dropout = nn.Dropout(dropout)

        # TODO Ensure that the model is deterministic (the bias term is added)
        print(self)
        print(list(self.all_parameters))

    def forward(self, sequence, task_id):
        """The forward step in the classifier.
        :param sequence: The sequence to pass through the network.
        :param task_id: The task on which to perform forward pass.
        :return scores: The "probability" distribution for the classes.
        """

        res = self.inputs[task_id](sequence)
        res = self.dropout(res)

        for layer in self.shared:
            res = self.dropout(layer(res))

        lstm_out, _ = self.lstm[task_id](res)

        output = self.outputs[task_id](lstm_out.view(len(sequence), -1))

        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist

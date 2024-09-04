"""
This module defines the IntentClassifier class and model architecture, 
which is a Bidirectional LSTM-based neural network model designed for 
intent detection in text. The model uses a series of LSTM layers 
and linear layers with GELU activations and dropout for regularization.
The model configuration can be customized via the LSTMConfig and 
LinearConfig classes.
"""

import torch
from torch import nn
from torch.nn import init


class IntentClassifier(nn.Module):
    """
    A Bidirectional LSTM-based classifier for intent detection. This model 
    consists of an embedding layer, a bidirectional LSTM, and multiple 
    linear layers with GELU activation functions and dropout for 
    regularization.

    Args:
        config (dict): Configuration dictionary containing model parameters 
        such as:
            - 'input_size' (int): Size of the input feature vector.
            - 'hidden_size' (int): # of features in the LSTM's hidden state.
            - 'num_layers' (int): # of recurrent layers in the LSTM.
            - 'dropout_lstm' (float): Dropout for the LSTM layers.
            - 'dropout_linear' (float): Dropout for the linear layers.
            - 'num_labels' (int): # of output classes for the classification.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding.from_pretrained(
          '/data/embedding_matrix.pt', freeze=True)

        self.bi_lstm = nn.LSTM(config['input_size'], 
                               config['hidden_size'],
                               config['num_layers'], 
                               dropout=config['dropout_lstm'],
                               batch_first=True, bidirectional=True)

        self.dropout_bilstm = nn.Dropout(p=config['dropout_lstm'])
        self.act1 = nn.GELU()

        self.linear1 = nn.Linear(config['hidden_size'] * 2, 
                                 config['hidden_size'] - 100)
        
        self.norm1 = nn.LayerNorm(config['hidden_size'] - 100)
        self.dropout_linear1 = nn.Dropout(p=config['dropout_linear'])
        self.act_lin1 = nn.GELU()

        self.linear2 = nn.Linear(config['hidden_size'] - 100, 
                                 config['hidden_size'] // 2)
        
        self.norm2 = nn.LayerNorm(config['hidden_size'] // 2)
        self.dropout_linear2 = nn.Dropout(p=config['dropout_linear'])
        self.act_lin2 = nn.GELU()

        self.output = nn.Linear(config['hidden_size'] // 2, 
                                config['num_labels'])

        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    init.xavier_uniform_(param)
                elif 'linear' in name:
                    init.xavier_normal_(param)

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of the model. Processes the input through the embedding 
        layer, bidirectional LSTM, and a series of linear layers to 
        produce class scores.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).
            h0 (torch.Tensor, optional): Initial hidden state for the LSTM. 
                Defaults to None, which initializes it as zeros.
            c0 (torch.Tensor, optional): Initial cell state for the LSTM. 
                Defaults to None, which initializes it as zeros.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels) 
            containing the class scores for each input sequence in the batch.
        """
        if h0 is None:
            h0 = torch.zeros(self.config['num_layers'] * 2, x.size(0), 
                             self.config['hidden_size'])
        if c0 is None:
            c0 = torch.zeros(self.config['num_layers'] * 2, x.size(0), 
                             self.config['hidden_size'])

        x = self.embedding(x)

        x, _ = self.bi_lstm(x, (h0, c0))
        x = self.act1(self.dropout_bilstm(x))

        x = self.linear1(x[:, -1, :])
        x = self.norm1(x)
        x = self.act_lin1(self.dropout_linear1(x))

        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act_lin2(self.dropout_linear2(x))

        x = self.output(x)

        return x
import torch 
import torch.nn as nn 
import torch.nn.init as init
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, 'embedding_matrix.pt')
embedding_matrix = torch.load(file_path)

class IntentClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        self.bi_lstm = nn.LSTM(config['input_size'], config['hidden_size'],
                             config['num_layers'], dropout=config['dropout_lstm'],
                             batch_first=True, bidirectional=True)

        self.dropout_bilstm = nn.Dropout(p=config['dropout_lstm'])
        self.act1 = nn.GELU()

        self.linear1 = nn.Linear(config['hidden_size'] * 2, config['hidden_size'] - 100)
        self.norm1 = nn.LayerNorm(config['hidden_size'] - 100)
        self.dropout_linear1 = nn.Dropout(p=config['dropout_linear'])
        self.act_lin1 = nn.GELU()

        self.linear2 = nn.Linear(config['hidden_size'] - 100, config['hidden_size'] // 2)
        self.norm2 = nn.LayerNorm(config['hidden_size'] // 2)
        self.dropout_linear2 = nn.Dropout(p=config['dropout_linear'])
        self.act_lin2 = nn.GELU()

        self.output = nn.Linear(config['hidden_size'] // 2, config['num_labels'])

        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    init.xavier_uniform_(param)
                elif 'linear' in name:
                    init.xavier_normal_(param)


    def forward(self, x, h0=None, c0=None):
      if h0 is None:
          h0 = torch.zeros(self.config['num_layers'] * 2, x.size(0), self.config['hidden_size'])
      if c0 is None:
          c0 = torch.zeros(self.config['num_layers'] * 2, x.size(0), self.config['hidden_size'])

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



    
    
    
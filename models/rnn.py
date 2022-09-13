'''RNN in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import BertTokenizer, BertModel
#bert = BertModel.from_pretrained('bert-base-uncased')
class RNNb(nn.Module): #Transformer
    def __init__(self):
        hidden_dim,output_dim,n_layers,bidirectional,dropout = 256,1,2,True,0.25
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,hidden_dim,num_layers = n_layers,bidirectional = bidirectional,batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        for name, param in self.named_parameters():                
            if name.startswith('bert'):
                param.requires_grad = False
    def forward(self, text, text_lengths):
        #text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        #hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        #hidden = [batch size, hid dim]
        output = self.out(hidden)
        #output = [batch size, out dim]
        return output
        """#embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded) #1layerLSTM
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        assert torch.equal(output[:,-1,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))"""
class RNN3(nn.Module): #Fasttext
    def __init__(self):
        vocab_size, embedding_dim, output_dim, pad_idx = 25002, 100, 1, 1
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        #embedded = [sent len, batch size, emb dim]
        #embedded = embedded.permute(1, 0, 2)
        #embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        #pooled = [batch size, embedding_dim]
        return self.fc(pooled)
class RNN1(nn.Module): #forcifar
    def __init__(self):
        super(RNN, self).__init__() # if use nn.RNN(), it hardly learns
        #self.cnn = nn.Conv2d(3, 64, kernel_size=(32,1), stride=(1,1), padding=0)
        self.rnn = nn.GRU(input_size=32*3, hidden_size=512, num_layers=5, batch_first=True, bidirectional=False)
        self.out = nn.Linear(512, 10)
    def forward(self, x):
        #x = self.cnn(x)
        #x = x.squeeze()
        #x = x.permute(0,2,1)
        x = x.reshape(-1,32,32*3)

        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        #r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        r_out, h_n = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

class RNN(nn.Module):
    def __init__(self):
        vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx = 25002, 100, 256, 1, 2, True, 0.5, 1
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        #embedded = self.dropout(embedded)
        #embedded = [sent len, batch size, emb dim]
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        #packed_output, (hidden, cell) = self.rnn(packed_embedded)
        packed_output, hidden = self.rnn(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell   = [num layers * num directions, batch size, hid dim]
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        #hidden = self.dropout(hidden)
        #hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)




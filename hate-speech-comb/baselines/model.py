import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        text_feat = torch.sum(text_emb, dim=1)
        text_out = self.fc(text_feat)
        return text_out


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        bidirectional=True,
        drop_p=0.5,
        pad_token_id=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, text_len.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, : self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_feat = self.drop(out_reduced)

        text_out = self.fc(text_feat)
        return text_out

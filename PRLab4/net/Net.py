import torch
from torch import nn

from net import SentimentAnalysisSettings
from net import SinusoidalPredictionSettings


class SinusoidalPredictionRNN(nn.Module):
    def __init__(self):
        super(SinusoidalPredictionRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=SinusoidalPredictionSettings.input_size,
            hidden_size=SinusoidalPredictionSettings.hidden_size,
            num_layers=SinusoidalPredictionSettings.num_layers,
            batch_first=SinusoidalPredictionSettings.batch_first,
        )
        self.out = nn.Linear(SinusoidalPredictionSettings.hidden_size, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


class SentimentAnalysisRNN(nn.Module):
    def __init__(self):
        super(SentimentAnalysisRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=SentimentAnalysisSettings.input_size,
            hidden_size=SentimentAnalysisSettings.hidden_size,
            num_layers=SentimentAnalysisSettings.num_layers,
            batch_first=SentimentAnalysisSettings.batch_first
        )
        self.out = nn.Linear(SentimentAnalysisSettings.hidden_size, 1)

    def forward(self, x):
        r_out, s = self.rnn(x)
        out = self.out(r_out[:, -1, :])
        return out

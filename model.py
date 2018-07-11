import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        
        embeddings = torch.cat((features, embeddings), 1)
        
        lstm_out, self.hidden = self.lstm(embeddings)
        
        output = self.linear(lstm_out)

        return output

    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # input should be (1, 1, embed_size)
        preds = []
        prediction = None
        max_len = 20
        count = 0
        states = None
        embeddings = inputs

        while count < max_len and prediction != 1:
            lstm_out, states = self.lstm(embeddings, states)
            output = self.linear(lstm_out)

            max_p, pred = output.max(2)

            prediction = pred.item()
            preds.append(prediction)
            embeddings = self.embed(pred)

            count += 1

        return preds

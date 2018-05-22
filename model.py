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

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        #print("Captions      ======", captions.shape)
        embeddings = self.embed(captions)
        #print("Embeddings    ======", embeddings.shape)

        features = features.unsqueeze(1)
        #print("Features      ======", features.shape)
        embeddings = torch.cat((features, embeddings), 1)
        #print("Concat        ======", embeddings.shape)

        lstm_out, self.hidden = self.lstm(embeddings)
        #print("LSTM out      ======", lstm_out.shape)

        output = self.linear(lstm_out)

        output = output[:, 1:, :].contiguous() # do not return CNN output (<start> word)

        return output

    def sample(self, inputs):
        # input should be (1, 1, embed_size)

        preds = []
        pred = None
        max_len = 10
        count = 0

        states = None

        while count < max_len:
            hidden, states = self.lstm(inputs, states)
            output = self.linear(hidden)
            _, pred = output.max(2)
            inputs = self.embed(pred)
            print(pred.shape)
            print(inputs.shape)

            preds.append(pred.item())
            count += 1

        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        return preds

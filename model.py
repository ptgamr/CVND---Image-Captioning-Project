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
        # print("Captions      ======", captions.shape)
        embeddings = self.embed(captions[:, :-1])
        # print("Embeddings    ======", embeddings.shape)

        features = features.unsqueeze(1)
        #print("Features      ======", features.shape)
        embeddings = torch.cat((features, embeddings), 1)
        #print("Concat        ======", embeddings.shape)

        lstm_out, self.hidden = self.lstm(embeddings)
        #print("LSTM out      ======", lstm_out.shape)

        output = self.linear(lstm_out)

        # output = output[:, 1:, :].contiguous() # do not return CNN output (<start> word)

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
            #print("embedding shape ============", embeddings.shape)
            #print(embeddings)
            lstm_out, states = self.lstm(embeddings, states)
            output = self.linear(lstm_out)

            #print("output =====================", output.shape)
            # output = output[:, -1:, :]

            max_p, pred = output.max(2)

            #print("prediction =================", max_p.data[0].tolist())
            #print("index ======================", pred.data[0].tolist())
            #print("\n")
            prediction = pred.item()
            preds.append(prediction)
            embeddings = self.embed(pred)

            count += 1

        return preds

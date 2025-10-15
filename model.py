import torch
import torch.nn as nn
import torchvision.models as models

# ------------------ Encoder ------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # freeze pretrained ResNet

        modules = list(resnet.children())[:-1]  # remove final fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)  # [batch, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # flatten
        features = self.bn(self.linear(features))
        return features  # [batch, embed_size]

# ------------------ Decoder ------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        features: [batch, embed_size]
        captions: [batch, seq_len]
        """
        embeddings = self.embed(captions)  # [batch, seq_len, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # prepend image feature
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs  # [batch, seq_len+1, vocab_size]

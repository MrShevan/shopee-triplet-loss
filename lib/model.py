import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 192, 3, stride=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(192, 192, 1, stride=1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.Conv2d(192, 384, 3, stride=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(384, 384, 1, stride=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 256, 1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

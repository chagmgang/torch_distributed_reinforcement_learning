import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(4, 64)
        self.l2 = nn.Linear(64, 64)

        self.p1 = nn.Linear(64, 2)

        self.v1 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))

        p = self.p1(x)

        v = self.v1(x)

        prob = self.softmax(p)

        return prob, v

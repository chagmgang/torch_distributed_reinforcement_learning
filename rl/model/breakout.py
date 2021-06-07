import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueModel(nn.Module):

    def __init__(self):
        super(ValueModel, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 512)

        self.value = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
        

if __name__ == '__main__':

    state = torch.ones([1, 4, 84, 84])
    model = ValueModel()
    model(state)

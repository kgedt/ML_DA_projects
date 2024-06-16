import torch


class DepressiiNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=3)
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0
        )
        self.act1 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=6)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, 1, 0)
        self.act2 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.BatchNorm2d(num_features=12)
        self.conv3 = torch.nn.Conv2d(12, 24, 3, 1, 0)
        self.act3 = torch.nn.LeakyReLU()
        self.bn4 = torch.nn.BatchNorm2d(num_features=24)
        self.conv4 = torch.nn.Conv2d(24, 48, 3, 1, 0)
        self.act4 = torch.nn.LeakyReLU()
        self.mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = torch.nn.BatchNorm2d(num_features=48)
        self.fc1 = torch.nn.Linear(108 * 108 * 48, 24)
        self.act5 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(24, 8)
        self.soft = torch.nn.Softmax()

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn4(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.mp(x)
        x = self.bn5(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act5(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEmotion(nn.Module):
    def __init__(self, num_classes, regularization_lambda=0.001):
        super(DeepEmotion, self).__init__()

        self.regularization_lambda = regularization_lambda

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3)

        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, num_classes)
        

        self.local_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        self.local_fc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2),
        )

        self.local_fc[2].weight.data.zero_()
        self.local_fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def stn(self, x):
        xs = self.local_net(x)
        xs = xs.view(-1, 640)       # 10 * 3 * 3
        theta = self.local_fc(xs)
        theta = theta.view(-1, 2, 3)

        # grid = F.affine_grid(theta, x.size(), align_corners=True)
        # x = F.grid_sample(x, grid, align_corners=True)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def calculate_regularization_loss(self):
        # Calculate the L2 regularization loss for the weights in the last two fully-connected layers
        regularization_loss = 0.0
        for param in self.fc1.parameters():
            regularization_loss += torch.norm(param, p=2)  # L2 norm
        for param in self.fc2.parameters():
            regularization_loss += torch.norm(param, p=2)  # L2 norm

        return regularization_loss

    def compute_loss(self, outputs, labels):
        # Cross-entropy loss
        classification_loss = nn.CrossEntropyLoss()(outputs, labels)

        # L2 regularization loss
        # regularization_loss = self.calculate_regularization_loss()

        # Total loss with regularization
        # total_loss = classification_loss + self.regularization_lambda * regularization_loss

        return classification_loss

    def forward(self, x):
        grid = self.stn(x)
        localization_grid_resized = F.interpolate(grid, size=(9, 9), mode='bilinear', align_corners=False)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))

        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.relu(self.maxpool4(x))

        x = F.dropout(x)

        x = x * localization_grid_resized

        x = x.view(-1, 810)
        x = self.fc1(x)
        x = self.fc2(x)

        # x = F.softmax(x, dim=1)

        return x
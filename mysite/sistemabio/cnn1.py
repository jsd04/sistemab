#CONVOLUTIONAL NETWORK 
from torch import nn
# pip install torchsummary
from torchsummary import summary

# crear una red neuronal convolucional
class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax

        # conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        
        # conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # flatten
        self.flatten = nn.Flatten()
        # linear
        self.linear = nn.Linear(128 * 5 * 4, 10)
        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        """
        definicion del proceso de forward en la red neuronal
        """

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    # instanciar la red neuronal convolucional
    cnn = CNNNetwork()
    # mostrar la informacion de la red neuronal convolucional
    summary(cnn.cpu(), (1, 64, 44))

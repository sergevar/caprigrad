import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer1 = self._block(1, 2)  # Start from single channel, output 2 channels
        self.layer2 = self._block(2, 4)  # Double the channel count

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer1(x)
        print(f"After first block:\n{x}")

        x = self.layer2(x)
        print(f"After second block:\n{x}")

        return x
    
# Initialize the model
model = SimpleCNN()

# Set a manual seed for reproducible results
torch.manual_seed(42)

# Initialize the weights of the model
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)

# Define a (1, 8, 8) sized tensor as input to represent a single channel 8x8 image
# The tensor is initialized with manually defined values
input_tensor = torch.FloatTensor([[
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 9],
    [3, 4, 5, 6, 7, 8, 9, 10],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [5, 6, 7, 8, 9, 10, 11, 12],
    [6, 7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 12, 13, 14, 15]
]])

# Propagate the input_tensor through the model
output = model(input_tensor.unsqueeze(0))

# Print the output
print("Output:")
print(output)
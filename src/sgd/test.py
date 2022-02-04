import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sgd import SGD as SGD_cpu

torch.manual_seed(0)

# Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# print(net)

# Loss Function
## Let’s try a random 32x32 input
input = torch.randn(1, 1, 32, 32)
output = net(input)
# print(out)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

## Optimizer - Select C++ extension (CPU) or native PyTorch version
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# optimizer = SGD_cpu(net.parameters(), lr=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.01, dampening=0.0, nesterov=True)
optimizer = SGD_cpu(net.parameters(), lr=0.01, momentum=0.01, dampening=0.0, nesterov=True)

# Training loop
for i in range(5):
    print(i)
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

    print('** Model data **')
    print('w:', net.conv1.weight)
    print('b:', net.conv1.bias)
    # print('w.grad:', net.weight.grad)
    # print('b.grad:', net.bias.grad)
    print('** Model prediction **')
    print(output)
import torch
from conv import Conv

torch.manual_seed(0)
N = 100
x = torch.rand([N, 1, 4, 4])*5
# Let the following command be the true function
# y = 2.3 + 5.1*x
# Get some noisy observations
y_obs = 0.2*torch.randn([N, 1, 3, 3])

# w = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)

in_ch = 1
out_ch = 2
k_s = 4

# conv_cpp = Conv(in_channels=in_ch, out_channels=out_ch, kernel_size=k_s, stride=1, padding=1)
conv_torch = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_s, stride=1, padding=1)
conv_cpp = Conv(in_channels=in_ch, out_channels=out_ch, kernel_size=k_s, stride=1, padding=1)

# set same internal parameters
conv_torch.weight = conv_cpp.weight
conv_torch.bias = conv_cpp.bias

# Train both models
gamma = 0.01
for i in range(5):
    # Forward pass
    print(i)
    conv_cpp.zero_grad()
    conv_torch.zero_grad()
    # use new weight to calculate loss
    y_pred_cpp = conv_cpp(x)
    y_pred = conv_torch(x)

    # Loss computing
    mse_cpp = torch.mean((y_pred_cpp - y_obs) ** 2)
    mse = torch.mean((y_pred - y_obs) ** 2)

    # Backward pass
    mse.backward()
    mse_cpp.backward()
    print('C++ extension data')
    print('w_cpp:', conv_cpp.weight)
    print('b_cpp:', conv_cpp.bias)
    print('w_cpp.grad:', conv_cpp.weight.grad)
    print('b_cpp.grad:', conv_cpp.bias.grad)
    print()
    print('PyTorch data')
    print('w:', conv_torch.weight)
    print('b:', conv_torch.bias)
    print('w.grad:', conv_torch.weight.grad)
    print('b.grad:', conv_torch.bias.grad)

    # gradient descent, don't track
    with torch.no_grad():
        conv_cpp.weight -= gamma*conv_cpp.weight.grad
        conv_cpp.bias -= gamma*conv_cpp.bias.grad
        conv_torch.weight -= gamma*conv_torch.weight.grad
        conv_torch.bias -= gamma*conv_torch.bias.grad

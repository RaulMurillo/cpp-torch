import torch
from linear import Linear

torch.manual_seed(0)
N = 100

in_ch = 10
out_ch = 5

x = torch.rand([N, in_ch])*5
y_obs = 2.3 + 5.1*torch.randn([N, out_ch])

linear_cpp = Linear(in_features=in_ch, out_features=out_ch)
linear_torch = torch.nn.Linear(in_features=in_ch, out_features=out_ch)

# set same internal parameters
linear_torch.weight = linear_cpp.weight
linear_torch.bias = linear_cpp.bias

loss_fn = torch.nn.MSELoss()


# Train both models
gamma = 0.01
for i in range(5):
    # Forward pass
    print(i)
    linear_cpp.zero_grad()
    linear_torch.zero_grad()
    # use new weight to calculate loss
    y_pred_cpp = linear_cpp(x)
    y_pred = linear_torch(x)

    # Loss computing
    mse_cpp = loss_fn(y_pred_cpp, y_obs)
    mse = loss_fn(y_pred, y_obs)

    # Backward pass
    mse.backward()
    mse_cpp.backward()
    print('** C++ extension data **')
    print('w_cpp:', linear_cpp.weight)
    print('b_cpp:', linear_cpp.bias)
    print('w_cpp.grad:', linear_cpp.weight.grad)
    print('b_cpp.grad:', linear_cpp.bias.grad)
    print()
    print('** PyTorch data **')
    print('w:', linear_torch.weight)
    print('b:', linear_torch.bias)
    print('w.grad:', linear_torch.weight.grad)
    print('b.grad:', linear_torch.bias.grad)

    # gradient descent, don't track
    with torch.no_grad():
        linear_cpp.weight -= gamma*linear_cpp.weight.grad
        linear_cpp.bias -= gamma*linear_cpp.bias.grad
        linear_torch.weight -= gamma*linear_torch.weight.grad
        linear_torch.bias -= gamma*linear_torch.bias.grad

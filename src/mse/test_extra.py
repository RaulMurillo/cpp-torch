##############
# Extra Test #
##############
'''
Instructions: run this test file twice, setting the LOSS_EXT
variable equal to `True` and `False` at each execution.
Then, compare both outputs, and assert they are equal.
'''
import torch
from mse import MSELoss

torch.manual_seed(0)
N = 100
in_ch = 8
out_ch = 5

LOSS_EXT = True

x = torch.rand([N, 2, in_ch]) * 5
y_obs = 2.3 + 5.1*torch.randn([N, 2, out_ch])

model = torch.nn.Linear(in_features=in_ch, out_features=out_ch)

loss_torch = torch.nn.MSELoss()
loss_cpu = MSELoss()

ITERS = 10
# Train both models
gamma = 0.01
for i in range(ITERS):
    # Forward pass
    print(i)
    model.zero_grad()
    # use new weight to calculate loss
    y_pred = model(x)

    # Loss computing
    mse_torch = loss_torch(y_pred, y_obs)
    mse_cpp = loss_cpu(y_pred, y_obs)

    print()
    print('** PyTorch data **')
    print('loss:', mse_torch)
    print()
    print('** C++ extension data **')
    print('loss:', mse_cpp)
    print()

    # Backward pass - only apply one backward!
    if LOSS_EXT:
        mse_cpp.backward()
    else:
        mse_torch.backward()
    print('** Model data **')
    print('w:', model.weight)
    print('b:', model.bias)
    print('w.grad:', model.weight.grad)
    print('b.grad:', model.bias.grad)

    # gradient descent, don't track
    with torch.no_grad():
        model.weight -= gamma*model.weight.grad
        model.bias -= gamma*model.bias.grad

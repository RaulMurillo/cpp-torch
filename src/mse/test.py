import torch
from mse import MSELoss as MSELoss_cpu

torch.manual_seed(0)

y = torch.ones([2, 3], requires_grad=True)
y_pred = y+torch.randn([2, 3])

# Different losses
mse_mean_loss = torch.nn.MSELoss(reduction='mean')
mse_sum_loss = torch.nn.MSELoss(reduction='sum')
mse_none_loss = torch.nn.MSELoss(reduction='none')

mse_mean_loss_cpu = MSELoss_cpu(reduction='mean')
mse_sum_loss_cpu = MSELoss_cpu(reduction='sum')
mse_none_loss_cpu = MSELoss_cpu(reduction='none')


mean_loss = mse_mean_loss(y_pred, y)
sum_loss = mse_sum_loss(y_pred, y)
none_loss = mse_none_loss(y_pred, y)

print('mean_loss:', mean_loss)
print('sum_loss:', sum_loss)
print('none_loss:', none_loss)

mean_loss_cpu = mse_mean_loss_cpu(y_pred, y)
sum_loss_cpu = mse_sum_loss_cpu(y_pred, y)
none_loss_cpu = mse_none_loss_cpu(y_pred, y)

print('mean_loss_cpu:', mean_loss_cpu)
print('sum_loss_cpu:', sum_loss_cpu)
print('none_loss_cpu:', none_loss_cpu)


mean_grad = mean_loss.backward()
sum_grad = sum_loss.backward()
none_grad = none_loss.backward()

print('mean_grad:', mean_grad)
print('sum_grad:', sum_grad)
print('mean_grad:', mean_loss.grad)
print('sum_grad:', mean_loss.grad)
# print('none_grad:', none_grad)

mean_grad_cpu = mean_loss_cpu.backward()
sum_grad_cpu = sum_loss_cpu.backward()
# none_grad_cpu = none_loss_cpu.backward()

print('mean_grad_cpu:', mean_grad_cpu)
print('sum_grad_cpu:', sum_grad_cpu)
print('mean_grad_cpu:', mean_loss_cpu.grad)
print('sum_grad_cpu:', mean_loss_cpu.grad)
# print('none_grad:', none_grad)


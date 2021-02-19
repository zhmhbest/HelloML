import torch

x = torch.ones(2, 2, requires_grad=True)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

y = 2 * torch.pow(x + 2, 2)
# tensor([[18., 18.],
#         [18., 18.]], grad_fn=<MulBackward0>)

out = y.mean()
# tensor(18., grad_fn=<MeanBackward0>)

out.backward(torch.tensor(1))
print(x.grad)
# tensor([[3., 3.],
#         [3., 3.]])

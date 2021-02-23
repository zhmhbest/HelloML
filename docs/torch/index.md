<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Pytorch](../index.html)

[TOC]

## 自动微分

#### demo1

```python
import torch

x = torch.tensor(2., dtype=torch.float, requires_grad=True)
print(x)
# tensor(2., requires_grad=True)

y = 4 * torch.pow(x + 1, 2) + 3
print(y)
# tensor(39., grad_fn=<AddBackward0>)

y.backward()  # = y.backward(torch.tensor(2.))
print(x.grad)
# tensor(24.)
```

$y = 4(x+1)^2 + 3$

$
    \dfrac{{\rm d}y}{{\rm d}x} = 8(x + 1)
$；$
    \dfrac{{\rm d}y}{{\rm d}x}\bold{|}_{x=2} = 24.0
$

#### demo2

```python
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

y = 3 * torch.pow(x + 1, 2)
print(y)
# tensor([[12., 12.],
#         [12., 12.]], grad_fn=<MulBackward0>)

out = y.mean()
print(out)
# tensor(12., grad_fn=<MeanBackward0>)

out.backward()  # = out.backward(torch.tensor(1.))
print(x.grad)
# tensor([[3., 3.],
#         [3., 3.]])
```

$
    x
    = \left[\begin{array}{c}
            x_1 & x_2
        \\  x_3 & x_4
    \end{array}\right]
$

$
    y
    = 3(x+1)^2
    = \left[\begin{array}{c}
            y_1 & y_2
        \\  y_3 & y_4
    \end{array}\right]
$

$
    o
    = \dfrac{1}{4} \sum\limits_{i} y_{i}
    = \dfrac{1}{4} \sum\limits_{i} 3(x_i+1)^2
$

$
    \dfrac{∂o}{∂x_i} = \dfrac{3}{2}(x_i+1)
$；$
    \dfrac{∂o}{∂x_i} \bold{|}_{x_i=1}
    = 3.0
$

# autograd

```python
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
```

$
    x = \left[\begin{array}{c}
            x_1 & x_2
        \\  x_3 & x_4
    \end{array}\right]
    = \left[\begin{array}{c}
            1.0 & 1.0
        \\  1.0 & 1.0
    \end{array}\right]
$

$
    y
    = \left[\begin{array}{c}
            y_1 & y_2
        \\  y_3 & y_4
    \end{array}\right]
    = 2(x+2)^2
    = \left[\begin{array}{c}
            18.0 & 18.0
        \\  18.0 & 18.0
    \end{array}\right]
$

$
    \begin{array}{l}
        o \\\\\\\\
    \end{array}
    \begin{array}{l}
            = \dfrac{1}{4} y = \dfrac{1}{2} (x+2)^2
    \\\\    = \dfrac{1}{4} \sum\limits_{i} y_{i} = 18.0
    \end{array}
$

$
    \dfrac{∂o}{∂x} = x+2
$

$
    \dfrac{∂o}{∂x} \bold{|}_{x=1}
    = \left[\begin{array}{c}
            3.0 & 3.0
        \\  3.0 & 3.0
    \end{array}\right]
$

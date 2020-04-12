import torch
from torchselfintersection import cpu

from torchselfintersection.selfintersections import SelfIntersections


a = torch.rand(40, 9)
x = cpu.forward(a)
m_py = SelfIntersections()
x_py = m_py(a)
print(a.shape, x.shape)
print(x)
print(x_py)
# print(torch.allclose(x, x_py))
print((x - x_py).abs().sum())

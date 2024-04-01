import torch

# Ensure that we're using PyTorch in a way that matches your C++ computations
torch.set_grad_enabled(True)

# a(logits) and c are both independently requiring gradient computation
a = torch.tensor([5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)
c = torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 1010.0], requires_grad=True)

softmax_a = torch.nn.functional.softmax(a, dim=0)
softmax_a.retain_grad()
result = torch.sum(softmax_a * c)
result.retain_grad()

# Backward operation
result.backward()

# Print the values
print("a: ", a)
print("c: ", c)
print("softmax_a: ", softmax_a)
print("result: ", result)

# Print the gradients
print("softmax_a.grad: ", softmax_a.grad)
print("result.grad: ", result.grad)
print("a.grad: ", a.grad)
print("c.grad: ", c.grad)

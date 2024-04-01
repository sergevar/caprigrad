import torch
import torch.nn.functional as F

# Create a tensor
a = torch.tensor([5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)

# Apply GELU
gelu_a = F.gelu(a, approximate="none")

# Print input (a) and output (gelu_a) tensor values
print("Input Tensor 'a': ", a)
print("Output Tensor after GELU: ", gelu_a)

# Perform backward pass
gelu_a.backward(torch.ones_like(a))

# Print gradients of a
print("Gradient w.r.t 'a' after backward pass: ", a.grad)

##########

a = torch.tensor([5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)

gelu_approx = F.gelu(a, approximate="tanh")

# Print input (a) and output (gelu_a) tensor values
print("Output Tensor after GELU Approx: ", gelu_approx)

# Perform backward pass
gelu_approx.backward(torch.ones_like(a))

# Print gradients of a
print("Gradient w.r.t 'a' after backward pass: ", a.grad)

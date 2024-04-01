import torch

# Assuming input is your input tensor
input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

# Forward pass
sum_sq = torch.sum(input**2)
mean_sq = sum_sq / input.numel()
scale = 1.0 / torch.sqrt(mean_sq + 1e-6)
output = input * scale
print("FORWARD:")
print("Output:\n", output)
output = output.sum()  # Needed to get a scalar for backward pass

# Backward pass
output.backward()
grad = input.grad
print("BACKWARD:")
print("Gradient:\n", grad)
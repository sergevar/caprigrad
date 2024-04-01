import torch

# Ensure that we're using PyTorch in a way that matches your C++ computations
torch.set_grad_enabled(True)

# a(logits) and c are both independently requiring gradient computation
logits = torch.tensor([5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True)
labels_one_hot = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)  # suppose class '2' is correct

# Compute softmax of logits
softmax_logits = torch.nn.functional.softmax(logits, dim=0)

# Compute cross entropy loss manually
loss = -torch.sum(labels_one_hot * torch.log(softmax_logits))  # this is the formula for one-hot cross entropy

# Backward operation
loss.backward()

# Print the values
print("Logits: ", logits)
print("One-hot labels: ", labels_one_hot)
print("softmax logits: ", softmax_logits)
print("Loss: ", loss)

# Print the gradient
print("Logits.grad: ", logits.grad)
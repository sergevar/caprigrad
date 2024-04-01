import torch

# Ensure that we're using PyTorch in a way that matches your C++ computations
torch.set_grad_enabled(True)

# a(logits) and c are both independently requiring gradient computation
logits = torch.tensor([
    [5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [-5.0, -3.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
], requires_grad=True)
labels_one_hot = torch.tensor([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
], dtype=torch.float, requires_grad=True)  # suppose class '2' is correct

# Compute softmax of logits
softmax_logits = torch.nn.functional.softmax(logits, dim=1)
softmax_logits.retain_grad()

# Compute cross entropy loss manually
# loss = -torch.sum(labels_one_hot * torch.log(softmax_logits))  # this is the formula for one-hot cross entropy
loss1 = -torch.sum(labels_one_hot[0] * torch.log(softmax_logits[0]))  # this is the formula for one-hot cross entropy
loss2 = -torch.sum(labels_one_hot[1] * torch.log(softmax_logits[1]))  # this is the formula for one-hot cross entropy
# avg
loss = (loss1 + loss2) / 2.0

loss1.retain_grad()
loss2.retain_grad()
loss.retain_grad()

# Backward operation
loss.backward()

to_print = {
    logits: "logits",
    labels_one_hot: "labels_one_hot",
    softmax_logits: "softmax_logits",
    loss1: "loss1",
    loss2: "loss2",
    loss: "loss"
}

# Print the values
for tensor, name in to_print.items():
    print(name, ": ", tensor)

# Print the gradient
for tensor, name in to_print.items():
    print(name, ".grad: ", tensor.grad)
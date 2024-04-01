import torch

# Ensure that we're using PyTorch in a way that matches your C++ computations
torch.set_grad_enabled(True)


    # ffml_set_data(M1, {0,0,0,0}, 1.0f);
    # ffml_set_data(M1, {0,0,1,0}, 2.0f);
    # ffml_set_data(M1, {0,0,2,0}, 3.0f);
    # ffml_set_data(M1, {0,1,0,0}, 4.0f);
    # ffml_set_data(M1, {0,1,1,0}, 5.0f);
    # ffml_set_data(M1, {0,1,2,0}, 6.0f);
    # ffml_set_data(M1, {1,0,0,0}, 7.0f);
    # ffml_set_data(M1, {1,0,1,0}, 8.0f);
    # ffml_set_data(M1, {1,0,2,0}, 9.0f);
    # ffml_set_data(M1, {1,1,0,0}, 10.0f);
    # ffml_set_data(M1, {1,1,1,0}, 11.0f);
    # ffml_set_data(M1, {1,1,2,0}, 12.0f);
    # ffml_set_data(M1, {2,0,0,0}, 13.0f);
    # ffml_set_data(M1, {2,0,1,0}, 14.0f);
    # ffml_set_data(M1, {2,0,2,0}, 15.0f);
    # ffml_set_data(M1, {2,1,0,0}, 16.0f);
    # ffml_set_data(M1, {2,1,1,0}, 17.0f);
    # ffml_set_data(M1, {2,1,2,0}, 18.0f);
    
    # ffml_set_data(M2, {0,0,0,0}, 0.11f);
    # ffml_set_data(M2, {0,0,1,0}, 0.22f);
    # ffml_set_data(M2, {0,1,0,0}, 0.33f);
    # ffml_set_data(M2, {0,1,1,0}, 0.44f);
    # ffml_set_data(M2, {0,2,0,0}, 0.55f);
    # ffml_set_data(M2, {0,2,1,0}, 0.66f);
    # ffml_set_data(M2, {1,0,0,0}, 0.77f);
    # ffml_set_data(M2, {1,0,1,0}, 0.88f);
    # ffml_set_data(M2, {1,1,0,0}, 0.99f);
    # ffml_set_data(M2, {1,1,1,0}, 1.10f);
    # ffml_set_data(M2, {1,2,0,0}, 1.21f);
    # ffml_set_data(M2, {1,2,1,0}, 1.32f);
    # ffml_set_data(M2, {2,0,0,0}, 1.43f);
    # ffml_set_data(M2, {2,0,1,0}, 1.54f);
    # ffml_set_data(M2, {2,1,0,0}, 1.65f);
    # ffml_set_data(M2, {2,1,1,0}, 1.76f);
    # ffml_set_data(M2, {2,2,0,0}, 1.87f);
    # ffml_set_data(M2, {2,2,1,0}, 1.98f);

    # for (uint64_t i = 0; i < 3; i++) {
    #     for (uint64_t j = 0; j < 2; j++) {
    #         for (uint64_t k = 0; k < 2; k++) {
    #             ffml_set_data(multiplicator, {i,j,k,0}, i+j+k);
    #         }
    #     }
    # }


# a(logits) and c are both independently requiring gradient computation
M1 = torch.tensor([
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
], requires_grad=True)
M2 = torch.tensor([
    [[0.11, 0.22], [0.33, 0.44], [0.55, 0.66]],
    [[0.77, 0.88], [0.99, 1.10], [1.21, 1.32]],
    [[1.43, 1.54], [1.65, 1.76], [1.87, 1.98]]
], requires_grad=True)

multiplicator = torch.tensor([
    [[0.0, 1.0], [1.0, 2.0]],
    [[1.0, 2.0], [2.0, 3.0]],
    [[2.0, 3.0], [3.0, 4.0]]
], requires_grad=True)

mm = M1 @ M2
mm.retain_grad()

out = mm * multiplicator
out.retain_grad()

# Backward operation
out.sum().backward()

# Print the values
print("M1: ", M1)
print("M1.grad: ", M1.grad)
print("M2: ", M2)
print("M2.grad: ", M2.grad)
print("multiplicator: ", multiplicator)
print("multiplicator.grad: ", multiplicator.grad)
print("mm: ", mm)
print("mm.grad: ", mm.grad)
print("out: ", out)
print("out.grad: ", out.grad)

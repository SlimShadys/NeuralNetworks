import torch

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print('Cuda available: {}'.format(torch.cuda.is_available()))
    print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
    print("===================================================")

# A Tensor with a Gradient Function
X = torch.randn((3,2), requires_grad=True)

print(F"X dtype: {X.dtype}")
print(F"X shape: {X.shape}")

# Another Tensor starting from X
Y = ((X.cos()) @ X.T**2).sum()

print(F"Y: {Y}")
print(F"Y Gradient: {Y.grad()}")
print(F"Y Gradient Function: {Y.grad_fn}")

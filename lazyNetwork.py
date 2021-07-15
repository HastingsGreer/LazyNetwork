import footsteps
import torch

if __name__ == "__main__":
    x = torch.randn((10, 10)).cuda()
    print(x.cpu())


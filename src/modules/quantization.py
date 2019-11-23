import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantization(nn.Module):
    def __init__(self, num_embedding):
        super(Quantization, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = 1
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)

    def forward(self, input):
        permutation = [0] + [2 + i for i in range(input.dim() - 2)] + [1]
        x = input.permute(permutation)
        flat_x = x.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        encoding = torch.argmin(distances, dim=1)
        encoding = encoding.reshape(x.size()[:-1])
        # encoding[:,:8].clamp_(0, 1)
        # encoding[:, 8:12].clamp_(0, 3)
        # encoding[:, 12:16].clamp_(0, 7)
        encoding[:,:1].clamp_(0, 1)
        encoding[:, :2].clamp_(0, 3)
        encoding[:, :3].clamp_(0, 5)
        encoding[:, :4].clamp_(0, 7)
        quantized = self.embedding(encoding)
        unpermutation = [0, input.dim() - 1] + [1 + i for i in range(input.dim() - 2)]
        quantized = quantized.permute(unpermutation)
        loss = F.mse_loss(quantized, input.detach()) + F.mse_loss(quantized.detach(), input)
        quantized = input + (quantized - input).detach()
        return quantized, encoding, distances, loss
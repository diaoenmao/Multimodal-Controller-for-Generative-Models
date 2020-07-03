import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from modules import VectorQuantization
from .utils import init_param

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.activation = Activation(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            Normalization(hidden_size),
            Activation(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            Normalization(hidden_size),
        )

    def forward(self, input):
        output = self.conv(input)
        output = self.activation(output + input)
        return output


class Encoder(nn.Module):
    def __init__(self, data_shape, hidden_size, num_res_block, embedding_size):
        super().__init__()
        blocks = [
            nn.Conv2d(data_shape[0], hidden_size, 4, 2, 1),
            Normalization(hidden_size),
            Activation(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            Normalization(hidden_size),
            Activation(inplace=True),
        ]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size))
        blocks.append(nn.Conv2d(hidden_size, embedding_size, 3, 1, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = input
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, data_shape, hidden_size, num_res_block, embedding_size):
        super().__init__()
        blocks = [nn.Conv2d(embedding_size, hidden_size, 3, 1, 1),
                  Normalization(hidden_size),
                  Activation(inplace=True)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size))
        blocks.extend([
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            Normalization(hidden_size),
            Activation(inplace=True),
            nn.ConvTranspose2d(hidden_size, data_shape[0], 4, 2, 1),
            nn.Tanh()
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = input
        x = self.blocks(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, data_shape=(3, 32, 32), hidden_size=128, num_res_block=2, embedding_size=64,
                 num_embedding=512, vq_commit=0.25):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.num_res_block = num_res_block
        self.embedding_size = embedding_size
        self.vq_commit = vq_commit
        self.encoder = Encoder(data_shape, hidden_size, num_res_block, embedding_size)
        self.quantizer = VectorQuantization(embedding_size, num_embedding)
        self.decoder = Decoder(data_shape, hidden_size, num_res_block, embedding_size)

    def encode(self, input):
        x = self.encoder(input)
        encoded, vq_loss, code = self.quantizer(x)
        return encoded, vq_loss, code

    def decode(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

    def decode_code(self, code):
        quantized = self.quantizer.embedding_code(code).transpose(1, -1).contiguous()
        decoded = self.decoder(quantized)
        return decoded

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        encoded, vq_loss, output['code'] = self.encode(x)
        decoded = self.decode(encoded)
        output['img'] = decoded
        output['loss'] = F.mse_loss(decoded, input['img']) + self.vq_commit * vq_loss
        return output


def vqvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['hidden_size']
    num_res_block = cfg['num_res_block']
    embedding_size = cfg['embedding_size']
    num_embedding = cfg['num_embedding']
    vq_commit = cfg['vq_commit']
    model = VQVAE(data_shape=data_shape, hidden_size=hidden_size, num_res_block=num_res_block,
                  embedding_size=embedding_size, num_embedding=num_embedding, vq_commit=vq_commit)
    model.apply(init_param)
    return model
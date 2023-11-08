import torch
import torch.nn as nn
import torch.nn.functional as F


class UnconditionalEncDecMHA(nn.Module):
    def __init__(self, time_dimension, device='cpu'):
        super(UnconditionalEncDecMHA, self).__init__()
        self.time_dimension = time_dimension
        self.bilstm = nn.LSTM(9, 96, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(256, 64)
        self.mha = nn.MultiheadAttention(256, 8)
        self.device = device

    def pos_encoding(self, t):
        channels = self.time_dimension
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        pos_enc_b = torch.cos(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, X, t):
        # Encode t using provided function pos_encoding and reshape
        X = X.squeeze()
        batch_size = X.shape[0]
        encoded_time = self.pos_encoding(t)

        # Transform X using bilstm and concatenate with encoded_time
        h, _ = self.bilstm(X)
        concat_h = torch.cat([encoded_time, h], dim=-1)

        # Pass concatenated tensor through MultiheadAttention
        h2, _ = self.mha(concat_h.permute(1, 0, 2), concat_h.permute(1, 0, 2), concat_h.permute(1, 0, 2))
        h2 = h2.permute(1, 0, 2)

        # Decode using linear layer
        out = self.linear(h2)

        return out


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                 groups=1, eps=1e-5, momentum=0.1):
        super(ConvBNGELU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class MHAEncoder(nn.Module):
    def __init__(self, time_embedding_dim):
        super(MHAEncoder, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.conv1 = ConvBNGELU(1, 128, 3, padding=1)
        self.mha1 = nn.MultiheadAttention(self.time_embedding_dim + 128, 4, batch_first=True)
        self.conv2 = ConvBNGELU(self.time_embedding_dim + 128, 512, 3, padding=1)
        self.conv3 = ConvBNGELU(512, 512, 3, padding=1)

    def forward(self, x, t):
        batch_size = x.shape[0]
        x = self.conv1(x)
        t = t.repeat(1, 1, 9, 64)
        x = torch.cat((x, t), dim=1)  # concatenate along channel axis
        total_channels = x.shape[1]
        """
        input to mha -> (sequence_length, batch_size, embedding_size)

        For example for images:
        For each image extract image patches and flatten them. Your batch size is the number of images. 
        Your sequence length is the number of patches per image. Your feature size is the length of a flattened patch.
        
        
        1.) 9 x 64 -mha1-> h1 -> mha2 -> y_hat(epsilon hat) (9 x 64)
        
        2.) 9x64 -linear-> 128x64 -concat time(t dim)-> 128+t x 64 -> mha1 -> 256 -linear-> 9 x 64

        """
        x = x.reshape((batch_size, total_channels, -1))
        x = x.permute(0, 2, 1)
        x, _ = self.mha1(x, x, x)
        x = x.permute(0, 2, 1)
        x = x.reshape((batch_size, total_channels, 9, 64))
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class EncDecWithMHA(nn.Module):
    def __init__(self, time_embedding_dim, device='cpu', time_dimension=128):
        super(EncDecWithMHA, self).__init__()

        self.device = device
        self.time_dimension = time_dimension

        # Time embedding layer
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU()
        )

        # Encoder
        self.encoder = MHAEncoder(self.time_dimension)

        # Decoder
        self.decoder = nn.Sequential(
            ConvBNGELU(512, 256, 3, padding=1),
            ConvBNGELU(256, 64, 3, padding=1),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def pos_encoding(self, t):
        channels = self.time_dimension
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = t.unsqueeze(-1)
        t = t.float()
        t = self.pos_encoding(t)
        t = t[:, :, None, None]

        # Encoder block
        x = self.encoder(x, t)

        # Decoder block
        x = self.decoder(x)

        return x


class EncoderDecoderBN(nn.Module):
    def __init__(self):
        super(EncoderDecoderBN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ConvBNGELU(2, 64, 3, padding=1),
            ConvBNGELU(64, 128, 3, padding=1),
            ConvBNGELU(128, 128, 3, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            ConvBNGELU(128, 64, 3, padding=1),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = t/1000
        t = t[:, None]
        t = t.repeat_interleave(9*64).view(batch_size, 9, 64)
        t = t[:, None, :, :]
        x = torch.concat((x, t), dim=1)

        # Encoder block
        x = self.encoder(x)

        # Decoder block
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    net = EncoderDecoderBN()
    # net = UNetConditional(num_classes=10, device="cpu")
    num_model_params = (sum([p.numel() for p in net.parameters()]))
    print(f"Number of model parameters = {num_model_params}")
    x = torch.randn(5, 3, 64, 64) # b x c x w x h
    t = x.new_tensor([10] * x.shape[0]).long()
    y = net(x, t)
    print(y)
    print(x.shape)
    print(t.detach().numpy())
    print(net(x, t).shape)

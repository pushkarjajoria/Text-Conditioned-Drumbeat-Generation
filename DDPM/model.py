import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


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
        h, _ = self.bilstm(X) # MIDI Features -> 1 vector
        concat_h = torch.cat([encoded_time, h], dim=-1)

        # Pass concatenated tensor through MultiheadAttention
        h2, _ = self.mha(concat_h.permute(1, 0, 2), concat_h.permute(1, 0, 2), concat_h.permute(1, 0, 2))
        h2 = h2.permute(1, 0, 2)

        # Decode using linear layer
        out = self.linear(h2)

        return out


class ConditionalEncDecMHA(nn.Module):
    def __init__(self, time_dimension, text_embedding_dim, device, lstm_embedding_dim=96):
        super(ConditionalEncDecMHA, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.lstm_hidden_size = lstm_embedding_dim
        self.time_dimension = time_dimension
        self.bilstm = nn.LSTM(9, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.linear_size = (self.lstm_hidden_size*2) + self.time_dimension + self.text_embedding_dim
        self.linear = nn.Linear(self.linear_size, 9)
        self.mha = nn.MultiheadAttention(self.linear_size, 8)
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

    def forward(self, X, t, text_embedding):
        # Encode t using provided function pos_encoding and reshape
        if len(X.shape) > 3:
            X = X.squeeze().permute(0, 2, 1)  # Batch x 64(seq_len) x 9(features/instruments)
        else:
            X = X.permute(0, 2, 1)
        encoded_time = self.pos_encoding(t)

        # Transform X using bilstm and concatenate with encoded_time
        h, _ = self.bilstm(X)
        time_text_context = torch.cat([encoded_time, text_embedding], dim=-1)
        # Change the shape from batch x features to batch x seq_len x features
        time_text_context = time_text_context[:, None, :].expand(-1, 128, -1)
        # Also need to concat the text embeddings here
        mha_input = torch.cat([h, time_text_context], dim=-1)
        # Use this as the input to the MHA

        # Pass concatenated tensor through MultiheadAttention
        mha_input = mha_input.permute(1, 0, 2)
        h2, _ = self.mha(mha_input, mha_input, mha_input)
        h2 = h2.permute(1, 0, 2)

        # Decode using linear layer
        out = self.linear(h2)

        return out


class ConditionalUNet(nn.Module):
    def __init__(self, z_dim=64, text_embedding_dim=64, time_dim=1, hidden_dim=256):
        super(ConditionalUNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.z_dim = z_dim
        self.text_embedding_dim = text_embedding_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.linear_encoder = nn.Sequential(
            nn.Linear(z_dim + text_embedding_dim + self.time_dim, hidden_dim),
            nn.ReLU()
        )
        self.down1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.sa1 = MultiheadAttention(hidden_dim * 2, num_heads=4)
        # Bottom of U-Net, deepest layer
        self.bot = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        # Simulate up-sampling
        self.up1 = nn.Linear(hidden_dim * 4, (hidden_dim * 2) - (time_dim + text_embedding_dim))
        self.sa2 = MultiheadAttention(hidden_dim * 2, num_heads=4)
        # Output transformation to predict noise
        self.outc = nn.Linear(hidden_dim * 2, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),  # Predicting epsilon of the same dimension as Z (256 -> 128)
        )

    def pos_encoding(self, t):
        channels = self.time_dim
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        pos_enc_b = torch.cos(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, z, t, text_embedding):
        # Embed time
        t_encoded = self.pos_encoding(t)

        # Concatenate z, text_embedding, and t_emb
        combined_context = torch.cat([text_embedding, t_encoded], dim=-1)
        combined_input = torch.cat([z, combined_context], dim=-1)

        # Encode
        encoded_input = self.linear_encoder(combined_input)  # (batch, 128) output
        z2 = self.down1(encoded_input)
        z2, _ = self.sa1(z2, z2, z2)
        # Bottom of U-Net, processing...
        z_bot = self.bot(z2)

        # Up-sampling with integration of timestep in each step if necessary
        z_up1 = self.up1(z_bot)
        z_up1_wt_context = torch.cat([z_up1, combined_context], dim=-1)
        z_up1, _ = self.sa2(z_up1_wt_context, z_up1_wt_context, z_up1_wt_context)
        # Continue with up-sampling...

        epsilon = self.outc(z_up1)
        return epsilon


class ConditionalLatentEncDecMHA(nn.Module):
    def __init__(self, time_dimension, text_embedding_dim, device, lstm_embedding_dim=96):
        super(ConditionalLatentEncDecMHA, self).__init__()
        self.text_embedding_dim = text_embedding_dim
        self.lstm_hidden_size = lstm_embedding_dim
        self.time_dimension = time_dimension
        self.bilstm = nn.LSTM(9, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.linear_size = (self.lstm_hidden_size*2) + self.time_dimension + self.text_embedding_dim
        self.linear = nn.Linear(self.linear_size, 9)
        self.mha = nn.MultiheadAttention(self.linear_size, 8)
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

    def forward(self, X, t, text_embedding):
        # Encode t using provided function pos_encoding and reshape
        if len(X.shape) > 3:
            X = X.squeeze().permute(0, 2, 1)  # Batch x 64(seq_len) x 9(features/instruments)
        else:
            X = X.permute(0, 2, 1)
        encoded_time = self.pos_encoding(t)

        # Transform X using bilstm and concatenate with encoded_time
        h, _ = self.bilstm(X)
        time_text_context = torch.cat([encoded_time, text_embedding], dim=-1)
        # Change the shape from batch x features to batch x seq_len x features
        time_text_context = time_text_context[:, None, :].expand(-1, 128, -1)
        # Also need to concat the text embeddings here
        mha_input = torch.cat([h, time_text_context], dim=-1)
        # Use this as the input to the MHA

        # Pass concatenated tensor through MultiheadAttention
        mha_input = mha_input.permute(1, 0, 2)
        h2, _ = self.mha(mha_input, mha_input, mha_input)
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
    net = ConditionalUNet()
    z_t = torch.rand((32, 128))
    t = torch.randint(1, 999, (32, ))
    text_emb = torch.rand((32, 64))
    noise = net(z_t, t, text_emb)
    print("Done")

import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DDPM.main import load_or_process_dataset
import torch.nn as nn


class PhaseShiftedMultiResolutionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(PhaseShiftedMultiResolutionLSTM, self).__init__()
        self.num_resolutions = 3  # For 32nd notes, 16th notes, 8th notes
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
                                          for _ in range(self.num_resolutions)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_resolutions):
            resolution_outputs = []
            # For each stride, also change the starting position
            for start in range(0, 2 ** i):
                sliced_input = x[:, start::2 ** i, :]
                lstm_out, _ = self.lstm_layers[i](sliced_input)
                resolution_outputs.append(lstm_out)
            # Combine outputs for different starts at the same resolution
            outputs.append(
                torch.cat(resolution_outputs, dim=1))  # Assuming concatenation along the sequence length dimension
        return outputs


class MultiResolutionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(MultiResolutionLSTM, self).__init__()
        self.num_resolutions = 3  # 32th notes, 16th notes, 8th notes
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
                                          for _ in range(self.num_resolutions)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_resolutions):
            # Apply LSTM with different stride to capture different resolutions
            lstm_out, _ = self.lstm_layers[i](x[:, ::2 ** i, :])  # Stride by 2^i
            outputs.append(lstm_out)
        return outputs


class NoFeatureExEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim_size, dropout=0.0, resolution=3):
        super(NoFeatureExEncoder, self).__init__()
        self.feature_extractor = nn.LSTM(input_size, hidden_size * resolution, num_layers,
                                         batch_first=True)  # Change this to LSTM with hidden_size*3 for consistency
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(128 * resolution * hidden_size,
                                embedding_dim_size)  # Concatenate outputs from all resolutions
        self.batch_norm = nn.BatchNorm1d(embedding_dim_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        features, _ = self.feature_extractor(x)
        # concatenated_features = torch.cat(features, dim=1)
        concatenated_features = features.reshape(
            (batch_size, -1))  # Concatenate outputs from all resolutions
        z = self.activation(self.linear(concatenated_features))
        z = self.batch_norm(z)
        z = self.dropout(z)
        return z


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim_size, dropout=0.0, resolution=3):
        super(Encoder, self).__init__()
        self.feature_extractor = PhaseShiftedMultiResolutionLSTM(input_size, hidden_size, num_layers)
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(128 * resolution * hidden_size,
                                embedding_dim_size)  # Concatenate outputs from all resolutions
        self.batch_norm = nn.BatchNorm1d(embedding_dim_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        concatenated_features = torch.cat(features, dim=1)
        concatenated_features = concatenated_features.reshape(
            (batch_size, -1))  # Concatenate outputs from all resolutions
        z = self.activation(self.linear(concatenated_features))
        z = self.batch_norm(z)
        z = self.dropout(z)
        return z


class Decoder(nn.Module):
    def __init__(self, embedding_dim_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_dim_size = embedding_dim_size

        # Define linear layers for expanding the latent space
        self.linear1 = nn.Linear(embedding_dim_size, 128 * 4)  # Expand to (batch, 128, 4)

        # Define LSTM layer for further processing
        # LSTM (input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
        #       dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        self.lstm = nn.LSTM(4, output_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(output_size, output_size, num_layers, batch_first=True)

    def forward(self, z):
        # Expand latent space using linear layers
        batch_size = z.shape[0]
        x = F.relu(self.linear1(z))
        x = x.reshape(batch_size, 128, 4)  # Add time step dimension
        """
        LSTM input (batch x seq_length x input_size)
        LSTM1: (batch, 128, 4) ---> (batch, 128, 9)
        LSTM2: (batch, 128, 9) ---> (batch, 128, 9)
        """
        # Decode the expanded features using LSTM
        lstm_out, _ = self.lstm(x)

        # Map LSTM output to desired output size
        decoded_midi, _ = self.lstm2(lstm_out)

        return decoded_midi


class EncoderDecoder(nn.Module):
    def __init__(self, config_path):
        super(EncoderDecoder, self).__init__()
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        encoder_config = config['encoder']
        self.encoder = Encoder(input_size=encoder_config['input_size'],
                                          hidden_size=encoder_config['hidden_size'],
                                          num_layers=encoder_config['num_layers'],
                                          embedding_dim_size=config['training']['embedding_dim_size'],
                                          dropout=encoder_config.get('dropout', 0.0))

        decoder_config = config['decoder']
        self.decoder = Decoder(embedding_dim_size=config['training']['embedding_dim_size'],
                               output_size=decoder_config["output_size"],
                               num_layers=decoder_config["num_layers"])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x)

        # Sample a scaling factor for the noise between 0.001 and 0.01
        if self.training:
            noise_scale = 0.01 + ((0.1 - 0.01) * torch.rand(1, device=self.device))  # 0.01 to 0.1

            # Generate random Gaussian noise with the same shape as z, scaled by the sampled factor
            noise = torch.randn_like(z, device=self.device) * noise_scale
            # Add the noise to z to make the model more robust
            z_noisy = z + noise

        else:
            z_noisy = z

        # Pass the noisy encoded features to the decoder
        decoded_midi = self.decoder(z_noisy)

        # Add reconstruction loss if needed
        return decoded_midi, z_noisy

    def decode_midi(self, z):
        return self.decoder(z)

    def encode(self, pianoroll):
        pianoroll = pianoroll.permute(0, 2, 1)
        return self.encoder(pianoroll)


if __name__ == "__main__":
    train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Len of dataset: {len(train_dataset)}")
    model = EncoderDecoder("Midi_Encoder/config.yaml")
    for midi, _ in train_loader:
        decoded_midi, z = model(midi)

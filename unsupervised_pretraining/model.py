import torch
import yaml
from torch import nn
from transformers import BertTokenizer, BertModel

# Load config from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class MidiEncoder(nn.Module):
    def __init__(self, num_drum_instruments: int, num_timeslices: int, midi_embedding_dim: int, dropout: float):
        super(MidiEncoder, self).__init__()

        self.num_drum_instruments = num_drum_instruments
        self.num_timeslices = num_timeslices

        # Reshape layer
        self.reshape = nn.Linear(num_drum_instruments, midi_embedding_dim)

        # Multi-Head Attention (MHA) layers (Transformers)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=midi_embedding_dim,
            nhead=8,  # number of heads in MHA
            dim_feedforward=512,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        # Fully connected layers to encode into midi_encoding_space
        self.fc_layers = nn.Sequential(
            nn.Linear(midi_embedding_dim * num_timeslices, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, midi_embedding_dim)
        )

    def forward(self, x):
        # Reshape the piano roll
        x = self.reshape(x)
        x = x.permute(1, 0, 2)  # Shape needed for transformer: [seq_len, batch, embedding_dim]

        # Pass through transformer layers
        x = self.transformer_encoder(x)

        # Flatten and pass through FC layers
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='google/bert_uncased_L-4_H-512_A-8'):
        super(TextEncoder, self).__init__()

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained model tokenizer and the model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model).to(self.device)

    def forward(self, texts):
        # Tokenize the input texts
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get the embeddings from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # Use the mean of the last layer hidden states as the sentence vector
        sentence_embeddings = last_hidden_states.mean(dim=1)

        return sentence_embeddings


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        # Linear layer for projection
        self.projection = nn.Linear(embedding_dim, projection_dim)
        # Activation function
        self.gelu = nn.GELU()
        # Fully connected layer
        self.fc = nn.Linear(projection_dim, projection_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Layer normalization
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLAMP(nn.Module):  # Contrastive LAnguage Music Pretraining
    def __init__(self, bert_embedding_dim: int, latent_dimension: int, midi_embedding_dim: int, number_timeslices: int,
                 number_instruments: int, dropout: float, temperature: float):
        super().__init__()
        # Define text and midi encoders
        self.text_encoder = TextEncoder(config['TextEncoder']['dropout'])
        self.midi_encoder = MidiEncoder(number_timeslices, number_instruments, latent_dimension, dropout)
        # Define projection heads for text and midi
        self.text_projection_head = ProjectionHead(bert_embedding_dim, latent_dimension, dropout)
        self.midi_projection_head = ProjectionHead(midi_embedding_dim, latent_dimension, dropout)


if __name__ == "__main__":
    # Initialize CLAMP model using parameters from the config file
    clamp_model = CLAMP(
        bert_embedding_dim=config['CLAMP']['bert_embedding_dim'],
        latent_dimension=config['CLAMP']['latent_dimension'],
        midi_embedding_dim=config['CLAMP']['midi_embedding_dim'],
        number_timeslices=config['CLAMP']['number_timeslices'],
        number_instruments=config['CLAMP']['number_instruments'],
        dropout=config['CLAMP']['dropout'],
        temperature=config['CLAMP']['temperature']
    )
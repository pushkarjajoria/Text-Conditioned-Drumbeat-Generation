import numpy as np
import torch
import yaml
from torch import nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# Load config from YAML
with open('unsupervised_pretraining/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class MidiEncoder(nn.Module):
    def __init__(self, num_drum_instruments: int, num_timeslices: int, midi_embedding_dim: int,
                 dropout: float, num_heads: int, transformer_ff_dim: int):
        super(MidiEncoder, self).__init__()

        self.num_drum_instruments = num_drum_instruments
        self.num_timeslices = num_timeslices

        # Reshape layer
        self.reshape = nn.Linear(num_drum_instruments, midi_embedding_dim)

        # Multi-Head Attention (MHA) layers (Transformers)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=midi_embedding_dim,  # This is the depth/size of the input embeddings (i.e., the number of
            # features the input has). It's the same for the output.
            nhead=num_heads,  # number of heads in MHA
            dim_feedforward=transformer_ff_dim,
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
        x = x.to(self.device).float()
        x = x.float()
        # Reshape the piano roll
        # x = self.reshape(x)
        x = x.permute(2, 0, 1)  # Shape needed for transformer: [seq_len, batch, embedding_dim]

        # Pass through transformer layers
        # This part is working fine.
        # We just need to make sure that the embedding dimention is transformed correctly from the number of instruments.
        x = self.transformer_encoder(x)

        # Flatten and pass through FC layers
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self, dropout: float, pretrained_model='google/bert_uncased_L-4_H-512_A-8'):
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

    def freeze(self):
        """Freeze all parameters of the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters of the model."""
        for param in self.model.parameters():
            param.requires_grad = True


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
        x = x.to(self.device)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected   # Residual Connection
        x = self.layer_norm(x)
        return x


class CLAMP(nn.Module):  # Contrastive LAnguage Music Pretraining
    def __init__(self):
        super(CLAMP, self).__init__()

        # Define text and midi encoders
        self.text_encoder = TextEncoder(dropout=config['TextEncoder']['dropout'])
        self.text_encoder.freeze()  # Freeze the weights of the Bert model
        self.midi_encoder = MidiEncoder(num_drum_instruments=config['MidiEncoder']['number_instruments'],
                                        num_timeslices=config['MidiEncoder']['number_timeslices'],
                                        midi_embedding_dim=config['MidiEncoder']['midi_embedding_dim'],
                                        dropout=config['MidiEncoder']['dropout'],
                                        num_heads=config['MidiEncoder']['num_heads'],
                                        transformer_ff_dim=config['MidiEncoder']['transformer_ff_dim'])

        # Define projection heads for text and midi
        bert_embedding_dim = config['TextEncoder']['bert_embedding_dim']
        latent_dimension = config['ProjectionHead']['projection_dim']
        midi_embedding_dim = config['MidiEncoder']['midi_embedding_dim']
        dropout = config['ProjectionHead']['dropout']
        self.text_projection_head = ProjectionHead(bert_embedding_dim, latent_dimension, dropout=dropout)
        self.midi_projection_head = ProjectionHead(midi_embedding_dim, latent_dimension, dropout=dropout)
        temperature_value = float(config['Training']['temperature'])  # Ensure it's a float
        self.temperature = nn.Parameter(torch.tensor([temperature_value]))

    def forward(self, piano_rolls, texts):
        piano_rolls = piano_rolls.to(self.device)
        texts = texts.to(self.device)
        """
        :param texts: A list of text inputs
        :param piano_rolls: A batch of numpy piano rolls
        :return: Projected embeddings for text and midi
        """

        # Get embeddings from the encoders
        bert_embeddings = self.text_encoder(texts)
        midi_embeddings = self.midi_encoder(piano_rolls)

        # Project the embeddings to the common latent space
        text_projected = self.text_projection_head(bert_embeddings)
        midi_projected = self.midi_projection_head(midi_embeddings)

        return text_projected, midi_projected

    def contrastive_loss(self, text_embeddings, midi_embeddings):
        """
        Computes the CLIP-style contrastive loss.

        Parameters:
        text_embeddings (torch.Tensor): Embeddings for text, shape (batch_size, embedding_dim)
        midi_embeddings (torch.Tensor): Embeddings for MIDI, shape (batch_size, embedding_dim)
        temperature (float): Temperature parameter for scaling the logits

        Returns:
        torch.Tensor: The contrastive loss
        """
        # # Normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, dim=1)
        midi_embeddings = F.normalize(midi_embeddings, dim=1)

        # Compute the similarity matrix (batch_size x batch_size)
        logits = torch.matmul(text_embeddings, midi_embeddings.T) * torch.exp(self.temperature)

        # Labels for the positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate the loss for text-to-MIDI and MIDI-to-text directions
        loss_text_to_midi = F.cross_entropy(logits, labels)
        loss_midi_to_text = F.cross_entropy(logits.T, labels)

        # Average the bidirectional losses
        loss = (loss_text_to_midi + loss_midi_to_text) / 2

        return loss

    def get_text_embeddings(self, texts):
        with torch.no_grad():
            return self.text_projection_head(self.text_encoder(texts))


if __name__ == "__main__":
    def create_array(r, c):
        # Create an array of zeros
        arr = np.zeros((r, c))

        # Determine the number of entries that are 10% of x * y
        num_samples = int(0.1 * r * c)

        # Sample values from the normal distribution
        samples = np.random.normal(loc=0.5, scale=0.1, size=num_samples)

        # Randomly choose indices to place the sampled values
        indices = np.random.choice(r * c, num_samples, replace=False)
        np.put(arr, indices, samples)
        return arr

    # Initialize CLAMP model using parameters from the config file
    clamp_model = CLAMP(
        bert_embedding_dim=config['TextEncoder']['bert_embedding_dim'],
        latent_dimension=config['ProjectionHead']['projection_dim'],
        midi_embedding_dim=config['MidiEncoder']['midi_embedding_dim'],
        number_timeslices=config['CLAMP']['number_timeslices'],
        number_instruments=config['CLAMP']['number_instruments'],
        temperature=config['CLAMP']['temperature']
    )

    text = "Rock 90s Hard Groovy Drums"
    midi = torch.tensor(create_array(config['CLAMP']['number_instruments'], config['CLAMP']['number_timeslices'])).unsqueeze(0)
    clamp_model = clamp_model.float()
    text_emb, midi_emb = clamp_model(text, midi)
    print("Done")

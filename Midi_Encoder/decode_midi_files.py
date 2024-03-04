# Set the path to your checkpoint
import os

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from DDPM.main import load_or_process_dataset
from Midi_Encoder.model import EncoderDecoder
from utils.midi_processing.mid2numpy import save_numpy_as_midi
from utils.utils import save_midi

checkpoint_path = "Midi_Encoder/runs/midi_autoencoder_run/final_model.pt"
device = 'cpu'
# Load dataset and initialize DataLoader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

# Initialize model and move it to the appropriate device
model = EncoderDecoder("Midi_Encoder/config.yaml").to(device)

# Load the model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)

# Directories for saving original and reconstructed MIDI files
original_dir = Path("Midi_Encoder/reconstruction/original")
reconstr_dir = Path("Midi_Encoder/reconstruction/reconstr")

# Create directories if they don't exist
original_dir.mkdir(parents=True, exist_ok=True)
reconstr_dir.mkdir(parents=True, exist_ok=True)

# Process and save MIDI batches
for midi_batch, tags in train_loader:
    reconstructionS, z = model(midi_batch)

    for i, (midi, reconstruction) in enumerate(zip(midi_batch, reconstructionS)):
        midi = (midi * 127).type(torch.uint8)
        reconstruction = (reconstruction * 127).type(torch.uint8)
        # Save the original MIDI file
        original_midi_path = os.path.join(original_dir, f"{tags[i]}.mid")
        midi = midi.numpy()
        save_numpy_as_midi(os.path.join(original_dir, f"{tags[i]}.mid"), midi, ghost_threshold=5)

        # Save the reconstructed MIDI file
        reconstruction = reconstruction.permute((1, 0)).numpy()
        save_numpy_as_midi(os.path.join(reconstr_dir, f"{tags[i]}.mid"), reconstruction, ghost_threshold=5)

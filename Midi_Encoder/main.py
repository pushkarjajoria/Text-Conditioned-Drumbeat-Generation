import os

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm
import numpy as np
# Assuming these imports are from your project structure
from DDPM.main import load_or_process_dataset
from Midi_Encoder.model import EncoderDecoder
from text_supervised_pretraining.main import EarlyStopping


def save_model_checkpoint(epoch, model, run_name):
    # Check if it's the correct epoch interval to save a checkpoint
    checkpoint_dir = os.path.join("Midi_Encoder/runs", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

    # Path to save the current checkpoint
    file_name = "model_checkpoint.pt" if epoch else "final_model.pt"
    checkpoint_path = os.path.join(checkpoint_dir, file_name)

    # Save the model state dict
    torch.save(model.state_dict(), checkpoint_path)

    # Log the model checkpoint to wandb
    wandb.save(checkpoint_path)


# Set manual seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize wandb
run_name = "midi_autoencoder_run"
wandb.init(project="midi_autoencoder", name=run_name)

# Load dataset and initialize DataLoader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model and move it to the appropriate device
model = EncoderDecoder("Midi_Encoder/config.yaml").to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()  # If you're still using MSE for any reason
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
early_stopping = EarlyStopping(patience=10, min_delta=0)

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for midi, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        midi = midi.to(device)  # Move data to the same device as the model

        optimizer.zero_grad()
        decoded_midi, z = model(midi)
        decoded_midi = decoded_midi.permute((0, 2, 1))
        loss = criterion(decoded_midi, midi).to(device)  # Ensure loss calculation is on the correct device
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Learning rate decay
    scheduler.step()

    # Calculate and log average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

    # Early stopping check
    early_stopping(avg_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    # Optional: Save model checkpoint
    if epoch+1 % 10 == 0:  # Corrected the condition to save every 10 epochs
        save_model_checkpoint(epoch, model, run_name)

save_model_checkpoint(None, model, run_name)

wandb.finish()

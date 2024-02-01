import os
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb
import yaml
from tqdm import tqdm
from DDPM.ddpm import Diffusion
from DDPM.model import ConditionalEncDecMHA
from unsupervised_pretraining.create_unsupervised_dataset import get_filenames_and_tags, MidiDataset
from unsupervised_pretraining.model import CLAMP
from unsupervised_pretraining.main import EarlyStopping, save_checkpoint
from utils.utils import get_data, save_midi
import torch.nn as nn


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(config):
    date_time_str = datetime.now().strftime("%m-%d %H:%M")
    run_name = f"Conditional DDPM {date_time_str}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project='BeatBrewer', config=config)

    # Initialize dataset and dataloader
    file_name_and_tags = get_filenames_and_tags(dataset_dir=config['dataset_dir'], filter_common_tags=True)
    train_dataset = MidiDataset(file_name_and_tags)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize models and optimizer
    clamp_model = CLAMP().to(device)
    clamp_model.load_state_dict(torch.load(config['clamp_model_path']))
    clamp_model.eval()
    print("Loaded the pretrained model successfully")

    model = ConditionalEncDecMHA(config['time_embedding_dimension'], clamp_model.latent_dimension, device).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss()
    diffusion = Diffusion()
    early_stopping = EarlyStopping(patience=10)

    # Training loop
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for drum_beats, text_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            text_embeddings = clamp_model.get_text_embeddings(text_data)
            drum_beats = drum_beats.to(device)
            t = diffusion.sample_timesteps(drum_beats.shape[0]).to(device)
            x_t, noise = diffusion.noise_drum_beats(drum_beats, t)
            predicted_noise = model(x_t, t, text_embeddings)

            noise = noise.squeeze()
            predicted_noise = predicted_noise.permute(0, 2, 1)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({'epoch': epoch, 'loss': avg_epoch_loss})
        print(f"Epoch {epoch} Loss: {avg_epoch_loss}")

        # Early stopping and checkpoint saving
        early_stopping(avg_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if (epoch + 1) % 5 == 0:
            print(f"Saving the model and sampling drum beats after {epoch} epoch")
            save_checkpoint(model, run_name, epoch,  wandb, save_type="checkpoint", dir="DDPM")
            # sampled_beats = diffusion.sample(model, n=5).numpy().squeeze()
            # save_midi(sampled_beats, config['results_dir'], epoch)

    # Final model saving and sample generation
    save_checkpoint(model, run_name, None, wandb, save_type="trained_models", dir="DDPM")


if __name__ == "__main__":
    config_path = 'DDPM/config.yaml'
    config = load_config(config_path)
    train(config)


import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb
import yaml
from tqdm import tqdm
from DDPM.latent_diffusion import LatentDiffusion
from DDPM.model import ConditionalLatentEncDecMHA, ConditionalUNet
from Midi_Encoder.model import EncoderDecoder
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


# Function to check for a pickled dataset and load it
def load_or_process_dataset(dataset_dir):
    pickle_file_path = os.path.join(dataset_dir, 'processed_midi_dataset.pkl')

    # Check if the pickled dataset exists
    if os.path.exists(pickle_file_path):
        print("Loading dataset from pickle file.")
        with open(pickle_file_path, 'rb') as f:
            midi_dataset = pickle.load(f)  # Load the entire MidiDataset object
    else:
        print("Processing dataset from scratch.")
        file_name_and_tags = get_filenames_and_tags(dataset_dir=dataset_dir, filter_common_tags=True)
        midi_dataset = MidiDataset(file_name_and_tags)  # Initialize the MidiDataset with file_name_and_tags

        # Saving the processed MidiDataset to a pickle file for future use
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(midi_dataset, f)  # Pickle the entire MidiDataset object
            print(f"Dataset saved to {pickle_file_path}")

    return midi_dataset


def train(config):
    torch.manual_seed(42)
    np.random.seed(42)

    date_time_str = datetime.now().strftime("%m-%d %H:%M")
    run_name = f"Latent conditional DDPM {date_time_str}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project='BeatBrewer', config=config)

    # Initialize dataset and dataloader
    train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Len of dataset: {len(train_dataset)}")

    # Initialize models and optimizer
    clamp_model = CLAMP().to(device)
    if torch.cuda.is_available():
        clamp_model.load_state_dict(torch.load(config['clamp_model_path'], map_location=torch.device('cuda')))
    else:
        clamp_model.load_state_dict(torch.load(config['clamp_model_path'], map_location=torch.device('cpu')))
    clamp_model.eval()
    print("Loaded the pretrained model successfully")

    model = ConditionalUNet(config['z_dimension'], clamp_model.latent_dimension, config['time_embedding_dimension']).to(device)
    model_state_path = "/ichec/home/users/pushkarj/Git/BeatBrewer/DDPM/checkpoint/Latent conditional DDPM 03-04 14:53/model_epoch_29.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_state_path))
    else:
        model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    print("Loaded the ConditionalUNet model successfully")

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss().to(device)
    diffusion = LatentDiffusion()
    early_stopping = EarlyStopping(patience=10)

    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = "Midi_Encoder/runs/midi_autoencoder_run/final_model.pt"
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))

    print("Loaded encoder decoder model successfully.")

    # Training loop
    for epoch in range(30, config['epochs']):
        epoch_loss = 0
        for drum_beats, text_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            text_embeddings = clamp_model.get_text_embeddings(text_data)
            drum_beats = drum_beats.to(device)
            drum_beat_latent_code = midi_encoder_decoder.encoder(drum_beats.permute(0, 2, 1))
            t = diffusion.sample_timesteps(drum_beat_latent_code.shape[0]).to(device)
            z_t, noise = diffusion.noise_z(drum_beat_latent_code, t)
            predicted_noise = model(z_t, t, text_embeddings)

            noise = noise.squeeze()
            predicted_noise = predicted_noise
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


def generate(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = LatentDiffusion()

    # Initialize models and optimizer
    clamp_model = CLAMP().to(device)
    clamp_model.load_state_dict(torch.load(config['clamp_model_path']))
    clamp_model.eval()
    print("Loaded the pretrained CLAMP model successfully")

    model = ConditionalUNet(config['z_dimension'], clamp_model.latent_dimension, config['time_embedding_dimension']).to(device)
    model_state_path = "DDPM/trained_models/Latent conditional DDPM 03-04 20:00/model_final.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_state_path))
    else:
        model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    model.eval()
    text = ["Rock Essentials 8th Rock Rock Basic", "Country song pop chorus"]
    # file_name_and_tags = get_filenames_and_tags(dataset_dir=config['dataset_dir'], filter_common_tags=True)
    # text_from_dataset = random.choices(list(file_name_and_tags.values()), k=5)
    text_prompts = text
    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = "Midi_Encoder/runs/midi_autoencoder_run/final_model.pt"
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))

    print("Loaded encoder decoder model successfully.")

    text_embeddings = clamp_model.get_text_embeddings(text_prompts)
    sampled_beats = diffusion.sample_conditional(model, n=len(text_prompts),
                                                 text_embeddings=text_embeddings, midi_decoder=midi_encoder_decoder).numpy().squeeze()
    file_names = text_prompts
    sampled_beats = sampled_beats.transpose((0, 2, 1))
    save_midi(sampled_beats, config['results_dir'], file_names=file_names)
    print("Done")


def reconstruct_dataset_midi(config):
    # Initialize dataset and dataloader
    train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    counter = 0
    for drum_beats, text_data in tqdm(train_loader):
        rand_number = random.random()
        if rand_number <= 0.1:
            counter += 1
            drum_beats = (drum_beats * 127).type(torch.uint8)
            save_midi(drum_beats, config['reconstruct_dir'], file_names=text_data)
        if counter >= 10:
            break


if __name__ == "__main__":
    config_path = 'DDPM/config.yaml'
    config = load_config(config_path)
    # train(config)
    generate(config)
    # reconstruct_dataset_midi(config)






import os
import pickle
import random
from collections import defaultdict
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
    run_name = f"No Mutli-LSTM DDPM {date_time_str}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project='BeatBrewer', config=config)

    # Initialize dataset and dataloader
    train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Len of dataset: {len(train_dataset)}")

    model = ConditionalUNet(time_encoding_dim=16).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss().to(device)
    diffusion = LatentDiffusion()
    early_stopping = EarlyStopping(patience=10)

    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = "/Midi_Encoder/runs/midi_autoencoder_run_lstm_test/final_model.pt"
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))
    midi_encoder_decoder.eval()

    print("Loaded encoder decoder model successfully.")

    # Training loop
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for drum_beats, text_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            drum_beats = drum_beats.to(device)
            drum_beat_latent_code = midi_encoder_decoder.encoder(drum_beats.permute(0, 2, 1))
            # normalized_drum_beat_latent_code = torch.tanh(drum_beat_latent_code)
            t = diffusion.sample_timesteps(drum_beat_latent_code.shape[0]).to(device)
            z_t, noise = diffusion.noise_z(drum_beat_latent_code, t)
            predicted_noise = model(z_t, t, text_data)

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
    diffusion = LatentDiffusion(latent_dimension=128)

    model = ConditionalUNet(time_encoding_dim=16).to(device)
    model_state_path = "AIMC results/High Noise/ddpm_model/model_final.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_state_path))
    else:
        model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    model.eval()
    text = [" ".join(['rock', '4-4', 'electronic', 'fill', 'ride', 'funk', 'fills', '8ths', '8-bar', 'shuffle', 'jazz',
                'half-time', 'blues', 'chorus', 'crash', 'verse', '8th', 'fusion', 'country', 'intro', 'shuffles',
                '16ths', 'metal', 'swing', 'quarter', 'hard', 'retro', 'bridge', 'tom', 'punk', 'trance', 'hats',
                'latin', 'kick', 'techno', 'slow', 'progressive', 'bongo', 'house', 'african', 'samba', 'intros',
                'triplet', 'bell', 'urban', 'ballad', 'snare', 'funky', 'fast', 'rides', 'hip', 'hop', 'toms', 'four',
                'downbeat', 'cowbell', 'pop']), "My Name is Pushkar"]
    text_prompts = text
    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = "AIMC results/High Noise/enc_dec_model/final_model.pt"
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))

    print("Loaded encoder decoder model successfully.")

    sampled_beats = diffusion.sample_conditional(model, n=len(text_prompts),
                                                 text_keywords=text, midi_decoder=midi_encoder_decoder).numpy().squeeze()
    file_names = list(map(lambda x: x[:25], text_prompts))
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


def get_keywords_map(config):
    train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    map_of_keywords = defaultdict(int)
    for _, text_data in tqdm(train_loader):
        curr_map = set()
        for keyw in text_data[0].split(" "):
            keyw = keyw.lower()
            if keyw not in curr_map:
                curr_map.add(keyw)
                map_of_keywords[keyw] += 1

    sorted_keywords = sorted(map_of_keywords.items(), key=lambda x: x[1], reverse=True)
    total_occurrences = sum(freq for _, freq in sorted_keywords)
    cumulative = 0
    threshold = total_occurrences * 0.95
    top_keywords = []

    for keyword, freq in sorted_keywords:
        cumulative += freq
        top_keywords.append(keyword)
        if cumulative >= threshold:
            break

    # Ask the user to include keywords or not
    chosen_keywords = []
    for keyword in top_keywords:
        response = input(f"Do you want to include '{keyword}'? (y/n): ").lower()
        if response == 'y':
            chosen_keywords.append(keyword)

    print("Chosen keywords:", chosen_keywords)


if __name__ == "__main__":
    config_path = 'DDPM/config.yaml'
    config = load_config(config_path)
    # train(config)
    generate(config)
    # reconstruct_dataset_midi(config)
    # get_keywords_map(config)
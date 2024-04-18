import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from DDPM.latent_diffusion import LatentDiffusion
from DDPM.main_latent_space import load_or_process_dataset, load_config
from torch.utils.data import DataLoader
from DDPM.model import ConditionalUNet, ConditionalUNetBERT
from Midi_Encoder.model import EncoderDecoder
from text_supervised_pretraining.model import CLAMP
from utils.utils import save_midi, save_midi_without_structure


def get_models(ddpm_model_path, ae_model_path, model_type="multihot"):
    if model_type == "multihot":
        ddpm_model = ConditionalUNet(time_encoding_dim=16).to(device)
    else:
        ddpm_model = ConditionalUNetBERT(time_encoding_dim=16).to(device)
    model_state_path = ddpm_model_path
    if torch.cuda.is_available():
        ddpm_model.load_state_dict(torch.load(model_state_path))
    else:
        ddpm_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    ddpm_model.eval()
    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = ae_model_path
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))
    midi_encoder_decoder.eval()
    return ddpm_model, midi_encoder_decoder


device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = 'DDPM/config.yaml'
config = load_config(config_path)

no_noise_ddpm, no_noise_ae = get_models("AIMC results/No Noise/ddpm_model/model_final.pth",
                                        "AIMC results/No Noise/enc_dec_model/final_model.pt")

bert_ddpm, bert_ae = get_models("AIMC results/Bert Encoding/ddpm_model/model_final.pth",
                                "AIMC results/No Noise/enc_dec_model/final_model.pt", "bert")

clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load(config['clamp_model_path'], map_location=device))
clamp_model.eval()

train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
# subset_size = 100  # For example, to use only 100 samples from your dataset
# train_dataset = Subset(train_dataset, list(range(subset_size)))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

diffusion = LatentDiffusion(latent_dimension=128)
file_mappings = []  # List to hold file names and their mappings for CSV
all_midi_data = []  # List to hold all midi data before saving
all_file_names = []  # List to hold all file names corresponding to the midi data
for drum_beats, text_tags in tqdm(train_loader):
    original_drumbeat = (drum_beats.detach().clamp(0, 1) * 127).type(torch.uint8).permute((0, 2, 1)).numpy()

    no_noise_drumbeats = diffusion.sample_conditional(no_noise_ddpm, n=len(text_tags),
                                                      text_keywords=text_tags,
                                                      midi_decoder=no_noise_ae).numpy().squeeze()

    bert_drumbeats = diffusion.sample_conditional_bert(bert_ddpm, n=len(text_tags),
                                                       text_keywords=text_tags, midi_decoder=bert_ae,
                                                       text_encoder=clamp_model).numpy().squeeze()

    tuple_of_empty_tags = tuple([" " for _ in range(10)])
    negative_noise_drumbeats = diffusion.sample_conditional(no_noise_ddpm, n=len(tuple_of_empty_tags),
                                                            text_keywords=tuple_of_empty_tags,
                                                            midi_decoder=no_noise_ae).numpy().squeeze()

    # Assuming 'results_dir' is in your config and points to where you want to save the files
    results_dir = config['survey_results_dir']

    for idx, tag in enumerate(text_tags):
        numbers = [1, 2, 3, 4]
        random.shuffle(numbers)
        categories = ['dataset', 'multihot', 'bert', 'negative']
        beats_arrays = [original_drumbeat, no_noise_drumbeats, bert_drumbeats, negative_noise_drumbeats]

        for number, category, beat_array in zip(numbers, categories, beats_arrays):
            file_name = f"{number}_{tag}.mid"
            # Append midi data and file name to their respective lists
            all_midi_data.append(beat_array[idx])
            all_file_names.append(file_name)
            file_mappings.append(
                {'file_name': file_name, 'text_tag': tag if category != 'negative' else 'N/A', 'category': category})
    break

# After accumulating all MIDI data and file names, adjust their shape as required and save
all_midi_data_array = np.stack(all_midi_data).transpose((0, 2, 1))  # Adjust shape as required by your save function
save_midi_without_structure(all_midi_data_array, config['survey_results_dir'], file_names=all_file_names)

# # Save the mapping to a CSV file
# df_mappings = pd.DataFrame(file_mappings)
# df_mappings.to_csv(f"{config['survey_results_dir']}/file_mappings.csv", index=False)

# Assuming file_mappings is already defined and populated
df_mappings = pd.DataFrame(file_mappings)

# Specify the file path
file_path = f"{config['survey_results_dir']}/file_mappings.csv"

# Check if the file exists to determine whether to write header
file_exists = os.path.isfile(file_path)

# Append to the file (if it exists), else create a new one with headers
df_mappings.to_csv(file_path, mode='a', index=False, header=not file_exists)
exit("Done!")

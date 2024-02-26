import os
import random
from collections import defaultdict

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from DDPM.ddpm import Diffusion
from DDPM.main import load_or_process_dataset, load_config
from DDPM.model import ConditionalEncDecMHA
from unsupervised_pretraining.model import CLAMP

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize dataset and dataloader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print(f"Len of dataset: {len(train_dataset)}")

grouped_tags = defaultdict(list)
for midi_tensor, file_tag in train_loader:
    # Split the tag by space and filter out the numeric part
    # Split the tag by space and filter out the numeric part
    parts = file_tag[0].split()
    numeric_part = ''.join(filter(str.isdigit, parts[-1]))
    if numeric_part:
        prefix = file_tag[0].rstrip(numeric_part)
        grouped_tags[prefix].append((midi_tensor, file_tag[0]))
grouped_tags = {key: value for key, value in grouped_tags.items() if len(value) > 1}
similar_midis = list(grouped_tags.values())

# Initialize models and optimizer
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load("unsupervised_pretraining/trained_models/0210_2142/model_final.pth"))
clamp_model.eval()
print("Loaded the pretrained model successfully")

config_path = 'DDPM/config.yaml'
config = load_config(config_path)
ddpm_model = ConditionalEncDecMHA(config['time_embedding_dimension'], clamp_model.latent_dimension, device).to(device)
ddpm_model.load_state_dict(torch.load(config['ddpm_model_path']))
ddpm_model.eval()
diffusion = Diffusion(num_time_slices=128)

# Initialize arrays to store distances
distances_within_dataset = []
distances_generated = []
distances_similar = []
distances_similar_generated = []

num_iterations = 210

for _ in tqdm(range(num_iterations)):
    # Randomly sample two MIDI files from the dataset
    index1, index2 = torch.randint(0, len(train_dataset), (2,))
    drum_beats1, text_data1 = train_dataset[index1]
    drum_beats2, text_data2 = train_dataset[index2]

    # Get MIDI embeddings for the sampled data points
    midi_embeddings1 = clamp_model.get_midi_embeddings(drum_beats1.unsqueeze(0).to(device))
    midi_embeddings2 = clamp_model.get_midi_embeddings(drum_beats2.unsqueeze(0).to(device))

    # Compute distance between the embeddings
    distance_within_dataset = torch.nn.functional.pairwise_distance(midi_embeddings1, midi_embeddings2).item()
    distances_within_dataset.append(distance_within_dataset)

    # Generate MIDI files using DDPM model
    generated_drum_beats1, generated_drum_beats2 = diffusion._sample_conditional(ddpm_model,
                                                                                 torch.stack((clamp_model.get_text_embeddings(text_data1), clamp_model.get_text_embeddings(text_data2))).squeeze())

    # Get MIDI embeddings for the generated data points
    generated_midi_embeddings1 = clamp_model.get_midi_embeddings(generated_drum_beats1.to(device))
    generated_midi_embeddings2 = clamp_model.get_midi_embeddings(generated_drum_beats2.to(device))

    # Compute distance between the embeddings
    distance_generated = torch.nn.functional.pairwise_distance(generated_midi_embeddings1, generated_midi_embeddings2).item()
    distances_generated.append(distance_generated)

    # Sample similar MIDI files
    similar_index1, similar_index2 = torch.randint(0, len(similar_midis), (2,))
    num_similar_variants1, num_similar_variants2 = len(similar_midis[similar_index1]), len(similar_midis[similar_index2])
    # Get similar MIDI embeddings
    drum_beats_similar1, text_data_similar1 = similar_midis[similar_index1][random.randint(0, num_similar_variants1-1)]
    drum_beats_similar2, text_data_similar2 = similar_midis[similar_index2][random.randint(0, num_similar_variants2-1)]

    # Get MIDI embeddings for the similar data points
    midi_embeddings_similar1 = clamp_model.get_midi_embeddings(drum_beats_similar1.to(device))
    midi_embeddings_similar2 = clamp_model.get_midi_embeddings(drum_beats_similar2.to(device))

    # Compute distance between the embeddings
    distance_similar = torch.nn.functional.pairwise_distance(midi_embeddings_similar1, midi_embeddings_similar2).item()
    distances_similar.append(distance_similar)

    # Generate MIDI files using DDPM model for similar MIDI files
    generated_drum_beats_similar1, generated_drum_beats_similar2 = diffusion._sample_conditional(ddpm_model,
                                                                                 torch.stack((clamp_model.get_text_embeddings(text_data_similar1), clamp_model.get_text_embeddings(text_data_similar2))).squeeze())

    # Get MIDI embeddings for the generated similar data points
    generated_midi_embeddings_similar1 = clamp_model.get_midi_embeddings(generated_drum_beats_similar1.to(device))
    generated_midi_embeddings_similar2 = clamp_model.get_midi_embeddings(generated_drum_beats_similar2.to(device))

    # Compute distance between the embeddings for similar generated MIDI files
    distance_similar_generated = torch.nn.functional.pairwise_distance(generated_midi_embeddings_similar1, generated_midi_embeddings_similar2).item()
    distances_similar_generated.append(distance_similar_generated)

# Convert lists to numpy arrays
distances_within_dataset = np.array(distances_within_dataset)
distances_generated = np.array(distances_generated)
distances_similar = np.array(distances_similar)
distances_similar_generated = np.array(distances_similar_generated)

# Plot distributions
plt.figure(figsize=(8, 6))

sns.kdeplot(distances_within_dataset, label='Within Dataset')
sns.kdeplot(distances_generated, label='Generated MIDI Files')
sns.kdeplot(distances_similar, label='Similar MIDI Files')
sns.kdeplot(distances_similar_generated, label='Similar Generated MIDI Files')

plt.xlabel('Distance')
plt.ylabel('Density')
plt.title('Distribution of Distances')
plt.legend()
print(os.curdir)
plt.savefig('./distances_distribution_plot.jpg')
plt.show()
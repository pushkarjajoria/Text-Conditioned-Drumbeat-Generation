# Initialize dataset and dataloader
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from DDPM.ddpm import Diffusion
from DDPM.main import load_or_process_dataset
from DDPM.model import ConditionalEncDecMHA
from text_supervised_pretraining.model import CLAMP
from utils.utils import save_midi


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config_path = 'DDPM/config.yaml'
config = load_config(config_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
diffusion = Diffusion(num_time_slices=128)

# Initialize models and optimizer
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load(config['clamp_model_path']))
clamp_model.eval()
print("Loaded the pretrained CLAMP model successfully")

model = ConditionalEncDecMHA(config['time_embedding_dimension'], clamp_model.latent_dimension, device).to(device)
model.load_state_dict(torch.load(config['ddpm_model_path']))
model.eval()

# Initialize dataset and dataloader
train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
train_dataset = Subset(train_dataset, indices=range(128))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print(f"Len of dataset: {len(train_dataset)}")


def calculate_note_density(beat):
    """Calculate the note density for each beat in a batch.

    Args:
    - beat: A numpy array of shape [batch_size, num_instruments, num_timesteps],
            where each element is the normalized velocity of a note.

    Returns:
    - A numpy array of note densities, each calculated as the fraction of timesteps
      with at least one non-zero velocity (note hit) across all instruments for each beat.
    """
    # Check for any note hits across instruments (axis=1) for each timestep in each beat
    note_hits = np.any(beat > (5/127), axis=1)  # This reduces the shape to [batch_size, num_timesteps]

    # Calculate density by summing note hits across timesteps (axis=1) and dividing by the number of timesteps
    note_densities = np.sum(note_hits, axis=1) / beat.shape[2]  # Normalize by number of timesteps

    return note_densities


original_note_densities = []
generated_note_densities = []

for drum_beats, text_data in tqdm(train_loader, desc="Computing note densities"):
    text_prompts = text_data
    text_embeddings = clamp_model.get_text_embeddings(text_prompts)

    # Generate sampled beats assuming it returns a tensor with shape [batch_size, num_instruments, num_timesteps]
    sampled_beats = diffusion.sample_conditional(model, n=len(text_prompts),
                                                 text_embeddings=text_embeddings).cpu().numpy()

    # Calculate note densities for the batch
    original_densities = calculate_note_density(drum_beats.numpy())
    generated_densities = calculate_note_density(sampled_beats/127)

    # Extend lists with batch results
    original_note_densities.extend(original_densities)
    generated_note_densities.extend(generated_densities)

import seaborn as sns
import matplotlib.pyplot as plt

# Convert lists to numpy arrays for easier handling
original_note_densities_np = np.array(original_note_densities)
generated_note_densities_np = np.array(generated_note_densities)

# Plot PDFs using KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(original_note_densities_np, label='Original', fill=True)
sns.kdeplot(generated_note_densities_np, label='Generated', fill=True)
plt.xlabel('Note Density')
plt.ylabel('Probability Density')
plt.title('PDF of Note Densities')
plt.legend()
plt.show()

# Plot Box Plots for direct comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=[original_note_densities_np, generated_note_densities_np],
            notched=True, # With notches to show the confidence interval of the median
            palette="Set2")
plt.xticks(ticks=[0, 1], labels=['Original', 'Generated'])
plt.ylabel('Note Density')
plt.title('Box Plot of Note Densities')
plt.show()
# Plot the two note density in a single plot as a distribution

print("Original Note Densities Statistics:")
print(f"Mean: {np.mean(original_note_densities_np):.2f}")
print(f"Median: {np.median(original_note_densities_np):.2f}")
print(f"Standard Deviation: {np.std(original_note_densities_np):.2f}\n")
print("*"*16)
print("Generated Note Densities Statistics:")
print(f"Mean: {np.mean(generated_note_densities_np):.2f}")
print(f"Median: {np.median(generated_note_densities_np):.2f}")
print(f"Standard Deviation: {np.std(generated_note_densities_np):.2f}")
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from DDPM.main import load_or_process_dataset, load_config
from unsupervised_pretraining.model import CLAMP
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize dataset and dataloader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print(f"Len of dataset: {len(train_dataset)}")

# Initialize models and optimizer
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load("unsupervised_pretraining/trained_models/0210_2142/model_final.pth"))
clamp_model.eval()
print("Loaded the pretrained model successfully")

config_path = 'DDPM/config.yaml'
config = load_config(config_path)

# Initialize arrays to store distances
same_pair_distances = []
m_i_t_j_distances = []
mi_mj_distances = []
t_i_t_j_distances = []

num_iterations = 1000

for _ in tqdm(range(num_iterations)):
    # Randomly sample two MIDI files from the dataset
    index1, index2 = torch.randint(0, len(train_dataset), (2,))
    d_i, t_i = train_dataset[index1]
    d_j, t_j = train_dataset[index2]

    # Get MIDI embeddings for the sampled data points
    m_i_embeddings = clamp_model.get_midi_embeddings(d_i.unsqueeze(0).to(device))
    m_j_embeddings = clamp_model.get_midi_embeddings(d_j.unsqueeze(0).to(device))
    t_i_embeddings = clamp_model.get_text_embeddings(t_i)
    t_j_embeddings = clamp_model.get_text_embeddings(t_j)

    mi_ti_distance = torch.nn.functional.pairwise_distance(m_i_embeddings, t_i_embeddings)
    mj_tj_distance = torch.nn.functional.pairwise_distance(m_j_embeddings, t_j_embeddings)
    same_pair_distances.append(mi_ti_distance.item())
    same_pair_distances.append(mj_tj_distance.item())

    cross_distance1 = torch.nn.functional.pairwise_distance(m_i_embeddings, t_j_embeddings)
    cross_distance2 = torch.nn.functional.pairwise_distance(m_j_embeddings, t_i_embeddings)
    m_i_t_j_distances.append(cross_distance1.item())
    m_i_t_j_distances.append(cross_distance2.item())

    midi_distance = torch.nn.functional.pairwise_distance(m_i_embeddings, m_j_embeddings)
    mi_mj_distances.append(midi_distance.item())

    text_distance = torch.nn.functional.pairwise_distance(t_i_embeddings, t_j_embeddings)
    t_i_t_j_distances.append(text_distance.item())


# Plot KDE plots
plt.figure(figsize=(12, 8))
sns.kdeplot(same_pair_distances, label="Same Pair Distances (Mi, Ti)", linewidth=2)
sns.kdeplot(m_i_t_j_distances, label="MIDI-Text Distances (Mx, Ty)", linewidth=2)
sns.kdeplot(mi_mj_distances, label="MIDI Distances (Mi, Mj)", linewidth=2)
sns.kdeplot(t_i_t_j_distances, label="Text Distances (Ti, Tj)", linewidth=2)
plt.xlabel('Distance')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Distances')
plt.legend()
plt.savefig("/Users/pushkarjajoria/Git/BeatBrewer/distance_within_dataset.jpg")
plt.show()

# Print relevant statistics
print("Statistics: \n")
print("Same Pair Distances (Mi, Ti):")
print(f"  Mean: {np.mean(same_pair_distances)}")
print(f"  Median: {np.median(same_pair_distances)}")
print(f"  Standard Deviation: {np.std(same_pair_distances)}")
print(f"  Minimum: {np.min(same_pair_distances)}")
print(f"  Maximum: {np.max(same_pair_distances)}")
print("\nMIDI-Text Distances (Mx, Ty):")
print(f"  Mean: {np.mean(m_i_t_j_distances)}")
print(f"  Median: {np.median(m_i_t_j_distances)}")
print(f"  Standard Deviation: {np.std(m_i_t_j_distances)}")
print(f"  Minimum: {np.min(m_i_t_j_distances)}")
print(f"  Maximum: {np.max(m_i_t_j_distances)}")
print("\nMIDI Distances (Mi, Mj):")
print(f"  Mean: {np.mean(mi_mj_distances)}")
print(f"  Median: {np.median(mi_mj_distances)}")
print(f"  Standard Deviation: {np.std(mi_mj_distances)}")
print(f"  Minimum: {np.min(mi_mj_distances)}")
print(f"  Maximum: {np.max(mi_mj_distances)}")
print("\nText Distances (Ti, Tj):")
print(f"  Mean: {np.mean(t_i_t_j_distances)}")
print(f"  Median: {np.median(t_i_t_j_distances)}")
print(f"  Standard Deviation: {np.std(t_i_t_j_distances)}")
print(f"  Minimum: {np.min(t_i_t_j_distances)}")
print(f"  Maximum: {np.max(t_i_t_j_distances)}")

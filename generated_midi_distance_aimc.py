import pickle
from itertools import combinations

import numpy as np
import torch
from scipy.spatial.distance import pdist, cdist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from DDPM.latent_diffusion import LatentDiffusion
from DDPM.main_latent_space import load_or_process_dataset, load_config
from DDPM.model import ConditionalUNet
from Midi_Encoder.model import EncoderDecoder


def get_models(ddpm_model_path, ae_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm_model = ConditionalUNet(time_encoding_dim=16).to(device)
    model_state_path = ddpm_model_path
    if torch.cuda.is_available():
        ddpm_model.load_state_dict(torch.load(model_state_path))
    else:
        ddpm_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    ddpm_model.eval()
    autoencoder_config_path = "Midi_Encoder/config.yaml"
    autoencoder_model_path = ae_model_path
    midi_encoder_decoder = EncoderDecoder(autoencoder_config_path).to(device)
    midi_encoder_decoder.eval()
    if torch.cuda.is_available():
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path))
    else:
        midi_encoder_decoder.load_state_dict(torch.load(autoencoder_model_path, map_location=torch.device('cpu')))

    return ddpm_model, midi_encoder_decoder


def compute_stats(distances):
    return np.mean(distances), np.std(distances), np.min(distances), np.max(distances)


def compute_midi_distance(batch_of_midi_files):
    distance_between_midis = []
    # Flatten the last two dimensions (128x9) of each MIDI file for distance calculation
    flattened_midis = batch_of_midi_files.reshape(batch_of_midi_files.shape[0], -1)
    # Iterate over all unique pairs of MIDI files
    for i, j in combinations(range(len(flattened_midis)), 2):
        # Compute the mean Euclidean distance between the flattened MIDI representations
        distance = np.linalg.norm(flattened_midis[i] - flattened_midis[j])/flattened_midis.shape[-1]
        # Add the computed distance to the list
        distance_between_midis.append(distance)
    return distance_between_midis


def compute_midi_distance_between(midi_pianorolls_one, midi_pianorolls_two):
    all_distances = []
    # Flatten the last two dimensions (128x9) of each MIDI file for distance calculation
    flattened_one = midi_pianorolls_one.reshape(midi_pianorolls_one.shape[0], -1)
    flattened_two = midi_pianorolls_two.reshape(midi_pianorolls_two.shape[0], -1)

    # Compute pairwise distances between every midi in set one and every midi in set two
    for midi_one in flattened_one:
        for midi_two in flattened_two:
            distance = np.linalg.norm(midi_one - midi_two)/midi_one.shape[-1]
            all_distances.append(distance)

    # Sort distances to find the 1% smallest ones
    all_distances.sort()
    one_percent_index = int(len(all_distances) * 0.01)
    one_percent_least_distances = all_distances[:max(1, one_percent_index)]  # Ensure at least one distance is included

    return one_percent_least_distances


def compute_1_percent_least_distances(generated_midi_embeddings, dataset_midi_embeddings):
    # Compute the pairwise Euclidean distances between generated MIDIs and dataset MIDIs
    distances = cdist(generated_midi_embeddings, dataset_midi_embeddings, 'euclidean')

    # Flatten the distance matrix to sort all distances together
    flattened_distances = distances.flatten()

    # Sort the flattened array of distances
    sorted_distances = np.sort(flattened_distances)

    # Select the 1% smallest distances
    one_percent_size = int(len(sorted_distances) * 0.01)
    one_percent_least_distances = sorted_distances[
                                  :max(one_percent_size, 1)]  # Ensure at least one distance is included

    # The final list should be of length 1134 if the total number of distances is 113400
    return one_percent_least_distances


diffusion = LatentDiffusion(latent_dimension=128)

prompts = ['latin triplet', '4-4 electronic', 'funky 16th', 'rock fill 8th',
           'blues shuffle', 'pop ride', 'funky blues', 'latin rock']

low_noise_ddpm, low_noise_ae = get_models("AIMC results/Base Model Results/ddpm_model/model_final.pth",
                                          "AIMC results/Base Model Results/enc_dec_model/final_model.pt")

high_noise_ddpm, high_noise_ae = get_models("AIMC results/High Noise/ddpm_model/model_final.pth",
                                            "AIMC results/High Noise/enc_dec_model/final_model.pt")

no_noise_ddpm, no_noise_ae = get_models("AIMC results/No Noise/ddpm_model/model_final.pth",
                                        "AIMC results/No Noise/enc_dec_model/final_model.pt")

models = [(low_noise_ddpm, low_noise_ae), (high_noise_ddpm, high_noise_ae), (no_noise_ddpm, no_noise_ae)]
config_path = 'DDPM/config.yaml'
config = load_config(config_path)
train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
subset_size = 100  # For example, to use only 100 samples from your dataset
train_dataset = Subset(train_dataset, list(range(subset_size)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

total_stats = {}
dataset_midi_space = []
for drum_beats, _ in tqdm(train_loader):
    dataset_midi_space.extend(drum_beats.numpy())

dataset_midi_space = (np.array(dataset_midi_space) * 255).astype(int)

for model_name, (ddpm_model, enc_dec_model) in zip(["Low Noise", "High Noise", "No Noise"], models):
    model_stats = {
        "midi": [],
        "latent": [],
        "dataset_midi": [],
        "dataset_latent": []
    }
    # Compute distances from the dataset in both spaces
    # This would involve comparing each of the 10 generated samples against the entire dataset
    # Assuming there's a method to get the entire dataset's embeddings in latent space for this comparison
    dataset_latent_embeddings = []

    # For MIDI space, assuming a function that converts dataset MIDI to the same space as sampled_midi
    for drum_beats, _ in tqdm(train_loader):
        dataset_batch_z = enc_dec_model.encode(drum_beats)
        dataset_latent_embeddings.extend(dataset_batch_z.detach().numpy())
    dataset_latent_embeddings = np.array(dataset_latent_embeddings)

    for prompt in prompts:
        prompt_repeated = [prompt] * 10  # Copy the same prompt 10 times in a list
        # Generate 10 MIDI files
        # Assume diffusion.sample_conditional returns a batch of generated MIDI files based on the prompt
        sampled_midi = diffusion.sample_conditional(ddpm_model, n=10, text_keywords=prompt_repeated, midi_decoder=enc_dec_model)

        # Compute distances in the MIDI space
        # Assuming sampled_midi is in the correct format to compute distances directly
        distances_midi = compute_midi_distance(sampled_midi)
        model_stats["midi"].append(distances_midi)

        # Embed the MIDI to get 10 Z vectors in the latent space
        z = enc_dec_model.encode(sampled_midi.permute(0, 2, 1)/255).detach().numpy()
        distances_latent = pdist(z, 'euclidean')
        model_stats["latent"].append(distances_latent)

        # Compute the distance from all the dataset and choose the top 1% of datapoints for analysis
        distances_from_dataset_midi = compute_midi_distance_between(sampled_midi, dataset_midi_space)

        distances_from_dataset_latent = compute_1_percent_least_distances(z, dataset_latent_embeddings)

        # Assuming methods to compute stats for these selected top 1% distances
        model_stats["dataset_midi"].append(distances_from_dataset_midi)
        model_stats["dataset_latent"].append(distances_from_dataset_latent)

    # Compute and publish stats for the model
    with open("AIMC results/distance_results.txt", "a") as file:
        for space in ["midi", "latent", "dataset_midi", "dataset_latent"]:
            distances = np.concatenate(model_stats[space])
            mean, std, min_dis, max_dis = compute_stats(distances)
            output_text = f"{model_name} {space.capitalize()} Space - Mean distance: {mean:.2f}, Std: {std:.2f}, Min: {min_dis:.2f}, Max: {max_dis:.2f}\n"
            # Print to console
            print(output_text.strip())

            # Write to file
            file.write(output_text)
    # Add model stats to total_stats for use later
    total_stats[model_name] = model_stats

# Save the total_stats in a file
with open("AIMC results/distance_stats.pickle", "wb") as f:
    pickle.dump(total_stats, f)
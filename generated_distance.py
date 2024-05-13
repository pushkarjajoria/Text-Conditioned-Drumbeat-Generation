import copy
import random
from collections import defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from DDPM.latent_diffusion import LatentDiffusion
from DDPM.main_latent_space import load_or_process_dataset, load_config
from torch.utils.data import DataLoader
from DDPM.model import ConditionalUNet, ConditionalUNetBERT, MultiHotEncoderWithBPM
from Midi_Encoder.model import EncoderDecoder


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


def mh_vector_to_string(mh_vector):
    output_string = []
    for s in mh_vector[:-1]:
        output_string.append(str(s))
    return "".join(output_string)


device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = 'DDPM/config.yaml'
config = load_config(config_path)

no_noise_ddpm, no_noise_ae = get_models("AIMC results/No Noise/ddpm_model/model_final.pth",
                                        "AIMC results/No Noise/enc_dec_model/final_model.pt")
diffusion = LatentDiffusion(latent_dimension=128)

train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

random_indexes = [random.randint(0, len(train_dataset)-1) for _ in range(10)]
prompts = [train_dataset[i][1] for i in random_indexes]

generated_map = {}
for prompt in prompts:
    prompt_repeated = [prompt] * 10  # Copy the same prompt 10 times in a list
    # Generate 10 MIDI files
    # Assume diffusion.sample_conditional returns a batch of generated MIDI files based on the prompt
    no_noise_ddpm.eval()
    sampled_midi = diffusion.sample_conditional(no_noise_ddpm, n=10, text_keywords=prompt_repeated,
                                                midi_decoder=no_noise_ae)
    generated_map[prompt] = sampled_midi

same_prompt_hamming_distances = []
same_prompt_euclidean_distances = []
for _, generated_variants in tqdm(generated_map.items()):
    for i in range(len(generated_variants) - 1):
        for j in range(i + 1, len(generated_variants)):
            m_x, m_y = generated_variants[i], generated_variants[j]

            m_x_binary, m_y_binary = copy.deepcopy(m_x), copy.deepcopy(m_y)
            m_x_binary, m_y_binary = (np.array(m_x_binary) * 255).astype(int), (np.array(m_y_binary) * 255).astype(int)
            # Binarize both pianorolls m_x and m_y and compute Hamming distance using xor
            m_x_binary[m_x_binary <= 5] = 0
            m_x_binary[m_x_binary > 5] = 1
            m_y_binary[m_y_binary <= 5] = 0
            m_y_binary[m_y_binary > 5] = 1
            hamming_dis = np.sum(np.logical_xor(m_x_binary.flatten(), m_y_binary.flatten()))
            # Pass both m_x and m_y through the encoder to get z_x and z_y and compute euclidean distance
            m_x = m_x.unsqueeze(0).permute(0, 2, 1) / 255
            m_y = m_y.unsqueeze(0).permute(0, 2, 1) / 255
            enc_out = no_noise_ae.encode(torch.stack([m_x.squeeze(), m_y.squeeze()]))
            z_x, z_y = enc_out[0], enc_out[1]
            euclidean_dis = torch.sqrt(torch.sum((z_x - z_y) ** 2))
            same_prompt_hamming_distances.append(hamming_dis)
            same_prompt_euclidean_distances.append(euclidean_dis.detach().item())

different_prompt_hamming_distances = []
different_prompt_euclidean_distances = []
for _ in tqdm(range(1000)):
    i, j = 0, 0
    while i == j:
        i, j = random.randint(0, len(prompts) - 1), random.randint(0, len(prompts) - 1)
    bucket_i = generated_map[prompts[i]]
    bucket_j = generated_map[prompts[j]]

    x, y = random.randint(0, len(bucket_i) - 1), random.randint(0, len(bucket_j) - 1)

    m_x, m_y = bucket_i[x], bucket_j[y]

    m_x_binary, m_y_binary = copy.deepcopy(m_x), copy.deepcopy(m_y)
    m_x_binary, m_y_binary = (np.array(m_x_binary) * 255).astype(int), (np.array(m_y_binary) * 255).astype(int)
    # Binarize both pianorolls m_x and m_y and compute Hamming distance using xor
    m_x_binary[m_x_binary <= 5] = 0
    m_x_binary[m_x_binary > 5] = 1
    m_y_binary[m_y_binary <= 5] = 0
    m_y_binary[m_y_binary > 5] = 1
    hamming_dis = np.sum(np.logical_xor(m_x_binary.flatten(), m_y_binary.flatten()))
    # Pass both m_x and m_y through the encoder to get z_x and z_y and compute euclidean distance
    m_x = m_x.unsqueeze(0).permute(0, 2, 1) / 255
    m_y = m_y.unsqueeze(0).permute(0, 2, 1) / 255
    enc_out = no_noise_ae.encode(torch.stack([m_x.squeeze(), m_y.squeeze()]))
    z_x, z_y = enc_out[0], enc_out[1]
    euclidean_dis = torch.sqrt(torch.sum((z_x - z_y) ** 2))
    different_prompt_hamming_distances.append(hamming_dis)
    different_prompt_euclidean_distances.append(euclidean_dis.detach().item())

# Print min mean std for same_text_hamming_distances and same_text_euclidean_distances
same_text_min_h, same_text_mean_h, same_text_std_h = np.min(same_prompt_hamming_distances), \
                                                     np.mean(same_prompt_hamming_distances), \
                                                     np.std(same_prompt_hamming_distances)
same_text_min_e, same_text_mean_e, same_text_std_e = np.min(same_prompt_euclidean_distances), \
                                                     np.mean(same_prompt_euclidean_distances), \
                                                     np.std(same_prompt_euclidean_distances)

# Print min mean std for random_text_hamming_distances and random_text_euclidean_distances
diff_text_min_h, diff_text_mean_h, diff_text_std_h = np.min(different_prompt_hamming_distances), \
                                                     np.mean(different_prompt_hamming_distances), \
                                                     np.std(different_prompt_hamming_distances)
diff_text_min_e, diff_text_mean_e, diff_text_std_e = np.min(different_prompt_euclidean_distances), \
                                                     np.mean(different_prompt_euclidean_distances), \
                                                     np.std(different_prompt_euclidean_distances)

# Assuming the calculation of min, mean, and std for both same_text and random_text distances have been done

print("Statistics for Same Text Comparisons:")
print(f"Hamming Distance: Min = {same_text_min_h:.2f}, Mean = {same_text_mean_h:.2f}, Std Dev = {same_text_std_h:.2f}")
print(
    f"Euclidean Distance: Min = {same_text_min_e:.2f}, Mean = {same_text_mean_e:.2f}, Std Dev = {same_text_std_e:.2f}\n")

print("Statistics for Different Text Comparisons:")
print(f"Hamming Distance: Min = {diff_text_min_h:.2f}, Mean = {diff_text_mean_h:.2f}, Std Dev = {diff_text_std_h:.2f}")
print(
    f"Euclidean Distance: Min = {diff_text_min_e:.2f}, Mean = {diff_text_mean_e:.2f}, Std Dev = {diff_text_std_e:.2f}")

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Plot for Hamming distances
sns.kdeplot(same_prompt_hamming_distances, ax=axs[0], color='blue', label='Same Text Hamming')
sns.kdeplot(different_prompt_hamming_distances, ax=axs[0], color='orange', label='Diff Text Hamming')
axs[0].set_xlabel('Hamming Distance')
axs[0].set_ylabel('Density')
axs[0].legend()
axs[0].set_title('Hamming Distance Distributions')

# Plot for Euclidean distances
sns.kdeplot(same_prompt_euclidean_distances, ax=axs[1], color='green', label='Same Text Euclidean')
sns.kdeplot(different_prompt_euclidean_distances, ax=axs[1], color='red', label='Diff Text Euclidean')
axs[1].set_xlabel('Euclidean Distance')
axs[1].set_ylabel('Density')
axs[1].legend()
axs[1].set_title('Euclidean Distance Distributions')

plt.tight_layout()
plt.show()
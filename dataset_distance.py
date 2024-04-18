import copy
import random
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
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

train_dataset = load_or_process_dataset(dataset_dir=config['dataset_dir'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
mh_dataset = defaultdict(list)
for drum_beats, text_tags in tqdm(train_loader):
    text_mh_emb = MultiHotEncoderWithBPM.encode_batch(text_tags)[0]
    text_mh_emb_str = mh_vector_to_string(text_mh_emb)
    mh_dataset[text_mh_emb_str].append(drum_beats[0])

mh_dataset = list(filter(lambda x: len(x[1]) > 5, [(k, v) for k, v in mh_dataset.items()]))

same_text_hamming_distances = []
same_text_euclidean_distances = []

for i in tqdm(range(1000)):
    r_i = random.randint(0, len(mh_dataset)-1)
    bucket = mh_dataset[r_i]
    x, y = 0, 0
    while x == y:
        x, y = random.randint(0, len(bucket[1])-1), random.randint(0, len(bucket[1])-1)
    m_x, m_y = bucket[1][x], bucket[1][y]
    m_x_binary, m_y_binary = copy.deepcopy(m_x), copy.deepcopy(m_y)
    m_x_binary, m_y_binary = (np.array(m_x_binary) * 255).astype(int), (np.array(m_y_binary) * 255).astype(int)
    # Binarize both pianorolls m_x and m_y and compute Hamming distance using xor
    m_x_binary[m_x_binary <= 5] = 0
    m_x_binary[m_x_binary > 5] = 1
    m_y_binary[m_y_binary <= 5] = 0
    m_y_binary[m_y_binary > 5] = 1

    hamming_dis = np.sum(np.logical_xor(m_x_binary, m_y_binary))
    # Pass both m_x and m_y through the encoder to get z_x and z_y and compute euclidean distance
    z_x, z_y = no_noise_ae.encode(m_x.unsqueeze(0)), no_noise_ae.encode(m_y.unsqueeze(0))
    euclidean_dis = torch.sqrt(torch.sum((z_x - z_y) ** 2))
    same_text_hamming_distances.append(hamming_dis)
    same_text_euclidean_distances.append(euclidean_dis.detach().item())

# Randomly sample 2 datapoints from the dataset
# Repeat 'n' times while computing pairwise distances between them (both in pianoroll and latent)
train_loader_2 = DataLoader(train_dataset, batch_size=2, shuffle=True)

random_text_hamming_distances = []
random_text_euclidean_distances = []
for _ in tqdm(range(1000)):
    # Binarize both pianorolls m_x and m_y and compute Hamming distance using xor
    x, y = 0, 0
    while x == y:
        x, y = random.randint(0, len(train_dataset)-1), random.randint(0, len(train_dataset)-1)
    (m_x, _), (m_y, _) = train_dataset[x], train_dataset[y]
    m_x_binary, m_y_binary = copy.deepcopy(m_x), copy.deepcopy(m_y)
    m_x_binary, m_y_binary = (np.array(m_x_binary) * 255).astype(int), (np.array(m_y_binary) * 255).astype(int)
    # Binarize both pianorolls m_x and m_y and compute Hamming distance using xor
    m_x_binary[m_x_binary <= 5] = 0
    m_x_binary[m_x_binary > 5] = 1
    m_y_binary[m_y_binary <= 5] = 0
    m_y_binary[m_y_binary > 5] = 1
    hamming_dis = np.sum(np.logical_xor(m_x_binary, m_y_binary))
    # Pass both m_x and m_y through the encoder to get z_x and z_y and compute euclidean distance
    z_x, z_y = no_noise_ae.encode(m_x.unsqueeze(0)), no_noise_ae.encode(m_y.unsqueeze(0))
    euclidean_dis = torch.sqrt(torch.sum((z_x - z_y) ** 2))
    random_text_hamming_distances.append(hamming_dis)
    random_text_euclidean_distances.append(euclidean_dis.detach().item())


# Print min mean std for same_text_hamming_distances and same_text_euclidean_distances
same_text_min_h, same_text_mean_h, same_text_std_h = np.min(same_text_hamming_distances), \
                                                     np.mean(same_text_hamming_distances), \
                                                     np.std(same_text_hamming_distances)
same_text_min_e, same_text_mean_e, same_text_std_e = np.min(same_text_euclidean_distances), \
                                                     np.mean(same_text_euclidean_distances), \
                                                     np.std(same_text_euclidean_distances)

# Print min mean std for random_text_hamming_distances and random_text_euclidean_distances
random_text_min_h, random_text_mean_h, random_text_std_h = np.min(random_text_hamming_distances), \
                                                     np.mean(random_text_hamming_distances), \
                                                     np.std(random_text_hamming_distances)
random_text_min_e, randon_text_mean_e, random_text_std_e = np.min(random_text_euclidean_distances), \
                                                     np.mean(random_text_euclidean_distances), \
                                                     np.std(random_text_euclidean_distances)

# Assuming the calculation of min, mean, and std for both same_text and random_text distances have been done

print("Statistics for Same Text Comparisons:")
print(f"Hamming Distance: Min = {same_text_min_h:.2f}, Mean = {same_text_mean_h:.2f}, Std Dev = {same_text_std_h:.2f}")
print(f"Euclidean Distance: Min = {same_text_min_e:.2f}, Mean = {same_text_mean_e:.2f}, Std Dev = {same_text_std_e:.2f}\n")

print("Statistics for Random Text Comparisons:")
print(f"Hamming Distance: Min = {random_text_min_h:.2f}, Mean = {random_text_mean_h:.2f}, Std Dev = {random_text_std_h:.2f}")
print(f"Euclidean Distance: Min = {random_text_min_e:.2f}, Mean = {randon_text_mean_e:.2f}, Std Dev = {random_text_std_e:.2f}")

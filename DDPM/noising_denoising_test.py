import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from DDPM.ddpm import Diffusion
from DDPM.main import load_or_process_dataset, load_config
from DDPM.model import ConditionalEncDecMHA
from text_supervised_pretraining.model import CLAMP
from utils.utils import save_numpy_as_midi

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize dataset and dataloader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print(f"Len of dataset: {len(train_dataset)}")

# Initialize models and optimizer
clamp_model = CLAMP().to(device)
clamp_model.load_state_dict(torch.load("text_supervised_pretraining/trained_models/0210_2142/model_final.pth"))
clamp_model.eval()
print("Loaded the pretrained model successfully")

config_path = 'DDPM/config.yaml'
config = load_config(config_path)
ddpm_model = ConditionalEncDecMHA(config['time_embedding_dimension'], clamp_model.latent_dimension, device).to(device)
ddpm_model.load_state_dict(torch.load(config['ddpm_model_path']))
ddpm_model.eval()
diffusion = Diffusion(num_time_slices=128)

noising_steps = [10, 25, 50, 100, 200, 500]
num_samples = 4

for i in range(num_samples):
    random_idx = random.randint(0, len(train_loader) - 1)
    midi, text = train_dataset[random_idx]

    # Get the text tag
    text_tag = '_'.join(text.split())

    # Save the original MIDI file
    original_folder = "DDPM/noising_test/original"
    os.makedirs(original_folder, exist_ok=True)
    original_file_path = os.path.join(original_folder, f"original_{text_tag}.mid")
    save_numpy_as_midi(original_file_path, (midi * 127).type(torch.uint8).numpy(), ghost_threshold=5)

    for noise_steps in tqdm(noising_steps, desc=f"Iter {i}"):
        # Create directory for noised and denoised MIDI files
        noise_folder = f"DDPM/noising_test/noised{noise_steps}"
        denoise_folder = f"DDPM/noising_test/denoised{noise_steps}"
        os.makedirs(noise_folder, exist_ok=True)
        os.makedirs(denoise_folder, exist_ok=True)

        # Noise the MIDI
        noise_steps_tensor = torch.tensor(noise_steps).unsqueeze(0)
        noised_midi, _ = diffusion.noise_drum_beats(midi, noise_steps_tensor)
        noised_midi_int = (noised_midi.clamp(0, 1).squeeze() * 127).type(torch.uint8).numpy()
        noised_file_path = os.path.join(noise_folder, f"noised{noise_steps}_{text_tag}.mid")
        save_numpy_as_midi(noised_file_path, noised_midi_int, ghost_threshold=5)

        # Denoise the MIDI
        text_embeddings = clamp_model.get_text_embeddings(text)
        noised_midi = torch.tensor(noised_midi).unsqueeze(0)
        denoised_midi = diffusion.denoise_drum_beats(ddpm_model, noised_midi.squeeze(0), noise_steps, text_embeddings)
        denoised_file_path = os.path.join(denoise_folder, f"denoised{noise_steps}_{text_tag}.mid")
        save_numpy_as_midi(denoised_file_path, denoised_midi.squeeze().numpy(), ghost_threshold=5)
        print("1 full iteration successful.")





import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from DDPM.ddpm import Diffusion
from DDPM.main import load_or_process_dataset, load_config
from DDPM.model import ConditionalEncDecMHA
from unsupervised_pretraining.model import CLAMP

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize dataset and dataloader
train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Len of dataset: {len(train_dataset)}")

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

# Randomly sample 128 MIDI files from the dataset
random_idxs = torch.randint(0, len(train_dataset), (128,))
samples = [train_dataset[i] for i in random_idxs]
drum_beat_samples, text_samples = list(map(lambda x: x[0], samples)), list(map(lambda x: x[1], samples))
drum_beat_samples = torch.stack(drum_beat_samples)
active_notes_arr = drum_beat_samples[drum_beat_samples > 5/128]
# Initialize lists to store velocities greater than 5/127
generated_high_velocities = []

# Generate drum beats in batches of 32xw
for i in tqdm(range(0, len(drum_beat_samples), 32)):
    batch_text_samples = text_samples[i:i+32]
    generated_drum_beats = diffusion._sample_conditional(ddpm_model, clamp_model.get_text_embeddings(batch_text_samples).squeeze())

    # Compute velocity statistics for the original and generated MIDI files
    generated_velocities = generated_drum_beats[generated_drum_beats > 5/127]

    # Add velocities greater than 5/127 to the lists
    generated_high_velocities.extend(generated_velocities)

generated_high_velocities = list(map(lambda x: x.detach().cpu().item(), generated_high_velocities))
# Plot the distribution of all velocities
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(active_notes_arr, label='Original MIDI Velocities (High)', color='blue')
sns.kdeplot(generated_high_velocities, label='Generated MIDI Velocities (High)', color='orange')
plt.xlabel('Velocity')
plt.ylabel('Density')
plt.title('Distribution of High Velocities (Velocity > 5/127)')
plt.legend()

# Save the plot as a JPEG file
plt.savefig('./velocity_distribution_plot.jpg')

# Show the plot
plt.show()

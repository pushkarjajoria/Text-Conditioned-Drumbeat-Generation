import json
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.midi_processing.mid2numpy import save_numpy_as_midi


class MusicDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)
        return x


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_midi(midis, midi_path, epoch=None, ghost_threshold=5, file_names=None, resolution=4):
    name_str = f"Epoch{epoch}" if epoch else "Final model"
    folder_path = os.path.join(midi_path, name_str)
    os.makedirs(folder_path, exist_ok=True)
    for i, midi_pianoroll in enumerate(midis):
        filename = file_names[i] + ".mid" if file_names else f"Sample{i}.mid"
        save_numpy_as_midi(os.path.join(folder_path, filename), midi_pianoroll, ghost_threshold, resolution)


def save_midi_demo(midis, midi_path, epoch=None, ghost_threshold=5, file_names=None, resolution=4):
    folder_path = midi_path
    os.makedirs(folder_path, exist_ok=True)
    for i, midi_pianoroll in enumerate(midis):
        filename = file_names[i] + ".mid" if file_names else f"Sample{i}.mid"
        save_numpy_as_midi(os.path.join(folder_path, filename), midi_pianoroll, ghost_threshold, resolution)


def save_midi_without_structure(midis, midi_path, ghost_threshold=5, file_names=None, resolution=4):
    folder_path = os.path.join(midi_path)
    os.makedirs(folder_path, exist_ok=True)
    for i, midi_pianoroll in enumerate(midis):
        # filename = file_names[i] + ".mid" if file_names else f"Sample{i}.mid"
        save_numpy_as_midi(os.path.join(folder_path, file_names[i]), midi_pianoroll, ghost_threshold, resolution)


def get_data(args):
    data = np.load(args.dataset_path)
    data = data.astype(np.float32)
    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
    # Scaling values from -1 to 1
    data = ((data/127) * 2) - 1
    # create the dataset
    dataset = MusicDataset(data)
    # create the dataloader object
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(args):
    os.makedirs("../../checkpoint", exist_ok=True)
    os.makedirs("../../results", exist_ok=True)
    os.makedirs(os.path.join("../../checkpoint", args.run_name), exist_ok=True)
    os.makedirs(os.path.join("../../results", args.run_name), exist_ok=True)
    with open(f'checkpoint/{args.run_name}/hyperparamets.txt', 'w') as f:
        # Write dictionary to file as JSON
        f.write(json.dumps(vars(args), indent=4))


if __name__ == "__main__":
    midis = np.random.rand(2, 9, 64)
    midis *= 127
    save_midi(midis, os.path.join("../results", "DDPM_Unconditional_groove_monkee"), 100)
import json
import os

import numpy.matlib
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from midi_processing.mid2numpy import numpy2midi, save_numpy_as_midi


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


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def save_midi(midis, midi_path, epoch, ghost_threshold=5):
    folder_path = os.path.join(midi_path, f"Epoch{epoch}")
    os.makedirs(folder_path, exist_ok=True)
    for i, midi_pianoroll in enumerate(midis):
        save_numpy_as_midi(os.path.join(folder_path, f"Sample{i}.mid"), midi_pianoroll, ghost_threshold)


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


def get_image_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    with open(f'models/{args.run_name}/hyperparamets.txt', 'w') as f:
        # Write dictionary to file as JSON
        f.write(json.dumps(vars(args), indent=4))


if __name__ == "__main__":
    midis = np.random.rand(2, 9, 64)
    midis *= 127
    save_midi(midis, os.path.join("results", "DDPM_Unconditional_groove_monkee"), 100)
import os
import fnmatch
from collections import Counter
import torch
from torch.utils.data import Dataset

from utils.midi_processing.mid2numpy import read_midi, midi2numpy


class MidiDataset(Dataset):
    def __init__(self, file_tag_map, transform=None):
        self.midi_tensors = []
        self.tags = []
        for file_path, tag in file_tag_map.items():
            # Convert MIDI to NumPy array and then to PyTorch tensor at load time
            numpy_data = midi2numpy(read_midi(file_path))
            midi_tensor = torch.from_numpy(numpy_data)

            self.midi_tensors.append(midi_tensor)
            self.tags.append(tag)

        self.transform = transform

    def __len__(self):
        return len(self.midi_tensors)

    def __getitem__(self, idx):
        midi_tensor = self.midi_tensors[idx]
        tags = self.tags[idx]

        # Optional transformation
        if self.transform:
            midi_tensor = self.transform(midi_tensor)

        return midi_tensor, tags


def get_top_tags(fileTagMap, topN):
    # Create a Counter object to hold the tags and their frequencies
    tagCounter = Counter()

    # Iterate through the fileTagMap and update the Counter with tags
    for tags in fileTagMap.values():
        tagCounter.update(tags)

    # Get the topN most common tags
    topTags = tagCounter.most_common(topN)

    return topTags


def get_filenames_and_tags(dataset_dir='../datasets/Groove_Monkee_Mega_Pack_GM'):
    # Dictionary to store file paths and tags
    file_tag_map = {}

    # Walk through directory
    for dir_name, subdir_list, file_list in os.walk(dataset_dir):
        for fname in fnmatch.filter(file_list, '*.mid'):
            # Extract tags from parent folder names and the file name
            tags = dir_name.split(os.sep)[1:] + fname.split('.')[0].split('_')
            tags = [tag for tag_part in tags for tag in tag_part.split(' ')]

            # Store in dictionary with file path as the key
            file_path = os.path.join(dir_name, fname)
            file_tag_map[file_path] = tags

    return file_tag_map

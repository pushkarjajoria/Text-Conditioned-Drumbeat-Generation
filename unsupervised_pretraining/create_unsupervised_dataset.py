import collections
import os
import fnmatch
import pickle
from collections import Counter
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.midi_processing.mid2numpy import read_midi, midi2numpy


class MidiDataset(Dataset):
    def __init__(self, file_tag_map, transform=None):
        self.midi_tensors = []
        self.tags = []
        for file_path, tag in file_tag_map.items():
            try:
                # Convert MIDI to NumPy array and then to PyTorch tensor at load time
                numpy_data = midi2numpy(read_midi(file_path))
                if numpy_data.shape[0] != 9 or numpy_data.shape[1] != 64:
                    # For now, we are only working with 9 instruments in the midi file and 64 timeslices. The plan is to
                    # include other time signatures in the training aswell.
                    continue
                midi_tensor = torch.from_numpy(numpy_data)
                midi_tensor = midi_tensor/127.  # Normalize the midi date from [0-1]
                self.midi_tensors.append(midi_tensor)
                self.tags.append(tag)
            except ZeroDivisionError as e:
                print(f"Unable to process {file_path}")
                print(e.args)
                continue

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


def get_top_tags(file_tag_map, top_n):
    # Create a Counter object to hold the tags and their frequencies
    tag_counter = Counter()

    # Iterate through the fileTagMap and update the Counter with tags
    for tags in file_tag_map.values():
        tag_counter.update(tags)

    # Get the topN most common tags
    topTags = tag_counter.most_common(top_n)

    return topTags


def get_filenames_and_tags(dataset_dir='../../datasets/Groove_Monkee_Mega_Pack_GM', filter_common_tags=True):
    # Dictionary to store file paths and tags
    file_tag_map = {}
    filter_list = ["..", "datasets", "Groove_Monkee_Mega_Pack_GM", "GM", "Bonus"]
    # Walk through directory
    for dir_name, subdir_list, file_list in os.walk(dataset_dir):
        for fname in fnmatch.filter(file_list, '*.mid'):
            # Extract tags from parent folder names and the file name
            tags = dir_name.split(os.sep)[1:] + fname.split('.')[0].split('_')
            tags = [tag for tag_part in tags for tag in tag_part.split(' ')]

            # Store in dictionary with file path as the key
            file_path = os.path.join(dir_name, fname)
            if filter_common_tags:
                tags = list(filter(lambda x: x not in filter_list, tags))
            tags = " ".join(tags)
            file_tag_map[file_path] = tags

    return file_tag_map


if __name__ == "__main__":
    file_name_and_tags = get_filenames_and_tags()
    dataset = {}  # Dictionary to store NumPy arrays along with their tags
    timeslices = []  # List to store the first dimensions for the histogram
    second_dimensions = set()  # Set to store unique second dimensions

    for midi_path, midi_tags in tqdm(file_name_and_tags.items()):
        try:
            np_drum_track = midi2numpy(read_midi(midi_path))
            num_timeslices = np_drum_track.shape[1]
            timeslices.append(num_timeslices)
            if num_timeslices not in second_dimensions:
                print(f"{num_timeslices} -> {midi_path}")
            second_dimensions.add(num_timeslices)
            dataset[midi_path] = {'tags': midi_tags, 'data': np_drum_track, 'timeslices': num_timeslices}
        except Exception as e:
            print(f"Error processing file {midi_path}: {str(e)}")
            continue

    # Save the dataset as a pickle file
    pickle_file = "../../datasets/midi_dataset.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {pickle_file}")

    # Plot a histogram of the first dimensions of the NumPy arrays
    timeslice_counter = collections.Counter(timeslices)
    print("Counts of each unique timeslices:")
    for size, count in sorted(timeslice_counter.items(), key=lambda x: (x[1], -x[0]), reverse=True):
        print(f"Size: {size}, Count: {count}")

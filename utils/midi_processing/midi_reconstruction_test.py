import collections
import random

import numpy as np
from matplotlib import pyplot as plt
import pypianoroll
from tqdm import tqdm

from unsupervised_pretraining.create_unsupervised_dataset import get_filenames_and_tags
from utils.midi_processing.mid2numpy import read_midi, midi2numpy
from utils.utils import save_midi


def midi_to_numpy(file_path):
    # Load a MIDI file as a Multitrack object
    multitrack = pypianoroll.read(file_path)

    # For simplicity, let's assume we only deal with the first track
    # and you would adjust this logic based on your specific use case
    # Especially considering you are interested in 9 specific drums
    track = multitrack.tracks[0]
    pianoroll = track.pianoroll
    return pianoroll


class MidiReconstructionStatistics:
    def __init__(self, genres):
        self.average_mse = 0
        self.genres = genres
        self.genres_mse = {genre: [] for genre in genres}
        # Initialize more statistics variables as needed

    def publish_stats(self):
        # Calculate and print average MSE
        all_mse_values = [mse for genre_mses in self.genres_mse.values() for mse in genre_mses]
        self.average_mse = np.mean(all_mse_values)
        print(f"Average MSE: {self.average_mse}")

        # Print MSE by genre
        for genre, mses in self.genres_mse.items():
            if mses:
                print(f"{genre} - Average MSE: {np.mean(mses)}")
            else:
                print(f"{genre} - No data")

        # Optionally, create plots
        plt.figure(figsize=(10, 5))
        genre_mse_avgs = [np.mean(mses) if mses else 0 for genre, mses in self.genres_mse.items()]
        plt.bar(self.genres, genre_mse_avgs)
        plt.title("Average MSE by Genre")
        plt.xlabel("Genre")
        plt.ylabel("MSE")
        plt.show()

    def compare_midi_files(self, midi1, midi2, folder_tags: str):
        # This is a simplified approach to compare MIDI files based on their NumPy representations.
        # A more sophisticated method might be required for a comprehensive comparison.
        mse = np.mean((midi1 - midi2) ** 2)
        self.genres_mse[folder_tags].append(mse)


if __name__ == "__main__":
    file_name_and_tags = get_filenames_and_tags(dataset_dir="../../datasets/Groove_Monkee_Mega_Pack_GM")
    counter = 0
    for midi_path, midi_tags in tqdm(file_name_and_tags.items()):
        try:
            np_drum_track = midi2numpy(read_midi(midi_path))
            num_timeslices = np_drum_track.shape[1]
            if num_timeslices == 128:
                if (chance := random.uniform(0, 1)) <= 0.99:
                    continue
                print(chance)
                midi_name = midi_path.split("/")[-1].split(".")[0]
                print(f"\nReconstructing: {midi_path}")
                save_midi([np_drum_track], f"../../DDPM/reconstruction/drum_tracks/", file_names=[midi_name], resolution=1)
                counter += 1
        except Exception as e:
            # print(f"Error processing file {midi_path}: {str(e)}")
            continue
        if counter >= 10:
            break
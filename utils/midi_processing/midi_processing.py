import os
import numpy as np
from mido import MidiFile, MetaMessage
from tqdm import tqdm
from vector_to_midi import write_midi_and_audio_files_from_vectors


def read_midi_files_and_return_vectors_and_tags(root_directory):
    """Recursively read MIDI files from a nested directory with subfolders and converts them into 16th note vectors
    with all folders and parent folder as the corresponding tags for each vector"""
    all_vectors = []
    for subdir, dirs, files in tqdm(os.walk(root_directory)):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Check if the file is a MIDI file
            if file_path.endswith('.mid') or file_path.endswith('.midi'):
                try:
                    mid = MidiFile(file_path)
                except:
                    print('Error reading MIDI file:', file_path)
                    continue
                # Extract all subfolder names and root directory as tags for the vector
                tag = []
                for d in os.path.relpath(subdir, root_directory).split(os.sep):
                    tag.append(d)
                # Extract the tempo from the MIDI file
                tempo = None
                for msg in mid:
                    if isinstance(msg, MetaMessage) and msg.type == 'set_tempo':
                        tempo = msg.tempo
                        break
                if tempo is None:
                    print('Error: no tempo found in MIDI file:', file_path)
                    continue
                # Calculate the 16th note delta time of the MIDI file
                ticks_per_beat = mid.ticks_per_beat
                ticks_per_sixteenth_note = ticks_per_beat / 4
                # Create the 2D array for the track with the MIDI drum instrument on one axis and the 16th note on another axis
                max_time = int(mid.length * tempo * 50000 / ticks_per_beat)  # Max time in ticks
                max_instr = 128
                track_array = [[0 for x in range(16)] for y in range(max_instr)]
                # Fill the array with the velocities of the notes played
                for i, track in enumerate(mid.tracks):
                    time_counter = 0
                    for message in track:
                        time_counter += message.time
                        if message.type == 'note_on':
                            time = int(time_counter / ticks_per_sixteenth_note)
                            instr = message.note
                            velocity = message.velocity
                            # Modify the assigned indices according to the updated 2D array shape
                            track_array[instr][time % 16] = velocity
                all_vectors.append({"vector": np.array(track_array), "tag": tag})
    return all_vectors


# Usage example
if __name__ == "__main__":
    root_directory = '/Users/pushkarjajoria/Downloads/Groove Monkee Free MIDI GM'
    vectors = read_midi_files_and_return_vectors_and_tags(root_directory)
    write_midi_and_audio_files_from_vectors([vectors[0]], "./")
    print(vectors)
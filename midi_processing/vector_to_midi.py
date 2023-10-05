import os
import mido
import numpy as np


def write_midi_and_audio_files_from_vectors(vector_list, output_directory, bpm=120):
    """Writes single-track MIDI and WAV files from the drum notes in the input list"""

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create MIDI track with drum notes
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set the time signature, tempo, and drum channel
    ticks_per_beat = mid.ticks_per_beat
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8))
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message('program_change', program=0, channel=9, time=0))

    # Iterate over each vector and generate drum notes
    for vector in vector_list:
        # Convert vector to note array with 16th note resolution
        note_array = np.array(vector)
        note_array = np.repeat(note_array, mid.ticks_per_beat // 4)
        note_array = np.tile(note_array, 4)
        note_array = note_array[:mid.ticks_per_beat * 4]
        note_array = note_array.tolist()

        # Append drum notes to MIDI track
        current_tick = 0
        for note in note_array:
            if note != 0:
                track.append(mido.Message('note_on', note=note, velocity=100, time=current_tick))
                track.append(mido.Message('note_off', note=note, velocity=0, time=mid.ticks_per_beat // 16))
            current_tick += mid.ticks_per_beat // 16

    # Write MIDI file
    midi_file_path = os.path.join(output_directory, '../output.mid')
    mid.save(midi_file_path)


# Usage example
if __name__ == "__main__":
    vectors = [[0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42],
               [36, 0, 0, 0, 42, 0, 0, 0, 36, 0, 0, 0, 42, 0, 0, 0]]
    bpm = 120
    output_directory = './'
    write_midi_and_audio_files_from_vectors(vectors, output_directory, bpm)

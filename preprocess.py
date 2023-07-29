
import music21 as m21
import os
import json
from tensorflow import keras
import numpy as np

KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [
     0.25, 
     0.50,
     0.75,
     1.0,
     1.5,
     2,
     3,
     4
]
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64;
MAPPPING_PATH = "mapping.json"

def load_songs_in_kern(dataset_path):
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs 
                  
def has_acceptable_duration(song, acceptable_durations):
    for note in song.flatten().notesAndRests:
          if note.duration.quarterLength not in acceptable_durations:
               return False
    return True


#changes major notes to C-major and minor notes to A-minor
def transpose(song):
     parts = song.getElementsByClass(m21.stream.Part)
     measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
     key = measures_part0[0][4]

     if not isinstance(key, m21.key.Key):
          key = song.analyze("key")
    
     if key.mode == "major":
          interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
     elif key.mode == "minor":
          interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

     transposed_song = song.transpose(interval)
     
     return transposed_song

def encode_song(song, time_step = 0.25):
     encoded_song = []
     for event in song.flatten().notesAndRests:

          if isinstance(event, m21.note.Note):
               symbol = event.pitch.midi
          elif isinstance(event, m21.note.Rest):
               symbol = "r"

          steps = int(event.duration.quarterLength / time_step)

          for step in range(steps):
               if step == 0:
                    encoded_song.append(symbol)
               else:
                    encoded_song.append('_')

     encoded_song = " ".join(map(str, encoded_song))
     return encoded_song



def preprocess(dataset_path):
       print("Loading songs....")
       songs = load_songs_in_kern(dataset_path)
       print(f"loaded {len(songs)} songs.")

       for i, song in enumerate(songs):
            if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
                 continue
            
            song = transpose(song)

            encoded_song = encode_song(song)

            save_path = os.path.join(SAVE_DIR, str(i))
            with open(save_path, "w") as fp:
               fp.write(encoded_song)

def load(file_path):
     with open(file_path,"r") as fp:
          song = fp.read()
     return song

#creates a single file that contains all songs symbols
def create_single_file_dataset(dataset_path,  file_dataset_path, sequence_length):
     new_song_delimiter = "/ " * sequence_length
     songs = ""

     for path, _, files in os.walk(dataset_path):
          for file in files:
               file_path = os.path.join(path, file)
               song = load(file_path)
               songs = songs + song + " " + new_song_delimiter
     songs = songs[:-1]

     with open(file_dataset_path, "w") as fp:
          fp.write(songs)
     return songs
          
#creates a json file with symbol dictionary
def create_mapping(songs, mapping_path):
     mappings = {}
     songs = songs.split()
     vocabulary = list(set(songs))

     for i, symbol in enumerate(vocabulary):
          mappings[symbol] = i

     with open(mapping_path, "w") as fp:
          json.dump(mappings, fp, indent = 4)

#convert the sybols to int
def convert_songs_to_int(songs):
     int_songs  = []
     with open(MAPPPING_PATH, "r") as fp:
          mappings = json.load(fp)

     songs = songs.split()
     for symbol in songs:
          int_songs.append(mappings[symbol])
     
     return int_songs

def generate_training_sequences(sequence_length):
      
      songs = load(SINGLE_FILE_DATASET)
      int_songs = convert_songs_to_int(songs)
      
      inputs = []
      targets = []
      num_sequences = len(int_songs) - sequence_length

      for i in range(num_sequences):
           inputs.append(int_songs[i:i + sequence_length])
           targets.append(int_songs[sequence_length + i])
     
      vocabulary_size = len(set(int_songs))
      inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
      targets = np.array(targets)

      return inputs, targets
      
          
def main():
     preprocess(KERN_DATASET_PATH)
     songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
     create_mapping(songs, MAPPPING_PATH)
     inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
     

if __name__ == "__main__":
     main()
     """songs = load_songs_in_kern(KERN_DATASET_PATH)
     print(f" loaded{len(songs)} songs.")
     song = songs[3]
     preprocess(KERN_DATASET_PATH)
     transposed_song = transpose(song)
     transposed_song.show()
     songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)"""
     
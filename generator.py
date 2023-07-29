from tensorflow import keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPPING_PATH
import numpy as np
import music21 as m21

class Generator:

    def __init__(self, model_path = "model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPPING_PATH, "r") as fp:
            self.mappings = json.load(fp)
         
        self._start_symbols =["/"] * SEQUENCE_LENGTH

    def generate(self, seed, num_steps, max_sequence_length , temperature):
        seed = seed.split()
        melody = seed
        seed  = self._start_symbols + seed
        
        seed = [self.mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            seed = seed[-max_sequence_length:] 
            onehot_seed = keras.utils.to_categorical(seed, num_classes = len(self.mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)

            seed.append(output_int)

            output_symbol = [k for k, v in self.mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)
        return melody


    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p = probabilities)

        return index
    
    def save_melody(self, music, step_duration = 0.25 ,format = "midi", file_name = "m1.mid" ):
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        for i ,symbol in enumerate(music):

            if symbol != "_" or i + 1 == len(music) :
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)

                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)
                    
                    stream.append(m21_event)

                    step_counter = 1

                start_symbol = symbol


            else:
                step_counter += 1

        stream.write(format, file_name)


if __name__ == "__main__":
    g = Generator()
    seed = "77 _ 76 _ 79 _ _ _ 81 _ _ _ 76 _ _"
    music = g.generate(seed, 500, SEQUENCE_LENGTH, 0.8)
    print(music)
    g.save_melody(music)
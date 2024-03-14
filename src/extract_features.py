import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 

tfk = tf.keras 
tfkl = tf.keras.layers


class LstmAe(tfk.Model):
    def __init__(self, latent_dim, text_data):
        super().__init__()

        self.inputs = tfkl.InputLayer(
            input_shape=(171,), dtype=tf.string,
        )

        self.txt_vec = tfkl.TextVectorization(
            max_tokens=None, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=True,
            standardize="lower_and_strip_punctuation",
            )
        self.txt_vec.adapt(data=text_data, batch_size=8, steps=None)
        
        self.vocab = np.array(self.txt_vec.get_vocabulary())

        self.emb = tfkl.Embedding(
            input_dim=self.txt_vec.vocabulary_size(),
            output_dim=latent_dim,
            )
        self.enc = tfkl.Bidirectional(
            tfkl.LSTM(
                units=latent_dim,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="relu",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                return_sequences=True,
                name="encoder1"
                )
            )
        self.dec1 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=10,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="relu",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                return_sequences=True,
                name="decoder1"
            )
        )
        self.dec2 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=25,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="tanh",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                return_sequences=False,
                name="decoder2"
            )
        )
        self.outputs = tfkl.Dense(
            units=self.txt_vec.vocabulary_size(), activation="softmax"
            )

    def call(self, inputs):
        print(f"inputs:, {inputs.shape}")
        print(f"examples: {inputs[5:9]}")
        x = self.inputs(inputs)
        print(f"inputs: {inputs.numpy()[5:9]}")
        print(f"examples: {x[5:9]}")
        x = self.txt_vec(x)
        print(f"txt_vec: {x.shape}")
        print(f"examples: {self.vocab[100:105]}")
        x = self.emb(x)
        print(f"emb: {x.shape}")
        x = self.enc(x)
        print(f"enc: {x.shape}")
        x = self.dec1(x)
        print(f"dec1: {x.shape}")
        x = self.dec2(x)
        print(f"dec2: {x.shape}")
        x = self.outputs(x)
        print(f"outputs: {x.shape}")
        return x
    

class TrainTestLstmAe:
    def __init__(self, data: pd.DataFrame=None, n_epochs: int= 1):
        super().__init__()
        self.data = data
        self.n_epochs = n_epochs
    
    @staticmethod
    def get_preprocess_data(
        data_path: str="../data/medium_movies_data.csv", 
        vocab_size: int = 124100 # 124,079 precise
        ) -> tuple:

        data = pd.read_csv(data_path)
        text_data = data.Synopsis.values
        labels = data.Genre.values
        print(
            f"text data head: \n {text_data[:3]} \n" 
            f"text data shape: {text_data.shape} \n"
            f"labels head: \n {labels[:3]} \n"
            f"labels shape: {labels.shape} \n"
        )
        if vocab_size is None:  # a bit slower
            vocabulary = []
            max_seq_len = 0
            for synopsis in text_data:
                parsed_synopsis = np.unique(synopsis.lower().strip().split(" ")).tolist()
                if len(parsed_synopsis) > max_seq_len:
                    max_seq_len = len(parsed_synopsis)
                for word in parsed_synopsis:
                    if word not in vocabulary:
                        vocabulary.append(vocabulary)            
            vocab_size = len(vocabulary)
            n_classes = [i.lower() for i in np.unique(labels)]
            print(
                f"vocabulary size {len(vocabulary)}"
                f"Number of classes: {n_classes}"
                )
        else:
            max_seq_len = 171
        txt_vec = tfkl.TextVectorization(
            max_tokens=vocab_size, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=True,
            standardize="lower_and_strip_punctuation",
        )
        txt_vec.adapt(
            data=text_data, batch_size=8, steps=None
        )
        
        return txt_vec, labels, max_seq_len

    def get_text_data(
        data_path: str="../data/medium_movies_data.csv", 
        ) -> tuple:

        data = pd.read_csv(data_path)
        text_data = data.Synopsis.values
        labels = data.Genre.values
        print(
            f"text data head: \n {text_data[:3]} \n" 
            f"text data shape: {text_data.shape} \n"
            f"labels head: \n {labels[:3]} \n"
            f"labels shape: {labels.shape} \n"
        ) 

        return text_data, labels

    def train_val_test(self,):
        vectorized_text, labels = self.get_preprocess_data(
            data_path="../data/medium_movies_data.scv",
        )
        
        for k in range(5):
            print("....")





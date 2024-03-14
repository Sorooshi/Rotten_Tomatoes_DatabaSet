import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 

tfk = tf.keras 
tfkl = tf.keras.layers


class LstmAe(tfk.Model):
    def __init__(self, latent_dim):
        super().__init__()

        # self.latent_dim = latent_dim
        # self.embedding_dim = embedding_dim
        # self.vocab_size = vocab_size

        self.inputs = tfkl.Input(
            shape=(None, ), dtype=tf.string
        )

        self.txt_vec = tfkl.TextVectorization(
            max_tokens=None, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=True,
            standardize="lower_and_strip_punctuation",
        )

        self.emb = tfkl.Embedding(
            input_dim=self.txt_vec.vocabulary_size(),
            output_dim=latent_dim,
            )
        self.enc = tfkl.Bidirectional(
            tfkl.LSTM(
                units=latent_dim,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="relu",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                name="encoder 1"
                )
            )
        self.dec1 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=10,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="relu",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                name="decoder 1"
            )
        )
        self.dec2 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=20,  # hp.Int('units', min_value=2, max_value=100, step=5), 
                activation="tanh",  # hp.Choice("activation", ["relu", "tanh"]), 
                # dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1),
                name="decoder 2"
            )
        )

    def call(self, inputs):
        x = self.inputs(inputs)
        x = self.txt_vec.adapt(data=x, batch_size=8, steps=None)
        x = self.emb(inputs)
        x = self.enc(x)
        x = self.dec1(x)
        x = self.dec2(x)
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
            for synopsis in text_data:
                parsed_synopsis = np.unique(synopsis.lower().strip().split(" ")).tolist()
                for word in parsed_synopsis:
                    if word not in vocabulary:
                        vocabulary.append(vocabulary)            
            vocab_size = len(vocabulary)
            n_classes = [i.lower() for i in np.unique(labels)]
            print(
                f"vocabulary size {len(vocabulary)}"
                f"Number of classes: {n_classes}"
                )

        txt_vec = tfkl.TextVectorization(
            max_tokens=vocab_size, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=True,
            standardize="lower_and_strip_punctuation",
        )
        txt_vec.adapt(
            data=text_data, batch_size=8, steps=None
        )
        
        return txt_vec, labels   

    def train_val_test(self,):
        vectorized_text, labels = self.get_preprocess_data(
            data_path="../data/medium_movies_data.scv",
        )
        
        for k in range(5):
            print("....")





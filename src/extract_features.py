import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 

tfk = tf.keras 
tfkl = tf.keras.layers


class LstmAe(tfk.Model):
    def __init__(self, latent_dim, txt_vec):
        super().__init__()

        # self.latent_dim = latent_dim
        # self.embedding_dim = embedding_dim
        # self.vocab_size = vocab_size

        # self.inputs = tfkl.Input(
        #     shape=(1, ), dtype=tf.string
        # )
        self.emb = tfkl.Embedding(
            input_dim=len(txt_vec.get_vocabulary()),
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
        # x = self.input(inputs)
        x = self.txt_vec.adapt(data=inputs, batch_size=8, steps=None)
        x = self.emb(x)
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
    def load_preprocess_data(data_path, vocab_size, ):

        data = pd.read_csv(data_path)
        text_data = data.Synopsis.values
        labels = data.Genre.values
        print(
            f"data head: \n {data.head()} \n" 
            f"data shape: {data.shape} \n"
            f"text data head: \n {text_data[:5]} \n" 
            f"text data shape: {text_data.shape} \n"
            f"labels head: \n {labels[:5]} \n"
            f"labels shape: {labels.shape} \n"
        )
        AUTOTUNE = tf.data.AUTOTUNE
        text_data = tf.data.Dataset.from_tensor_slices(text_data)
        txt_vec = tfkl.TextVectorization(
            max_tokens=vocab_size, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=True,
            standardize="lower_and_strip_punctuation",
        )
        text_data = txt_vec.adapt(
            map(lambda x, y: txt_vec(text_data)), batch_size=8, steps=None
        )
        text_data.cache().prefetch(buffer_size=AUTOTUNE)
        return text_data        

    def train_val_test(self,):
        x_train = None
        x_val = None
        x_test = None

        for k in range(5):
            print("....")





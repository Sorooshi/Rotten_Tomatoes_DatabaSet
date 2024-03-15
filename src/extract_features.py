import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 
from copy import deepcopy


tfk = tf.keras 
tfkl = tf.keras.layers


class LstmAe(tfk.Model):
    def __init__(self, latent_dim, text_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y = None
        self.max_seq_len = 100
        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.mae_metric = tfk.metrics.MeanAbsoluteError(name="mae")
        self.mse_metric = tfk.metrics.MeanSquaredError(name="mse")
        self.loss_fn = tfk.losses.MeanSquaredError(name="mse")

        self.inputs = tfkl.InputLayer(
            input_shape=(1,), dtype=tf.string,
            )
        self.txt_vec = tfkl.TextVectorization(
            max_tokens=None, 
            split="whitespace", ngrams=1, 
            output_mode="int", ragged=False,
            output_sequence_length=self.max_seq_len,
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
            units=self.max_seq_len, activation="tanh"
            )

    def call(self, inputs, ):
        x = self.inputs(inputs)
        x = self.txt_vec(x)
        self.y = deepcopy(x)
        x = self.emb(x)
        x = self.enc(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.outputs(x)
        return x
    
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(self.y, y_pred) # tfk.losses.mean_squared_error(self.y, y_pred)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(self.y, y_pred)
        self.mse_metric.update_state(self.y, y_pred)
        return {
            "loss": self.loss_tracker.result(), 
            "mae": self.mae_metric.result(), 
            "mse": self.mse_metric.result(),
            }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric, self.mse_metric]

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





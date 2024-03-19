import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 
from sklearn.model_selection import train_test_split


tfk = tf.keras 
tfkl = tf.keras.layers

class LstmAe(tfk.Model):
    def __init__(self, latent_dim: int = 50, 
                 vocabulary: list = [],
                 classification: bool = True, 
                 max_seq_len: int = 100, *args, **kwargs):
        super(LstmAe, self).__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        if classification:
            self.train_metric = tfk.metrics.Accuracy(name="acc")
            self.val_metric = tfk.metrics.Accuracy(name="acc_val")
            self.loss_fn = tfk.losses.SparseCategoricalCrossentropy(name="scc")
            pred_activation = "softmax"
        else:
            self.train_metric = tfk.metrics.MeanRelativeError(name="mre")
            self.val_metric = tfk.metrics.MeanRelativeError(name="mre_val")
            self.loss_fn = tfk.losses.MeanRelativeError(name="loss_fn")
            pred_activation = "tanh"

        self.inputs = tfkl.InputLayer(
            input_shape=(1,), dtype=tf.string,
            )
        self.txt_vec = tfkl.TextVectorization(
            max_tokens=None, 
            vocabulary = vocabulary,
            split="whitespace", ngrams=2, 
            output_mode="int", ragged=False,
            output_sequence_length=self.max_seq_len,
            standardize="lower_and_strip_punctuation",
            )
        self.emb = tfkl.Embedding(
            input_dim=self.txt_vec.vocabulary_size(),
            output_dim=latent_dim,
            )
        self.enc1 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=150,  
                activation="relu",  
                dropout=0.1,
                return_sequences=True,
                name="encoder1"
                )
            )       
        self.dec1 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=50,  
                activation="relu", 
                dropout=0.1,
                return_sequences=True,
                name="decoder1"
            )
        )
        self.dec2 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=100,  
                activation="tanh", 
                dropout=0.1,
                return_sequences=False,
                name="decoder2"
                )
            )
        self.outputs = tfkl.Dense(
            units=self.max_seq_len, activation=pred_activation,
            )

    def call(self, inputs, training=None):
        x = self.inputs(inputs, training)
        x = self.txt_vec(x)
        x = self.emb(x)
        x = self.enc1(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.outputs(x)
        return x 
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if y is None:
                y_true = self.inputs(self.txt_vec(x))
            else:
                y_true = self.inputs(self.txt_vec(y))
            loss_value = self.loss_fn(y_true, y_pred)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_metric.update_state(y_true, y_pred)
        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        y_pred = self(x, training=False)
        if y is None:
            y_true = self.inputs(self.txt_vec(x))
        else:
            y_true = self.inputs(self.txt_vec(y))
        self.val_metric(y_true, y_pred)

    def fit(self, train_data, test_data, n_epochs):
        train_total_loss, val_total_loss = [], []
        for epoch in range(n_epochs):
            print(f"epoch: {epoch+1}")
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                if step % 50 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, loss_value)
                    )
            train_metric = self.train_metric.result()
            train_total_loss.append(train_metric)
            print("Training metric over epoch: %.3f" % (float(train_metric),))
            self.train_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in test_data:
                self.test_step(x_batch_val, y_batch_val)

            val_metric = self.val_metric.result()
            val_total_loss.append(val_metric)
            self.val_metric.reset_states()
            print("Validation metric: %.3f" % (float(val_metric),))

        return train_total_loss, val_total_loss

class TrainTestLstmAe:
    def __init__(self, data: pd.DataFrame=None, n_epochs: int= 1):
        super().__init__()
        self.data = data
        self.n_epochs = n_epochs
    
    @staticmethod
    def get_vocabulary_and_max_len(
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

        vocabulary = txt_vec.get_vocabulary()

        return vocabulary, max_seq_len

    def get_train_test_data(
        data_path: str="../data/medium_movies_data.csv", 
        batch_size=8,
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

        x_train, x_test, _, _ = train_test_split(
            text_data, labels, test_size=0.05
            )

        train_data = tf.data.Dataset.from_tensor_slices((x_train, x_train))
        train_data = train_data.shuffle(buffer_size=1024).batch(batch_size=batch_size)
        test_data = tf.data.Dataset.from_tensor_slices((x_test, x_test))
        test_data = test_data.shuffle(buffer_size=1024).batch(batch_size=batch_size)

        return train_data, test_data

    def train_val_test(self,):
        vectorized_text, labels, max_len = self.get_preprocess_data(
            data_path="../data/medium_movies_data.scv",
        )
        
        for k in range(5):
            print(" to be completed ....")


class FineTuneLstmAe:
    def __init__(self, latent_dim: int = 50, 
                 vocabulary: list = [],
                 classification: bool = True, 
                 max_seq_len: int = 100, *args, **kwargs):
        super(FineTuneLstmAe).__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len
        self.classification = classification
        self.latent_dim = latent_dim

    def model_builder(self, hp):
        hp_units = hp.Int('units', min_value=10, max_value=512, step=10)
        hp_latent_dim = hp.Int('units', min_value=10, max_value=50, step=5)
        hp_activation = hp.Categorical('activation', )
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])


        model = tfk.Sequential()
        model.add(
            tfkl.InputLayer(input_shape=(1,), dtype=tf.string,)
        )
        model.add(
            tfkl.TextVectorization(
            max_tokens=None, 
            vocabulary = self.vocabulary,
            split="whitespace", ngrams=2, 
            output_mode="int", ragged=False,
            output_sequence_length=self.max_seq_len,
            standardize="lower_and_strip_punctuation",
            )
        )
        
        model.add(
            tfkl.Embedding(
            input_dim=self.txt_vec.vocabulary_size(),
            output_dim=hp_latent_dim,
            )
        )

        model.add(
            tfkl.Bidirectional(
                tfkl.LSTM(
                units=hp_units,  
                activation="relu",  
                dropout=0.1,
                return_sequences=True,
                name="encoder1"
                ))
        )

        model.add(

        )
    







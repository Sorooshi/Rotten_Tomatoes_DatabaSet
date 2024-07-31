import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 
from sklearn.model_selection import train_test_split


tfk = tf.keras 
tfkl = tf.keras.layers

class LstmAe(tfk.Model):
    def __init__(self, latent_dim: int = 50, 
                 ngrams : int = 2,
                 vocabulary: list = None,
                 classification: bool = True, 
                 max_seq_len: int = 100, *args, **kwargs):
        super(LstmAe, self).__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        if classification:
            self.train_metric = tfk.metrics.Accuracy(name="acc")
            self.val_metric = tfk.metrics.Accuracy(name="acc_val")
            self.loss_fn = tfk.losses.SparseCategoricalCrossentropy(name="loss_fn")
            pred_activation = "softmax"
        else:
            self.train_metric = tfk.metrics.LogCoshError()
            self.val_metric = tfk.metrics.LogCoshError()
            self.loss_fn = tfk.losses.Huber(name="loss_fn")  
            pred_activation = "tanh"

        self.inputs = tfkl.InputLayer(
            input_shape=(1,), dtype=tf.string,
            )
        self.txt_vec = tfkl.TextVectorization(
            max_tokens=None, 
            vocabulary = vocabulary,
            split="whitespace", ngrams=ngrams, 
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
            # y_true = self.inputs(self.txt_vec(x))
            loss_value = self.loss_fn(y, y_pred)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_metric.update_state(y, y_pred)
        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        y_pred = self(x, training=False)
        self.val_metric(y, y_pred)

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

class TrainTestLstmAe(LstmAe):
    def __init__(self, n_epochs: int= 1, *args, **kwargs):
        super(TrainTestLstmAe, self).__init__(*args, **kwargs)
        self.data = None
        self.labels = None 
        self.text_data = None
        self.vocabulary = None
        self.max_seq_len = None
        self.n_epochs = n_epochs
    
    def get_text_and_labels(
            self, data_path: 
            str="../data/medium_movies_data.csv", ):

        self.data = pd.read_csv(data_path)
        self.labels = self.data.Genre.values
        self.text_data = self.data.Synopsis.values

        print(
            f"text data head: \n {self.text_data[:3]} \n" 
            f"text data shape: {self.text_data.shape} \n"
            f"labels head: \n {self.labels[:3]} \n"
            f"labels shape: {self.labels.shape} \n"
        ) 
 
    def get_vocabulary(
            self, vocab_path = "../data/", 
            max_seq_len: int = 123,
            np_name = "medium.npz", 
            ngrams : int = 2, 
            ) -> tuple:
        """ returns, as attributes, the vocabulary (np.arr), its size (int),
        the maximum sequence length (int) and applied ngrams (int). """

        self.get_text_and_labels()

        if not os.path.isfile(os.path.join(vocab_path, np_name)): 
            txt_vec = tfkl.TextVectorization(
                max_tokens=None, 
                vocabulary = None,
                output_sequence_length=None,  # max_seq_len
                split="whitespace", ngrams=ngrams, 
                output_mode="int", ragged=False,
                standardize="lower_and_strip_punctuation",
                )
            txt_vec.adapt(
                data=self.text_data, batch_size=8, steps=None
                )
            self.vocabulary = txt_vec.get_vocabulary()
            self.vocab_size = txt_vec.vocabulary_size()
            self.max_seq_len = max_seq_len
            self.ngrams = ngrams
            
            np.savez(os.path.join(
                vocab_path, np_name), 
                max_seq_len = self.max_seq_len,
                vocabulary=self.vocabulary, 
                vocab_size=self.vocab_size,
                ngrams = self.ngrams,
                )
            
            # vocabulary = []
            # max_seq_len = 0
            # for synopsis in self.text_data:
            #     parsed_synopsis = np.unique(
            #         synopsis.lower().strip().split(" ")
            #         ).tolist()
            #     if len(parsed_synopsis) > max_seq_len:
            #         max_seq_len = len(parsed_synopsis)
            #     for word in parsed_synopsis:
            #         if word not in vocabulary:
            #             vocabulary.append(vocabulary)            
            # vocab_size = len(vocabulary)
            # n_classes = [i.lower() for i in np.unique(self.labels)]
            # print(
            #     f"vocabulary size {len(vocabulary)}"
            #     f"Number of classes: {n_classes}"
            #     )
            
            # np.savez(os.path.join(
            #     vocab_path, np_name), 
            #     vocabulary=self.vocabulary, 
            #     vocab_size=vocab_size,
            #     max_len_seq=max_seq_len,
            #     n_classes=n_classes,
            #     )
        else:
            data_npz = np.load(
                os.path.join(vocab_path, np_name)
                )
            self.vocabulary = data_npz["vocabulary"]
            self.max_seq_len = int(data_npz["max_seq_len"])
            self.vocab_size = int(data_npz["vocab_size"])
            self.ngrams = int(data_npz["ngrams"])
            
        # txt_vec = tfkl.TextVectorization(
        #     max_tokens=vocab_size, 
        #     vocabulary=vocabulary,
        #     split="whitespace", ngrams=ngrams, 
        #     output_mode="int", ragged=True,
        #     standardize="lower_and_strip_punctuation",
        # )
        # # txt_vec.adapt(
        #     data=self.text_data, batch_size=8, steps=None
        # )
        return self.vocabulary, self.vocab_size, self.max_seq_len, self.ngrams
    

    def get_train_test_data(self, batch_size=8, return_tensors=True) -> tuple:

        vocab, _, max_seq_len, ngrams = self.get_vocabulary()

        x_train, x_test, _, _ = train_test_split(
            self.text_data, self.labels, test_size=0.05
            )
        
        lstm_ae = LstmAe(
            vocabulary=vocab, 
            classification=False, 
            max_seq_len=max_seq_len, 
            ngrams=ngrams
            )
        lstm_ae.compile()
        lstm_ae.inputs(self.text_data)
        lstm_ae.txt_vec(self.text_data)
        y_train = lstm_ae.predict(x_train)
        y_test = lstm_ae.predict(x_test)
        print(
            f"x_train and y_train shapes: {x_train.shape, y_train.shape}"
            f"x_test and y_test shapes: {x_test.shape, y_test.shape}"
            )

        if return_tensors:
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.shuffle(buffer_size=1024).batch(batch_size=batch_size)
            test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_data = test_data.shuffle(buffer_size=1024).batch(batch_size=batch_size)
            return train_data, test_data, 
        else:
            return x_train, y_train, x_test, y_test
            


    def train_test_tuned_model(self,):
        vectorized_text, labels, max_len = self.get_preprocess_data(
            data_path="../data/medium_movies_data.scv",
        )
        
        for k in range(5):
            print(" to be completed ....")

class FineTuneLstmAe(TrainTestLstmAe):
    def __init__(self,  
                 classification: bool = True, 
                 *args, **kwargs):
        super(TrainTestLstmAe, self).__init__(*args, **kwargs)        
        self.classification = classification
        
        self.vocabulary, self.max_seq_len, \
        self.vocab_size, self.ngrams = TrainTestLstmAe().get_vocabulary()

        if self.classification:
            self.pred_activation = "softmax"
            self.loss_fn = tfk.losses.SparseCategoricalCrossentropy(name="loss_fn")
            self.metric = ["accuracy"]
            self.proj_name = "LSTM_AE-Cls"
            self.dir_path = "./"
        else:
            self.pred_activation = "tanh"
            self.loss_fn = tfk.losses.Huber(name="loss_fn")
            self.metric = ["logcosh"]
            self.proj_name = "LSTM_AE-Reg"
            self.dir_path = "./"

    def build(self, hp):
        hp_units = hp.Int(
            'units', min_value=32, max_value=256, step=32
            )
        hp_latent_dim = hp.Int(
            'units', min_value=10, max_value=50, step=5
            )
        hp_activation = hp.Choice(
            'activation', values = ["relu", "tanh", ] 
            )
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6]
            )
        hp_dropout = hp.Choice(
            'dropout', values=[0.0, 0.1, 0.4]
            )

        model = tfk.Sequential()
        model.add(
            tfkl.InputLayer(input_shape=(1,),
                            dtype=tf.string,)
        )
        model.add(
            tfkl.TextVectorization(
            max_tokens=None, 
            vocabulary = self.vocabulary,
            split="whitespace", ngrams=self.ngrams, 
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
                activation=hp_activation,  
                dropout=hp_dropout,
                return_sequences=True,
                name="encoder1"
                )
            )
        )
        model.add(
            tfkl.Bidirectional(
            tfkl.LSTM(
                units=hp_units,  
                activation=hp_activation, 
                dropout=hp_dropout,
                return_sequences=True,
                name="decoder1")
                )
        )
        model.add(
            tfkl.Bidirectional(
            tfkl.LSTM(
                units=hp_units,  
                activation=hp_activation, 
                dropout=hp_dropout,
                return_sequences=True,
                name="decoder2")
                )
        )
        model.add(
            tfkl.Dense(
            units=hp_units, 
            activation=hp_activation,
            )
        )
        model.add(
            tfkl.Dense(
            units=self.max_seq_len, activation=self.pred_activation,
            )
        )

        model.compile(
            loss=self.loss_fn,
            optimizer=tfk.optimizers.SGD(learning_rate=hp_learning_rate),
            metrics=self.metric,
        )

        return model

    
    def fine_tune_the_model(self, return_tensors=False):

        
        hp = kt.HyperParameters()
        model = self.build(hp)
        
        tuner = kt.BayesianOptimization(
            hypermodel=model, 
            objective="val_accuracy", 
            max_trials=10, 
            executions_per_trial=5,
            overwrite=True,
            directory=self.dir_path,
            project_name=self.proj_name,
            )

        print(tuner.search_space_summary())

        if return_tensors:
            train_data, val_data = TrainTestLstmAe().get_train_test_data(
                batch_size=8, return_tensors=True
                )
            tuner.search(
            train_data, epochs=10, validation_data=val_data
            )
        else:
            x_train, y_train, x_test, \
                y_test = TrainTestLstmAe().get_train_test_data(return_tensors=False)
            
            tuner.search(
            x_train, y_train, epochs=10, validation_data=(x_test, y_test)
            )
        
        # models = tuner.get_best_models(num_models=1)
        best_hps = tuner.get_best_hyperparameters(2)
        
        with open(os.path.join("./best_hps" + self.proj_name), 'r') as fp:
            fp.pickle(best_hps)

        return best_hps

class TuneApplyLstmAe():
    def __init__(self, 
                 n_epochs: int= 1, 
                 ngrams : int = 1, 
                 max_seq_len: int = 12, 
                 vocab_np_name: str = "medium.npz", 
                 data_path: str = "../data",
                 classification: bool = False, 
                 data_name: str = "medium_movie_data", 
                 verbose: int = 1,
                 *args, **kwargs):
        
        super(TuneApplyLstmAe, self).__init__(*args, **kwargs)
        self.labels = None 
        self.lstm_ae = None
        self.data_df = None
        self.text_data = None
        self.ngrams =  ngrams 
        self.vocabulary = None
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.data_path = data_path
        self.data_name = data_name
        self.max_seq_len = max_seq_len
        self.vocab_np_name = vocab_np_name
        self.classification = classification
        
        if self.classification:
            assert classification is not True, "there are some fundamental theoretical issues to be address."
            self.pred_activation = "softmax"
            self.loss_fn = tfk.losses.SparseCategoricalCrossentropy(
                name="loss_fn", reduction="sum_over_batch_size"
            )
            self.metric = ["accuracy"]
            self.proj_name = "LSTM_AE-Cls"
            self.dir_path = "./"
        else:
            self.pred_activation = "tanh"
            self.loss_fn = tfk.losses.Huber(
                name="loss_fn", reduction="sum_over_batch_size",
            )
            self.metric = ["logcosh"]
            self.proj_name = "LSTM_AE-Reg"
            self.dir_path = "./"
        
    
    def get_text_and_labels(self,):
        """ returns, as attributes, data_df, synopsis (np.arr), and labels (np.arr)"""
        self.data_df = pd.read_csv(os.path.join(
            self.data_path, self.data_name + ".csv"
            )
        )
        self.labels = self.data_df.Genre.values
        self.text_data = self.data_df.Synopsis.values
        
        if verbose >= 4:
            print(
                f"text data head: \n {self.text_data[:3]} \n" 
                f"text data shape: {self.text_data.shape} \n"
                f"labels head: \n {self.labels[:3]} \n"
                f"labels shape: {self.labels.shape} \n"
            ) 
 

    def get_vocabulary(self,) -> tuple:
        """ returns, as attributes, the vocabulary (np.arr), its size (int),
        the maximum sequence length (int) and applied ngrams (int). """

        self.get_text_and_labels(data_path=self.data_path)
        
        if not os.path.isfile(os.path.join(self.vocab_path, self.vocab_np_name)): 
            txt_vec = tfkl.TextVectorization(
                max_tokens=None, 
                vocabulary = None,
                output_sequence_length=None,  # max_seq_len
                split="whitespace", ngrams=self.ngrams, 
                output_mode="int", ragged=False,
                standardize="lower_and_strip_punctuation",
                )
            txt_vec.adapt(
                data=self.text_data, batch_size=1, steps=None
                )
            self.vocabulary = txt_vec.get_vocabulary()
            self.vocab_size = txt_vec.vocabulary_size()
            self.max_seq_len = self.max_seq_len
            self.ngrams = self.ngrams
            
            np.savez(os.path.join(
                self.vocab_path, self.vocab_np_name), 
                max_seq_len = self.max_seq_len,
                vocabulary=self.vocabulary, 
                vocab_size=self.vocab_size,
                ngrams = self.ngrams,
                )

        else:
            data_npz = np.load(
                os.path.join(self.vocab_path, self.vocab_np_name)
                )
            self.vocabulary = data_npz["vocabulary"]
            self.max_seq_len = int(data_npz["max_seq_len"])
            self.vocab_size = int(data_npz["vocab_size"])
            self.ngrams = int(data_npz["ngrams"])

        return self.vocabulary, self.vocab_size, self.max_seq_len, self.ngrams
    

    def get_train_test_data(self, batch_size=8, return_tensors=True) -> tuple:

        vocab, vocab_size, max_seq_len, ngrams = self.get_vocabulary()
        x_train, x_test, _, _ = train_test_split(
            self.text_data, self.labels, test_size=0.05
            )
        
        self.lstm_ae = LstmAe(
            vocabulary=vocab, 
            classification=False, 
            max_seq_len=max_seq_len, 
            ngrams=ngrams
            )
        self.lstm_ae.compile()
        self.lstm_ae.inputs(self.text_data)
        self.lstm_ae.txt_vec(self.text_data)
        y_train = self.lstm_ae.predict(x_train)
        y_test = self.lstm_ae.predict(x_test)

        if verbose >= 3:
            print(
                f"x_train and y_train shapes: {x_train.shape, y_train.shape}"
                f"x_test and y_test shapes: {x_test.shape, y_test.shape}"
                )

        if return_tensors:
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_data = train_data.shuffle(buffer_size=len(x_train)).batch(batch_size=batch_size)
            test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_data = test_data.shuffle(buffer_size=len(x_test)).batch(batch_size=batch_size)
            return train_data, test_data, 
        else:
            return x_train, y_train, x_test, y_test


    def build(self, hp):
        hp_units = hp.Int(
            'units', min_value=32, max_value=256, step=32
            )
        hp_latent_dim = hp.Int(
            'units', min_value=10, max_value=50, step=5
            )
        hp_activation = hp.Choice(
            'activation', values = ["relu", "tanh", ] 
            )
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6]
            )
        hp_dropout = hp.Choice(
            'dropout', values=[0.0, 0.1, 0.4]
            )

        model = tfk.Sequential()
        model.add(
            tfkl.Input(shape=(1,),
                       dtype=tf.string,)
        )
        model.add(
            tfkl.TextVectorization(
            max_tokens=None, 
            vocabulary = self.vocabulary,
            split="whitespace", ngrams=self.ngrams, 
            output_mode="int", ragged=False,
            output_sequence_length=self.max_seq_len,
            standardize="lower_and_strip_punctuation",
            )
        ) 
        model.add(
            tfkl.Embedding(
            input_dim=self.lstm_ae.txt_vec.vocabulary_size(),
            output_dim=hp_latent_dim,
            )
        )
        model.add( 
            tfkl.Bidirectional(
                tfkl.LSTM(
                units=hp_units,  
                activation=hp_activation,  
                dropout=hp_dropout,
                return_sequences=True,
                name="encoder1"
                )
            )
        )
        model.add(
            tfkl.Bidirectional(
            tfkl.LSTM(
                units=hp_units,  
                activation=hp_activation, 
                dropout=hp_dropout,
                return_sequences=True,
                name="decoder1")
                )
        )
        model.add(
            tfkl.Bidirectional(
            tfkl.LSTM(
                units=hp_units,  
                activation=hp_activation, 
                dropout=hp_dropout,
                return_sequences=True,
                name="decoder2")
                )
        )
        model.add(
            tfkl.Dense(
            units=hp_units, 
            activation=hp_activation,
            )
        )
        model.add(
            tfkl.Dense(
            units=self.max_seq_len, activation=self.pred_activation,
            )
        )

        model.compile(
            loss=self.loss_fn,
            optimizer=tfk.optimizers.SGD(learning_rate=hp_learning_rate),
            metrics=self.metric,
        )

        return model


    def fine_tune_the_model(self, return_tensors=False):
        
        hps = kt.HyperParameters()
        model = self.build(hp=hps)
        
        tuner = kt.BayesianOptimization(
            hypermodel=model, 
            objective="val_accuracy", 
            max_trials=10, 
            executions_per_trial=5,
            overwrite=True,
            directory=self.dir_path,
            project_name=self.proj_name,
            )

        print(tuner.search_space_summary())

        if return_tensors:
            train_data, val_data = self.get_train_test_data(
                batch_size=8, return_tensors=True
                )
            tuner.search(
            train_data, epochs=10, validation_data=val_data
            )
        else:
            x_train, y_train, x_test, \
                y_test = self.get_train_test_data(return_tensors=False)
            
            tuner.search(
            x_train, y_train, epochs=10, validation_data=(x_test, y_test)
            )
        
        # models = tuner.get_best_models(num_models=1)
        best_hps = tuner.get_best_hyperparameters(2)
        
        with open(os.path.join("./best_hps" + self.proj_name), 'r') as fp:
            fp.pickle(best_hps)

        return best_hps


    def grid_search_model_hps(self, ):
        return_tensors = True
        results = {}
        learning_rate = [1e-5, 1e-6]
        epochs = [100, 1000, 10000]
        latent_dim = [10, 50, ]
        ngrams = [1, 2, ]
        max_sequence_length = [50, 100, 175,]

        configs = itertools.product(
            learning_rate, epochs, 
            latent_dim, ngrams,
            max_sequence_length
        )
        
        for config in configs:

            results[config] = {}
            learning_rate = config[0]
            n_epochs = config[1]
            latent_dim = config[2]
            ngrams = config[3]
            max_seq_len = config[4]

            print(
                f"configuration {config} being applied"
            )

            if latent_dim <= max_seq_len:

                vocab, _, max_seq_len, ngrams = self.get_vocabulary()
                if return_tensors is False:
                    x_train, y_train, x_test, y_test = self.get_train_test_data(
                        return_tensors=False
                    )
                else:
                    train_data, val_data = self.get_train_test_data(
                        return_tensors=True
                    )

                mdl = LstmAe(
                    latent_dim=latent_dim, ngrams=ngrams, 
                    classification=False, vocabulary=vocab, 
                    max_seq_len=max_seq_len, 
                )

                optimizer = tfk.optimizers.SGD(learning_rate=learning_rate)
                mdl.compile(optimizer=optimizer)

                train_loss, val_loss = mdl.fit(
                    train_data=train_data, test_data=val_data, n_epochs=n_epochs
                    )
                
                results[config]["train_loss"] = train_loss
                results[config]["val_loss"] = val_loss
                results[config]["config"] = config

            else:
                print(
                    f"Not a reasonable config"
                )

        return results


    def train_test_tuned_model(self, configs):

        for k in range(1, 2):  

            print(
                f"configuration {configs} in {k} fold being applied"
            )

            learning_rate = configs[0]
            n_epochs = configs[1]
            latent_dim = configs[2]
            ngrams = configs[3]
            max_seq_len = configs[4]

            if latent_dim <= max_seq_len:

                vocab, _, max_seq_len, ngrams = self.get_vocabulary()
                train_data, val_data = self.get_train_test_data(
                    return_tensors=True
                )

                mdl = LstmAe(
                    latent_dim=latent_dim, ngrams=ngrams, 
                    classification=False, vocabulary=vocab, 
                    max_seq_len=max_seq_len, 
                )

                optimizer = tfk.optimizers.SGD(learning_rate=learning_rate)
                mdl.compile(optimizer=optimizer)

                train_loss, val_loss = mdl.fit(
                    train_data=train_data, test_data=val_data, n_epochs=n_epochs
                    )

                data_df = self.data_df

                with open("./data/medium_data_no_link_movies.pickle", "r") as fp:
                    no_link_movies = pickle.load(fp)

                

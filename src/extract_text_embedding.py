import os
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt 
from sklearn.model_selection import train_test_split

verbose = 1
tfk = tf.keras 
tfkl = tf.keras.layers

parser = argparse.ArgumentParser(
    description="Extract DF Features"
    )

parser.add_argument(
        "-tt", "--to_tune", default=0, type=int,
        help="To fine-tune the model (1), or to "
        "extract features with the fine-tuned hps (0)."
        )

parser.add_argument(
        "-dn", "--data_name", default="medium", type=str,
        help="The data set name, either, medium or large."
        )


class LstmAe(tfk.Model):
    def __init__(self, latent_dim: int = 50, 
                 ngrams : int = 2,
                 vocabulary: list = None,
                 classification: bool = True, 
                 max_seq_len: int = 100, 
                 *args, **kwargs):
        super(LstmAe, self).__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        if classification:
            assert classification is not True, "there are some fundamental theoretical issues to be address."
            self.train_metric = tfk.metrics.Accuracy(name="acc")
            self.val_metric = tfk.metrics.Accuracy(name="acc_val")
            self.loss_fn = tf.losses.SparseCategoricalCrossentropy(
                name="loss_fn", reduction='sum_over_batch_size',
            )
            pred_activation = "softmax"
        else:
            self.train_metric = tfk.metrics.MeanAbsoluteError()
            self.val_metric = tfk.metrics.MeanAbsoluteError()  
            self.loss_fn = tfk.losses.Huber(  # LogCosh
                 name="loss_fn", 
            )  
            pred_activation = "tanh"

        assert vocabulary is not None, "you should pass a valid vocabulary!"

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
                units=latent_dim,  
                activation="relu",  
                dropout=0.1,
                return_sequences=True,
                name="encoder1"
                )
            )     
        self.enc2 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=int(latent_dim/2),  
                activation="relu",  
                dropout=0.1,
                return_sequences=True,
                name="encoder1"
                )
            )       
        self.dec1 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=int(latent_dim/2),  
                activation="relu", 
                dropout=0.1,
                return_sequences=True,
                name="decoder1"
            )
        )
        self.dec2 = tfkl.Bidirectional(
            tfkl.LSTM(
                units=latent_dim,  
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
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.outputs(x)
        return x 
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)
            y_true = self.inputs(self.txt_vec(x))
            loss_value = self.loss_fn(y_true, y_pred)
            train_metric = self.train_metric(y_true, y_pred)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return loss_value, train_metric
    
    @tf.function
    def test_step(self, x, y):
        y_pred = self(x, training=False)
        y_true = self.inputs(self.txt_vec(x))
        return self.val_metric(y_true, y_pred)
        

    def fit(self, train_data, test_data, n_epochs):
        train_total_loss, val_total_loss = [], []
        for epoch in range(n_epochs):
            print(f"epoch: {epoch+1}")
            tmp_train_metric, tmp_val_metric = [], []
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                # print(x_batch_train.shape, y_batch_train.shape)
                loss_value, train_metric = self.train_step(x=x_batch_train, y=y_batch_train)
                tmp_train_metric.append(train_metric) 
                if step % 200 == 0:
                    print(
                        "Training loss and metric (for one batch) at step %d: %.3f, %.3f"
                        % (step, loss_value, train_metric)
                    )
            tmp_train_metric = np.asarray(tmp_train_metric)
            train_total_loss.append(tmp_train_metric.mean())
            self.train_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in test_data:
                val_metric = self.test_step(x_batch_val, y_batch_val)
                tmp_val_metric.append(val_metric)
            tmp_val_metric = np.asarray(tmp_val_metric)
            val_total_loss.append(tmp_val_metric.mean())
            self.val_metric.reset_states()
            print("Validation metric: %.3f" % tmp_val_metric.mean(),)

        return train_total_loss, val_total_loss


class GetConvertedData():
    def __init__(self, 
                 ngrams : int = 1, 
                 max_seq_len: int = 12, 
                 vocab_np_name: str = "medium.npz", 
                 data_path: str = "./data",
                 data_name: str = "medium_movies_data", 
                 verbose: int = 1,
                 *args, **kwargs):
        
        super(GetConvertedData, self).__init__(*args, **kwargs)
        self.labels = None 
        self.data_df = None
        self.text_data = None
        self.ngrams =  ngrams 
        self.vocabulary = None
        self.verbose = verbose
        self.data_path = data_path
        self.data_name = data_name
        self.max_seq_len = max_seq_len
        self.vocab_np_name = vocab_np_name
    
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
        
        return self.data_df, self.text_data, self.labels
 

    def get_vocabulary(self,) -> tuple:
        """ returns, as attributes, the vocabulary (np.arr), its size (int),
        the maximum sequence length (int) and applied ngrams (int). """

        _, _, _ = self.get_text_and_labels()
        
        # check whether the vocabulary exists (in npz format):
        if not os.path.isfile(os.path.join(self.data_path, self.vocab_np_name)): 
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
            
            # saving vocab as npy
            np.savez(os.path.join(
                self.data_path, self.vocab_np_name), 
                max_seq_len = self.max_seq_len,
                vocabulary=self.vocabulary, 
                vocab_size=self.vocab_size,
                ngrams = self.ngrams,
                )

        else:
            # loading the saved vocab as npy
            data_npz = np.load(
                os.path.join(self.data_path, self.vocab_np_name)
                )
            self.vocabulary = data_npz["vocabulary"]
            self.max_seq_len = int(data_npz["max_seq_len"])
            self.vocab_size = int(data_npz["vocab_size"])
            self.ngrams = int(data_npz["ngrams"])

        return self.vocabulary, self.vocab_size, self.max_seq_len, self.ngrams
    

    def get_train_test_data(self, batch_size=2, return_tensors=True) -> tuple:

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


class TuneApplyLstmAe():
    def __init__(self, 
                 n_epochs: int= 1, 
                 ngrams : int = 1, 
                 max_seq_len: int = 12, 
                 vocab_np_name: str = "medium.npz", 
                 data_path: str = ".,/data",
                 classification: bool = False, 
                 data_name: str = "medium_movies_data", 
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
        
    def grid_search_model_hps(self, ):

        return_tensors = True
        results = {}
        learning_rate = [1e-5, 1e-6]
        epochs = [10, 100, 2000]
        latent_dim = [5, 10, 50]
        ngrams = [1, 2, ]
        max_sequence_length = [10, 100, 175]

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

            data_getter  = GetConvertedData(
                 ngrams=ngrams, 
                 max_seq_len=max_seq_len, 
                 vocab_np_name=self.vocab_np_name, 
                 data_path= self.data_path,
                 data_name=self.data_name, 
                 verbose=1,
            )

            print(
                f"configuration {config} being applied"
            )

            if latent_dim <= max_seq_len:

                vocab, _, max_seq_len, ngrams = data_getter.get_vocabulary()

                if return_tensors is False:
                    x_train, y_train, x_test, y_test = data_getter.get_train_test_data(
                        return_tensors=False
                    )
                else:
                    train_data, val_data = data_getter.get_train_test_data(
                        return_tensors=True, batch_size=2,
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

                with open("./tmp_results/LSTM-AE_" + str(config) +".pickle", "wb") as fp:
                    pickle.dump(results, fp)

            else:
                print(
                    f"Not a reasonable config"
                )

        return results


    def train_and_extract_features_tuned_model(self, configs):

        for k in range(1, 2):  

            print(
                f"configuration {configs} in {k} fold being applied"
            )

            learning_rate = configs[0]
            n_epochs = configs[1]
            latent_dim = configs[2]
            ngrams = configs[3]
            max_seq_len = configs[4]

            data_getter  = GetConvertedData(
                 ngrams=ngrams, 
                 max_seq_len=max_seq_len, 
                 vocab_np_name=self.vocab_np_name, 
                 data_path= self.data_path,
                 data_name=self.data_name, 
                 verbose=1,
            )


            vocab, _, max_seq_len, ngrams = data_getter.get_vocabulary()
            train_data, val_data = data_getter.get_train_test_data(
                return_tensors=True, batch_size=2,
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

            data_df, _, _ = data_getter.get_text_and_labels()

            with open("./data/medium_data_no_link_movies.pickle", "rb") as fp:
                no_link_movies = pickle.load(fp)


            data_df = data_df.loc[~data_df.Title.isin(no_link_movies)]

            text_data = data_df.Synopsis.values

            y_preds = mdl.predict(text_data)
            x = mdl.inputs(text_data)
            x = mdl.txt_vec(x)
            embeddings = mdl.emb(x)
            print(
                f"embedding {embeddings.shape}"
            )

            # embeddings is of the size N * max_seq_len * latent_dim
            # to convert each synopsis embedding into a feature vector,
            # we compute the average the embedding of each synopsis text over 
            # the max_seq_len size, i.e., embeddings[i, :].mean(axis=0)

            embedding_features = embeddings.numpy().mean(axis=1)
            
            data_df = data_df.join(
                pd.DataFrame(
                    data=embedding_features, index=data_df.index,
                    columns=["Embedding-"+ str(f) for f in range(embedding_features.shape[1])]
                )
            )
            features = ["Runtime", "Box Office (Gross USA)", "Tomato Meter", "Audience Score", 
            "No. Reviews", "Genre"
            ]
            features += ["Embedding-"+ str(f) for f in range(embedding_features.shape[1])]

            data_df_x = data_df[features]

            data_df_x.to_csv("./data/medium_data_df_x.csv", index=True, columns=features)
            data_df_x.to_csv("./data/medium_data_x.csv", header=False, index=False)
            

                 
if __name__ == "__main__":
    
    args = parser.parse_args()
    to_tune = args.to_tune
    d_name = args.data_name
    

    if d_name == "medium":
        data_name = "medium_movies_data"
        vocab_np_name = "medium.npz"
    elif d_name == "large":
        data_name = "large_movies_data"
        vocab_np_name = "large.npz"

    tuner_applier = TuneApplyLstmAe(
        data_path="./data",
        data_name=data_name,  
        vocab_np_name=vocab_np_name
        )
    
    if to_tune == 1:

        results = tuner_applier.grid_search_model_hps()

        with open("./LSTM-AE_configs" + d_name +".pickle", "wb") as fp:
            pickle.dump(results, fp)
    else:
        # best config
        # lr, epochs, latent_dim, ngrams, max_seq_len
        config = (1e-5, 2, 5, 1, 10)
        tuner_applier.train_and_extract_features_tuned_model(configs=config)

    
   
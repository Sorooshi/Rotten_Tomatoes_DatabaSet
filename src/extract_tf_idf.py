import os
import re
import sys
import nltk
import pickle
import string
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from extract_text_embedding import GetConvertedData
from sklearn.feature_extraction.text import TfidfVectorizer



class ExtractTfIdf():
    def __init__(self, 
                 corpus: np.array, 
                 max_features: int,
                 data_df: pd.DataFrame, 
                 ngrams_rng: tuple = (1, 2), 
                 *args, **kwargs):
        super(*args, **kwargs).__init__()
        self.corpus = corpus
        self.max_features = max_features
        self.ngrams_rng = ngrams_rng
        self.wnl = WordNetLemmatizer()
        self.data_df = data_df
        # self.documents = list()
        # self.vocabulary = dict()
        # self.tf_idf = pd.DataFrame
        
        user_defined_stopwords = ["st","rd","hong","kong", "...", ] 
        a = nltk.corpus.stopwords.words('english')
        b = list(string.punctuation) + user_defined_stopwords
        self.stopwords = set(a).union(b)

    def preprocess(self, corpus, get_vocab):
        """returns a list of sets, where each set is lowercased, stripped, 
        splitted, and lemmatized version of the original the synopsis text."""

        noises = ["$", ",", ".", ":", ";", "(", ")", "()" ]
        documents = []
        for document in corpus:
            document = document.lower().strip().split()
            document = [re.sub(r"\[a-z0-9]+", "", word)  for word in document]
            for noise in noises:
                document = [word.replace(noise, "")  for word in document]
            document = [word for word in document if word not in self.stopwords]
            document = [self.wnl.lemmatize(word) for word in document]
            if get_vocab is False:
                document = " ".join(document)
            
            documents.append(document)

        return documents
    

    def get_vocabulary(self, docs):
        """returns a dict where keys are word and values are the integer indices."""
       
        cntr = 0
        vocabulary = {}
        for doc in docs:
            for word in doc:
                if word not in vocabulary.keys():
                    vocabulary[word] = cntr
                    cntr += 1

        return vocabulary
    
    def get_tf_idf(self, docs, vocab):

        vectorizer = TfidfVectorizer(
            input="content", 
            tokenizer=None, 
            strip_accents="ascii", 
            decode_error="ignore",
            stop_words=None,
            lowercase=False,
            max_df=1.0,
            min_df=1,
            ngram_range=self.ngrams_rng,
            max_features=self.max_features, 
            sublinear_tf=False, 
            smooth_idf=True,
            vocabulary=vocab,
        )

        x = vectorizer.fit_transform(docs)
        print(x.toarray().shape)

        tf_idf = pd.DataFrame(
            x.toarray(), index=self.data_df.Title, 
            columns=vectorizer.get_feature_names()
            )
        
        return tf_idf
    

    def get_feature_data(self, data_name):
        
        vocab = self.get_vocabulary(
            docs=self.preprocess(corpus=self.corpus, get_vocab=True)
            )
        # print(vocab, len(vocab))
        docs = self.preprocess(corpus=self.corpus, get_vocab=False)
        # print(docs, len(docs))
        tf_idf = self.get_tf_idf(docs=docs, vocab=None)
        # print(tf_idf, len(tf_idf))

        if data_name == "medium_movies_data":
                features = ["Runtime", "Box Office (Gross USA)", "Tomato Meter", "Audience Score", 
                "No. Reviews", "Genre"
                ]
        elif data_name == "large_movies_data":
                features = ["Runtime", "Tomato Meter", "Audience Score", 
                "No. Reviews", "Genre"
                ]
        
        data_x_df = self.data_df[features]
        data_x_df = data_x_df.join(tf_idf)

        return data_x_df
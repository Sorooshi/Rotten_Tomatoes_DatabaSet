import os
import re
import sys
import nltk
import pickle
import string
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from extract_text_embedding import GetConvertedData
from sklearn.feature_extraction.text import TfidfVectorizer


parser = argparse.ArgumentParser(
    description="Extract TF-IDF"
    )

parser.add_argument(
        "-dn", "--data_name", default="medium", type=str,
        help="The data set name, either, medium or large."
    )

parser.add_argument(
    "-mf", "--max_features", default=None, type=int, 
    help="maximum number of features to construct the vocabulary"
    )


class ExtractTfIdf():
    def __init__(self, 
                corpus: np.array, 
                max_features: int,
                data_df: pd.DataFrame, 
                ngrams_rng: tuple = (2, 2), 
                data_name: str = "medium_movies_data",
                *args, **kwargs):
        super(*args, **kwargs).__init__()
        self.corpus = corpus
        self.max_features = max_features
        self.ngrams_rng = ngrams_rng
        self.wnl = WordNetLemmatizer()
        self.data_df = data_df
        self.data_name = data_name
        
        user_defined_stopwords = [
            '00', '100', '10000', '100th', '118th', '11th', '120', '1300',
            '13th', '1429', '14th', '150', '1500', '1549', '155', '1558',
            '1590s', '1621', '1770', '1774', '1788', '1804', '1805', '181st',
            '1820', '1820s', '1823', '1830', '1836', '1840s', '1843', '1847',
            '1850', '1851', '1857', '1863', '1865', '1869', '1870s', '1875',
            '1878', '1879', '1880s', '1890s', '190', '1900s', '1909', '1912',
            '1913', '1914', '1917', '1928', '1932', '1937', '1938', '1943',
            '1947', '1953', '1956', '1957', '1959', '1966', '1986', '1989',
            '1991', '1992', '1997', '2000s', '2011', '2012', '2016', '2018',
            '2020', '2028', '2030s', '2035', '2045', '2054', '2077', '2084',
            '20s', '2154', '21st', '23rd', '250', '2505', '26th', '27', '300',
            '3000', '330000', '3po', '42', '48', '480', '4k', '500000', '54',
            '54th', '60th', '64', '65th', '66', '6th', '700', '72', '72nd',
            '79', '80s', '8th', '90s', '95'
            ] 
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
            max_df=1,
            min_df=1,
            ngram_range=self.ngrams_rng,
            max_features=self.max_features, 
            sublinear_tf=False, 
            smooth_idf=True,
            vocabulary=vocab,
        )

        x = vectorizer.fit_transform(docs)

        tf_idf = pd.DataFrame(
            x.toarray(), index=self.data_df.index, 
            columns=vectorizer.get_feature_names()
        )
        
        return tf_idf
    

    def get_feature_data(self, ):
        
        _ = self.get_vocabulary(
            docs=self.preprocess(corpus=self.corpus, get_vocab=True)
            )
        docs = self.preprocess(corpus=self.corpus, get_vocab=False)
        tf_idf = self.get_tf_idf(docs=docs, vocab=None)

        if self.data_name == "medium_movies_data":
            features = [
                "Title", "Runtime", "Box Office (Gross USA)", "Tomato Meter", 
                "Audience Score", "No. Reviews", "Genre"
            ]

            with open("./data/medium_data_no_link_movies.pickle", "rb") as fp:
                no_link_movies = pickle.load(fp)

        elif self.data_name == "large_movies_data":
            features = [
                "Title", "Runtime", "Tomato Meter", 
                "Audience Score", "No. Reviews", "Genre",
            ]
            with open("./data/large_data_no_link_movies.pickle", "rb") as fp:
                no_link_movies = pickle.load(fp)

        
        data_df = self.data_df.loc[~self.data_df.Title.isin(no_link_movies)]
        
        data_df_x = data_df[features]
        labels = data_df_x["Genre"].astype('category').cat.codes
        data_df_x = data_df_x.join(tf_idf)

        if self.data_name == "medium_movies_data":
            data_df_x.to_csv(
                "./data/medium_data_tfidf_df_x.csv",
                index=False, columns=data_df_x.columns
            )
            data_df_x.drop(columns=["Genre", "Title"], inplace=True)
            # the two following CSV files will be used for clustering directly
            data_df_x.to_csv(
                "./data/medium_data_tfidf_x.csv", 
                header=False, index=False
            )
            labels.to_csv(
                    "./data/medium_labels.csv", 
                    header=False, index=False
            )         

        elif self.data_name == "large_movies_data":
            data_df_x.to_csv(
                "./data/large_data_tfidf_df_x.csv",
                index=False, columns=data_df_x.columns
            )
            data_df_x.drop(columns=["Genre", "Title"], inplace=True)
            # the two following CSV files will be used for clustering directly
            data_df_x.to_csv(
                "./data/large_data_tfidf_x.csv", 
                header=False, index=False
            )
            labels.to_csv(
                    "./data/large_labels.csv", 
                    header=False, index=False
            )         


        return data_df_x
    

if __name__ == "__main__":

    args = parser.parse_args()
    d_name = args.data_name
    max_features = args.max_features 


    if d_name == "medium":
        data_name = "medium_movies_data"
        vocab_np_name = "medium.npz"
    elif d_name == "large":
        data_name = "large_movies_data"
        vocab_np_name = "large.npz"
    
    data_getter = GetConvertedData(
        vocab_np_name=vocab_np_name, 
        data_name=data_name,
    )     
    
    data_df, text_data, _ = data_getter.get_text_and_labels()

    tfidf_getter = ExtractTfIdf(
        data_df=data_df, 
        corpus=text_data, 
        max_features=max_features, 
        ngrams_rng=(1, 1),

    )

    df = tfidf_getter.get_feature_data()
    
    print(
        # f"df.head: {df.head()} \n"
        # f"df.describe: {df.describe()} \n"
        f"df.shape: {df.shape} \n"
    )



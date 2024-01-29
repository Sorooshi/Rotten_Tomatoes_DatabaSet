import pickle
from bs4 import BeautifulSoup
import pandas as pd 



def get_pickled_urls_all_genres(path):
    with open (path + ".pickle", "rb") as fp:
        genres_urls = pickle.load(fp)
    return genres_urls


row_data = pd.DataFrame(
    columns=[
        "Name", "Synopsis" , "Link", "Rating", "Language", "Director", "Producer", 
        "Writer", "Runtime", "Distributor", "Production Co", "CAST & CREW", "Genre 1", 
        "Genre 2", "Genre 3", "Ground Truth",
        ]
    )



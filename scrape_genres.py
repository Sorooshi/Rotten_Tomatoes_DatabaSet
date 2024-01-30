import os
import pickle
import requests 
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions

GENRES = [
    "action", "adventure", "animation", "anime", "biography", "comedy", "crime", 
    "documentary", "drama", "fantasy", "lgbtq", "history", "holiday", "horror", 
    "kids_and_family", "music", "musical", "mystery_and_thriller", "romance", 
    "sci_fi", "sports", "stand_up", "war", "western"
]

def get_pickled_urls_all_genres(path: str) -> pickle:
    with open (path + ".pickle", "rb") as fp:
        genres_urls = pickle.load(fp)
    return genres_urls


def get_data_per_url(ref_url: str) -> callable:
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
    driver.get(ref_url)
    # by_class = driver.find_elements(By.CLASS_NAME, "blah blah")
    synopsis = driver.find_element("xpath", '//*[@data-qa="movie-info-synopsis"]')
    movie_info = driver.find_elements("xpath", '//*[@data-qa="movie-info-item-value"]')
    top_casts = driver.find_elements("xpath", '//*[@data-qa="cast-crew-item-link"]')
    
    return synopsis, movie_info, top_casts
    

def get_data_per_url_detailed(ref_url: str) -> callable:
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
    driver.get(ref_url)
    # by_class = driver.find_elements(By.CLASS_NAME, "blah blah")
    synopsis = driver.find_element("xpath", '//*[@data-qa="movie-info-synopsis"]')
    top_casts = driver.find_elements("xpath", '//*[@data-qa="cast-crew-item-link"]')


    
    return synopsis, movie_info, top_casts

if __name__ == "__main__":

    # create an initial data frame to store the extracted information
    movie_data_df = pd.DataFrame(
        index=np.arange(int(len(GENRES)*150)), 
        columns=[
            "Title", "Synopsis", "Rating", "Genre", "Original Language", "Director", "Producer", "Writer", 
            "Release Date (Theaters)", "Box Office (Gross USA)", "Runtime", "Distributor", "Production Co", 
            "Sound Mix", "Top six Cast", "Link", "Initial Genre", 
            ]
        )
    
    all_genres_urls = get_pickled_urls_all_genres(path="urls_per_genres")
    issues = list()
    idx = 0
    for k, v in all_genres_urls.items():
        print(f" Genre: {k}")
        for kk, vv in v.items():
            tmp = kk.title().replace("_", " ")
            print(
                f"Title: {tmp}, "
                f"Link: {vv}"
            )
            synopsis, movie_info, top_casts = get_data_per_url(ref_url=vv)
            movie_data_df.loc[idx, "Title"] = kk.title().replace("_", " ")
            movie_data_df.loc[idx, "Link"] = vv
            movie_data_df.loc[idx, "Initial Genre"] = k
            movie_data_df.loc[idx, "Synopsis"] = synopsis.text
            if len(movie_info) < 13:
                print(f"issues in {vv}")
                issues.append(vv)
            
            # insert info from rating to sound mix
            for j in range(len(movie_info)):
                movie_data_df.iloc[idx, j+2] = movie_info[j].text
            # insert info of the top 6 casts as a list
            movie_data_df.iloc[idx, j+2+12] = top_casts.text
            idx += 1

            




    


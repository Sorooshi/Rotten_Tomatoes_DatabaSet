import os
import pickle
import requests 
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
from selenium import webdriver
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
    synopsis = driver.find_element('xpath', '//*[@data-qa="movie-info-synopsis"]')
    top_casts = driver.find_element('xpath', '//*[@id="cast-and-crew"]')
    movie_info = driver.find_element('xpath', '//*[@id="info"]')

    return synopsis, movie_info, top_casts


class RTScraper:
    """ scrapes the required contents of the given url. """

    def __init__(self, url_to_scrape: str) -> None:
        self.url_to_scrape = url_to_scrape
        self.opts = FirefoxOptions()
        self.opts.add_argument("--headless")
        self.driver = webdriver.Firefox(options=self.opts)
        self.driver.get(self.url_to_scrape)
       

    def get_synopsis(self, ) -> str:
        return self.driver.find_element('xpath', '//*[@data-qa="movie-info-synopsis"]').text
    
    def get_movie_info(self) -> dict:
        movie_info = self.driver.find_element('xpath', '//*[@id="info"]').text.split("\n")
        movie_info = {i.split(":")[0]: i.split(":")[1] for i in movie_info}
        return movie_info

    def get_top_casts(self, ):
        top_casts = self.driver.find_element('xpath', '//*[@id="cast-and-crew"]').text.split("\n")[1:-1]
        return top_casts

    def get_all_required_info(self, ):

        synopsis = self.get_synopsis()
        movie_info = self.get_movie_info()
        top_casts = self.get_top_casts()

        return synopsis, movie_info, top_casts
    

# def init_get_driver_per_url(ref_url: str):
#     """ Returns selenium webdriver and get the contents of the ref_url. """
#     opts = FirefoxOptions()
#     opts.add_argument("--headless")
#     driver = webdriver.Firefox(options=opts)
#     return driver.get(ref_url)

# def get_synopsis(driver ) -> str:
#     synopsis = driver.find_element('xpath', '//*[@data-qa="movie-info-synopsis"]').text
#     return synopsis

# def get_movie_info(driver) -> dict:
#     movie_info = driver.find_element('xpath', '//*[@id="info"]').text.split("\n")
#     movie_info = {i.split(":")[0]: i.split(":")[1] for i in movie_info}
#     return movie_info

# def get_top_casts(driver,):
#     top_casts = driver.find_element('xpath', '//*[@id="cast-and-crew"]').text.split("\n")[1:-1]
#     return top_casts


if __name__ == "__main__":

    # create an initial data frame to store the extracted information
    movie_data_df = pd.DataFrame(
        index=np.arange(int(len(GENRES)*150)), 
        columns=[
            "Title", "Synopsis", "Rating", "Genre", "Original Language", "Director", "Producer", "Writer", 
            "Release Date (Theaters)", "Release Date (Streaming)", "Box Office (Gross USA)", "Runtime", "Distributor", "Production Co", 
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
            driver = init_get_driver_per_url(ref_url=vv)

            movie_data_df.loc[idx, "Link"] = vv
            movie_data_df.loc[idx, "Initial Genre"] = k
            movie_data_df.loc[idx, "Title"] = kk.title().replace("_", " ")
            movie_data_df.loc[idx, "Synopsis"] = get_synopsis(driver=driver)

            movie_data_df.loc[idx, "Top Six Cast"] =get_top_casts
            if len(movie_info) < 13:
                print(f"issues in {vv}")
                issues.append(vv)
            
            # insert info from rating to sound mix
            for j in range(len(movie_info)):
                movie_data_df.iloc[idx, j+2] = movie_info[j].text
            idx += 1

            




    


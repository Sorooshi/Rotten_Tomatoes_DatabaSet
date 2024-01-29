import os
import pickle
import requests 
import pandas as pd 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions



def get_pickled_urls_all_genres(path):
    with open (path + ".pickle", "rb") as fp:
        genres_urls = pickle.load(fp)
    return genres_urls


def get_data_per_url(ref_url):
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(option=opts)
    driver.get(ref_url)
    # by_class = driver.find_elements(By.CLASS_NAME, "")
    # synopsis = driver.find_element("xpath", '//*[@data-qa="movie-info-synopsis"]').text
    # this will extract: synopsis, .... (sorted in the extraction order)
    movie_info = driver.find_elements("xpath", '//*[@data-qa="movie-info-synopsis"]')
    


    return None 


row_data = pd.DataFrame(
    columns=[
        "Name", "Synopsis" , "Link", "Rating", "Language", "Director", "Producer", 
        "Writer", "Runtime", "Distributor", "Production Co", "CAST & CREW", "Genre 1", 
        "Genre 2", "Genre 3", "Ground Truth",
        ]
    )



import pickle
import numpy as np
import pandas as pd 
from tqdm import trange
from selenium import webdriver
from selenium.webdriver import FirefoxOptions


def get_pickled_urls_all_genres(path: str) -> pickle:
    with open (path + ".pickle", "rb") as fp:
        genres_urls = pickle.load(fp)
    return genres_urls

def get_all_collected_urls(path):
    with open(path + ".txt", "r") as fp:
        all_urls = fp.readlines()

    return all_urls


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
        self.driver.quit()
    
        return synopsis, movie_info, top_casts
    

if __name__ == "__main__":

    # create an initial data frame to store the extracted information
    movie_data_df = pd.DataFrame(
        # index=np.arange(int(len(GENRES)*150)), 
        columns=[
            "Title", "Synopsis", "Rating", "Genre", "Original Language", "Director", "Producer", "Writer", 
            "Release Date (Theaters)", "Release Date (Streaming)", "Box Office (Gross USA)", "Runtime", "Distributor", "Production Co", 
            "Sound Mix", "Top Cast", "Aspect Ratio", "View the collection", "Link", 
            ]
        )
    
    all_urls = get_all_collected_urls(path="collected_urls")
    issues = list()
    idx = 0
    for u in trange(len(all_urls)):
        url = all_urls[u]
        tmp_title = url.split("m/")[-1].title().replace("_", " ")
        print(
            f"Title: {tmp_title}, "
            f"Link: {url}"
            )
        rts = RTScraper(url)
        synopsis, movie_info, top_casts = rts.get_all_required_info()
        try:
            movie_data_df.loc[idx, "Link"] = url
            movie_data_df.loc[idx, "Title"] = tmp_title
            movie_data_df.loc[idx, "Synopsis"] = synopsis
            movie_data_df.loc[idx, "Top Cast"] = top_casts

            for kk, vv in movie_info.items():
                movie_data_df.loc[idx, kk] = vv
        except:
            print(f" There is an issue in {url}")
            issues.append(url)

        idx += 1

    movie_data_df.to_csv("rotten_tomatoes_info.csv")
    with open ("issues.txt", "w") as fp:
        for issue in issues:
            fp.write(f"{issue}\n")




    


import json
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
        top_cast = self.driver.find_element('xpath', '//*[@id="cast-and-crew"]').text.split("\n")[1:-1]
        return top_cast

    def get_all_required_info(self, ):

        synopsis = self.get_synopsis()
        movie_info = self.get_movie_info()
        top_casts = self.get_top_casts()
        self.driver.quit()
    
        return synopsis, movie_info, top_casts
    

if __name__ == "__main__":
     
    to_json = True
    # all_urls = get_all_collected_urls(path="collected_urls")
    all_urls = [
        "https://www.rottentomatoes.com/m/godzilla_king_of_the_monsters_2019",
        "https://www.rottentomatoes.com/m/to_catch_a_killer_2023",
        "https://www.rottentomatoes.com/m/star_wars_episode_i_the_phantom_menace",
        "https://www.rottentomatoes.com/m/f9",
        "https://www.rottentomatoes.com/m/black_adam",
    ]

    issues = list()

    if to_json == True:
        movies_data = dict()
        for u in trange(len(all_urls)):
            url = all_urls[u]
            title_ = url.split("m/")[-1].title().replace("_", " ")
            # Construct an inner dict to store the movie info
            movies_data[url] = {}
            movies_data[url]["Title"] = title_
            try:
                rts = RTScraper(url)
                synopsis, movie_info, top_cast = rts.get_all_required_info()
                movies_data[url]["Synopsis"] = synopsis
                movies_data[url]["Info"] = movie_info
                movies_data[url]["Top Cast"] = top_cast
            except:
                print(f"There was an issue in {url}")
                issues.append(url)
        
        with open("rotten_tomatoes_movies_data.json", "w") as fp:
            json.dump(movies_data, fp)

    else:

        # create an initial data frame to store the extracted information
        movies_data = pd.DataFrame(
            # index=np.arange(int(len(GENRES)*150)), 
            columns=[
                "Title", "Synopsis", "Rating", "Genre", "Original Language", "Director", "Producer", "Writer", 
                "Release Date (Theaters)", "Rerelease Date (Theaters)", "Release Date (Streaming)", "Rerelease Date (Streaming)", 
                "Box Office (Gross USA)", "Runtime", "Distributor", "Production Co", 
                "Sound Mix", "Top Cast", "Aspect Ratio", "View the collection", "Link", 
                ]
            )

        all_urls = get_all_collected_urls(path="collected_urls")
        issues = list()
        for u in trange(len(all_urls)):
            url = all_urls[u]
            title_ = url.split("m/")[-1].title().replace("_", " ")
            try:
                rts = RTScraper(url)
                synopsis, movie_info, top_cast = rts.get_all_required_info()
                idx = 0
        
            except:
                print(f"There was an issue in {url}")
                issues.append(url)

            movies_data.loc[idx, "Link"] = url
            movies_data.loc[idx, "Title"] = title_
            movies_data.loc[idx, "Synopsis"] = synopsis
            movies_data.loc[idx, "Top Cast"] = top_cast
            for kk, vv in movie_info.items():
                movies_data.loc[idx, kk] = vv
            idx += 1

        movies_data.to_csv("rotten_tomatoes_movies_data.csv")


    with open ("issues.txt", "w") as fp:
        for issue in issues:
            fp.write(f"{issue}\n")




    


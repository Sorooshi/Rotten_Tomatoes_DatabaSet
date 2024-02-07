import json
import pickle
import pandas as pd 
from tqdm import trange
from selenium import webdriver
from selenium.webdriver import FirefoxOptions


def get_movies_data(path: str) -> json:
    with open (path + ".json", "rb") as fp:
        movies_data = json.load(fp)
    return movies_data

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
       
    def get_score_panel(self, ):
        score_panel = self.driver.find_element('xpath', '//*[@data-qa="score-panel"]').text.split("\n")
        return score_panel
    

if __name__ == "__main__":
     
    to_json = True
    movies_data = get_pickled_data(path="rotten_tomatoes_movies_data.json")
    issues = list()

    if to_json == True:
        for k, v in movies_data.items():
            url = k
            title_ = url.split("m/")[-1].title().replace("_", " ")
            # Construct an inner dict to store the movie info
            try:
                rts = RTScraper(url)
                score_panel = rts.get_score_panel()
                movies_data[url]["Score Panel"] = score_panel
            except Exception as error:
                print(
                    f"In {url} \n"
                    f"{error} \n"
                    f"occurred !"
                    )
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
        idx = 0
        for u in trange(len(all_urls)):
            url = all_urls[u]
            title_ = url.split("m/")[-1].title().replace("_", " ")
            try:
                rts = RTScraper(url)
                synopsis, movie_info, top_cast = rts.get_all_required_info()
                movies_data.loc[idx, "Link"] = url
                movies_data.loc[idx, "Title"] = title_
                movies_data.loc[idx, "Synopsis"] = synopsis
                movies_data.loc[idx, "Top Cast"] = top_cast
                for kk, vv in movie_info.items():
                    movies_data.loc[idx, kk] = vv
                idx += 1

            except Exception as error:
                print(
                    f"In {url} \n"
                    f"{error} \n"
                    f"occurred !"
                    )
                issues.append(url)

           
        movies_data.to_csv("rotten_tomatoes_movies_data.csv")


    with open ("issues.txt", "w") as fp:
        for issue in issues:
            fp.write(f"{issue}\n")




    


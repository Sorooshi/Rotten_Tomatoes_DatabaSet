import json
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
        self.driver.quit()
        return score_panel
    

if __name__ == "__main__":

    movies_data = get_movies_data(path="rotten_tomatoes_movies_data")
    issues = list()

    for url, v in movies_data.items():
        title_ = url.split("m/")[-1].title().replace("_", " ")
        print(f"Adding score panel of {title_} from {url}")
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
    
    with open("rotten_tomatoes_movies_data_with_score_panels.json", "w") as fp:
        json.dump(movies_data, fp)

    with open ("issues_score_panels.txt", "w") as fp:
        for issue in issues:
            fp.write(f"{issue}\n")




    


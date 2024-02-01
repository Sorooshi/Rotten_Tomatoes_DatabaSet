import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions


BASE_URL = "https://www.rottentomatoes.com/browse/movies_at_home/"

GENRES = [
    "action", "adventure", "animation", "anime", "biography", "comedy", "crime", 
    "documentary", "drama", "fantasy", "lgbtq", "history", "holiday", "horror", 
    "kids_and_family", "music", "musical", "mystery_and_thriller", "romance", 
    "sci_fi", "sports", "stand_up", "war", "western"
]

CRITICS = [
    "rotten", "fresh", "certified_fresh",
]


""" three sample links of three genres."""
"https://www.rottentomatoes.com/browse/movies_at_home/critics:rotten~genres:action~sort:popular"
"https://www.rottentomatoes.com/browse/movies_at_home/critics:fresh~genres:action~sort:popular"
"https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:action"


def collect_all_urls(max_page_range: int = 5) -> list:
    urls = list()        

    for genre in GENRES:
        for critic in CRITICS:
            
            base_url = BASE_URL + "critics:" + critic + "~genres:" + genre

            print(
                f"The main url to collect movies from {genre} and {critic} is {base_url}"
                )
            
            opts = FirefoxOptions()
            opts.add_argument("--headless")
            driver = webdriver.Firefox(options=opts)
            ref_url = base_url + "?page=" + str(max_page_range)
            driver.get(ref_url)
            all_links = driver.find_elements(By.TAG_NAME, "a")
            for link in all_links:
                url = link.get_attribute("href")  # "text", etc.
                if isinstance(url, str):
                    if url not in urls and "m" in url.split("/"):  # movies are separated by a "m"
                        urls.append(url)
            driver.quit()

    return urls


if __name__ == "__main__":
    
    urls = collect_all_urls(max_page_range=5)
    print(
        f"In total {len(urls)} has been collected" 
        )

    with open("collected_urls.txt", "w") as fp:
        for url in urls:
            fp.write(f"{url}\n")
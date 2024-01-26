import os
import pickle
import requests 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

# os.system("sudo safaridriver --enable")

""" 
"ENTERTAINMENT",  "FAITH & SPIRITUALITY", "HEALTH & WELLNESS", 
"HOUSE & GARDEN", "NATURE", "NEWS", "REALITY", "SOAP", "SPECIAL INTEREST", 
"TALK SHOW", "TRAVEL", "VARIETY",   were empty
"SHORT" has one movie
"""

GENRES = [
    "action", "adventure", "animation", "anime", "biography", "comedy", "crime", 
    "documentary", "drama", "fantasy", "lgbtq", "history", "holiday", "horror", 
    "kids_and_family", "music", "musical", "mystery_and_thriller", "romance", 
    "sci_fi", "sports", "stand_up", "war", "western"
]

URLS_TO_SCRAPE = [
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:action~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:adventure~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:animation~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:anime~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:biography~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:comedy~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:crime~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:documentary~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:drama~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:fantasy~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:lgbtq~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:holiday~sort:popular"
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:horror~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:kids_and_family~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:music~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:musical~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:mystery_and_thriller~sort:popular",
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:romance~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:sci_fi~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:sports~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:stand_up~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:war~sort:popular", 
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~genres:western~sort:popular", 
]


def get_urls_2(base_url: str, max_page_range: int) -> list:

    urls, contents = list(), list()

    for page in range(1, max_page_range, 1):
        ref_url = base_url + "?page=" + str(page)
        print("ref_url:\n", ref_url)
        requested_urls = requests.get(ref_url)
        soup = BeautifulSoup(requested_urls.text, "html.parser")
        for url in soup.find_all("a"):
            link = url.get("href")
            # print("link\n", link)
            content = url.get("")
            if link not in urls:
                urls.append(link)
                contents.append(content)
    return urls, contents


def get_urls_per_genre(base_url: str, max_page_range: int) -> list:
    
    urls = list()        
    driver = webdriver.Firefox()
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



def get_urls(urls_to_scrape: list, max_page_range: int = 5) -> dict:
    urls = {}
    for i in range(len(urls_to_scrape)):
        print(f"Genres: {GENRES[i]}")
        urls[GENRES[i]] = {}
        urls_per_genre = get_urls_per_genre(
            base_url=urls_to_scrape[i], max_page_range=max_page_range
            )
        for url in range(len(urls_per_genre)):
            name = urls_per_genre[url].split("/")[-1]
            urls[GENRES[i]][name] = urls_per_genre[url]

    return urls

urls_per_genres = get_urls(urls_to_scrape=URLS_TO_SCRAPE, max_page_range=5)

print(len(urls_per_genres))

# with open("urls.txt", "w") as fp:
#     for url in urls:
#         fp.write(f"{url}\n")

with open ("urls_per_genres.pickle", "wb") as fp: 
    pickle.dump(urls_per_genres, fp)



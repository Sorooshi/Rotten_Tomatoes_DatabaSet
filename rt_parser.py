import pickle
import requests 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


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


def get_urls(base_url: str, max_page_range: int) -> list:
    
    urls = list()

    for page in range(1, max_page_range):
        driver = webdriver.Safari()
        ref_url = base_url + "?page=" + str(page)
        print("page:", page, ref_url)        
        driver.get(ref_url)
        all_links = driver.find_elements(By.TAG_NAME, "a")
        for link in all_links:
            url = link.get_attribute("href")  # "text", etc.
            if url not in urls:
                urls.append(url)
    driver.quit()
    
    return urls

url = "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~sort:a_z"

urls = get_urls(base_url=url, max_page_range=20)

# urls, _ = get_urls_2(base_url=url, max_page_range=15)

# print(urls)

print(len(urls))


with open("urls.txt", "w") as fp:
    for url in urls:
        fp.write(f"{url}\n")

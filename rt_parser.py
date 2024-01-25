import requests 
from bs4 import BeautifulSoup


def get_urls(url: str, page_range: int) -> list:
    urls = []
    for page in range(page_range):
        requested_urls = requests.get(url + str(page))
        soup = BeautifulSoup(requested_urls, "html.parser")
        for j in soup.find_all("a"):
            link = j.get("href")
            if not link in urls:
                print(link)
                urls.append(link)
    return urls

urls = get_urls("https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh?page=", 3)
print(urls)
print(len(urls))








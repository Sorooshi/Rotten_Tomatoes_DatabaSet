import requests 
from bs4 import BeautifulSoup


def get_urls(ref_url: str, max_page_range: int) -> list:
    ref_url = ref_url + str(max_page_range)
    requested_urls = requests.get(ref_url)
    soup = BeautifulSoup(requested_urls.text, "html.parser")
    urls = list()
    for url in soup.find_all("a"):
        link = url.get("href")
        if link not in urls:
            urls.append(link)
    return urls

urls = get_urls("https://www.rottentomatoes.com/browse/movies_at_home/?page=", 4)
print(urls)
print(len(urls))

with open ("urls_text.txt", "w") as fp:
    fp.writelines(urls)







import requests 
from bs4 import BeautifulSoup

def get_urls(ref_url: str, max_page_range: int) -> list:
    if max_page_range:
        ref_url = ref_url + str(max_page_range)
    
    requested_urls = requests.get(ref_url)
    soup = BeautifulSoup(requested_urls.text, "html.parser")
    urls = list()
    contents = list()
    for url in soup.find_all("a"):
        link = url.get("href")
        content = url.content
        if link not in urls:
            urls.append(link)
            contents.append(content)
    return urls, contents


urls, contents = get_urls(
    "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~sort:a_z?page=9", None
    )
    # "https://www.rottentomatoes.com/browse/movies_at_home/critics:certified_fresh~sort:a_z?page=6"
print(urls)
print(len(urls))
print(len(contents))

with open ("urls.txt", "w") as fp:
    for url in urls:
        fp.write(f"{url}\n")


with open ("contents.txt", "w") as fp:
    for content in contents:
        fp.write(f"{content}\n")



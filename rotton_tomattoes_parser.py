import requests 
from bs4 import BeautifulSoup

url = "https://www.rottentomatoes.com/browse/movies_in_theaters/"
requested_urls = requests.get(url)

soup = BeautifulSoup(requested_urls.text, "html.parser")

urls = [link for link in soup.find_all("a")]

urls = []
for link in soup.find_all("a"):
    urls.append(link.get("href"))
    print(link.get("href"))






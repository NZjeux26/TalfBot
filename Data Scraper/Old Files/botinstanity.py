import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",  # sometimes helps to have a referer
    # Add additional headers if needed
}

url = "https://aagenielsen.dk/visallespil.php"
response = requests.get(url, headers=headers)

# Save to file for debugging
with open("page.html", "w", encoding=response.encoding) as f:
    f.write(response.text)

print("HTML saved to page.html")

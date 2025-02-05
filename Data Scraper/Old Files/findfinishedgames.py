import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}

# Fetch the page content
url = "https://aagenielsen.dk/visallespil.php"
response = requests.get(url, headers=headers)
html_content = response.content

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find all table rows (<tr> elements)
table_rows = soup.find_all("tr")

found = False  # Flag to indicate if any completed game was found

for index, row in enumerate(table_rows, start=1):
    # Get the text of the row using a separator to avoid words running together
    row_text = row.get_text(separator=" ", strip=True).lower()
    
    # Debug: print each row's text so you can inspect what is being processed
    print(f"DEBUG: Row {index}: {row_text}")
    
    # Filter criteria:
    #   1. Must be a Copenhagen Hnefatafl 11x11 game.
    #   2. Must not be ongoing.
    #   3. Must contain "won" (indicating a completed game).
    if ("copenhagen hnefatafl 11x11" in row_text and
        "ongoing" not in row_text and
        "won" in row_text):
        print("\nFound a completed Copenhagen Hnefatafl 11x11 game:")
        print(row_text)
        found = True

if not found:
    print("\nNo completed Copenhagen Hnefatafl 11x11 games found.")

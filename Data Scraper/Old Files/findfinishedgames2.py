from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Path to Brave browser and Chromedriver
brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
chrome_driver_path = "/opt/homebrew/bin/chromedriver"

# Configure ChromeOptions to use Brave
options = webdriver.ChromeOptions()
options.binary_location = brave_path

# Create the Chrome Service using your chromedriver path
service = Service(chrome_driver_path)

# Initialize the webdriver with the service and options
driver = webdriver.Chrome(service=service, options=options)

# Open the URL
url = "https://aagenielsen.dk/visallespil.php"
driver.get(url)

# Wait for the page to load initial content
time.sleep(5)

# (Optional) If there's a pagination button or "load more", find and click it:
# For example, if there's a button with text "Next", you might do:
# try:
#     next_button = driver.find_element(By.LINK_TEXT, "Next")
#     next_button.click()
#     time.sleep(5)  # Wait for the next page to load
# except Exception as e:
#     print("No pagination button found:", e)

# Alternatively, scroll to the bottom to trigger dynamic loading
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(5)  # Adjust wait time as necessary

# Get the full page source after dynamic content has loaded
html_content = driver.page_source
driver.quit()

# Parse the page source with BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")
table_rows = soup.find_all("tr")
print("Total <tr> elements found:", len(table_rows))

found = False
for index, row in enumerate(table_rows, start=1):
    row_text = row.get_text(separator=" ", strip=True).lower()
    # Debug output
    print(f"DEBUG: Row {index}: {row_text}")
    
    # Filtering criteria: must be a Copenhagen Hnefatafl 11x11 game,
    # must not be ongoing, and must indicate completion with "won"
    if ("copenhagen hnefatafl 11x11" in row_text and
        "ongoing" not in row_text and
        "won" in row_text):
        print("\nFound a completed Copenhagen Hnefatafl 11x11 game:")
        print(row_text)
        found = True

if not found:
    print("\nNo completed Copenhagen Hnefatafl 11x11 games found.")
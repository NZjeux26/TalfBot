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

try:
    # Navigate to the page
    url = "https://aagenielsen.dk/visallespil.php"
    driver.get(url)
    
    # Optionally, wait until the page has loaded (adjust the timeout as needed)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    # Wait for the "List" button to be clickable.
    # Adjust the XPath if necessary. This example looks for a <button> element that contains the text "List".
    list_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'List')]"))
    )
    
    # Click the "List" button
    list_button.click()
    
    # Wait for the new page or game data to load.
    # You might want to wait for a specific element that you know appears on the game data page.
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//table"))  # example: wait for a table element
    )
    
    # Optionally, add a short sleep to ensure the page content has fully updated.
    time.sleep(2)
    
    # Get the updated page source and parse it with BeautifulSoup
    html_source = driver.page_source
    soup = BeautifulSoup(html_source, "html.parser")
    
    # Now extract the game data.
    # This example finds all table rows; adjust the selector based on your page structure.
    game_rows = soup.find_all("tr")
    
    print("Extracted Game Data:")
    for row in game_rows:
        # You can adjust this printout or further parse the row as needed.
        row_text = row.get_text(separator=" ", strip=True)
        print(row_text)
        
finally:
    # Clean up and close the browser.
    driver.quit()

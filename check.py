from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Path to Brave browser
brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
chrome_driver_path = "/opt/homebrew/bin/chromedriver"

options = webdriver.ChromeOptions()
options.binary_location = brave_path

service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# Navigate to the games archive search page
driver.get("https://aagenielsen.dk/visallespil_soeg.php")

# Select the "Copenhagen Hnefatafl 11x11" radio button and click "Next"
copenhagen_hnefatafl_11x11 = driver.find_element(By.CSS_SELECTOR, 'input[value="16"]')
copenhagen_hnefatafl_11x11.click()

next_button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'button[type="submit"] b'))
)
next_button.click()

try:
    # Find all the match rows and check if they contain "won"
    table = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'table[cellspacing="2"]'))
    )
    match_rows = table.find_elements(By.CSS_SELECTOR, 'tr[bgcolor="#CCCCCC"]')

    for match_row in match_rows:
        match_text = match_row.text.lower()
        if "won" in match_text:
            list_button = match_row.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
            list_button.click()
            current_url = driver.current_url
            print(f"Navigated to: {current_url}")
            break
except Exception as e:
    print(f"Error occurred: {e}")
    print("Unable to find match rows.")

# 🛑 Keep the browser open for manual inspection
print("Browser will stay open for 60 seconds. Check if everything loads.")
time.sleep(60)  # Keep browser open
driver.quit()
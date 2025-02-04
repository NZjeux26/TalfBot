import time
import random
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Path to Brave and ChromeDriver
brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
chrome_driver_path = "/opt/homebrew/bin/chromedriver"

# Selenium Setup
options = Options()
options.binary_location = brave_path
options.add_argument("--headless")  # Run without opening a browser (optional)
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# Game Archive URL
archive_url = "https://aagenielsen.dk/visallespil_soeg.php"
driver.get(archive_url)
time.sleep(3)  # Allow time for the page to load

# Find the "Copenhagen Hnefatafl 11x11" radio button and click it
copenhagen_hnefatafl_11x11 = driver.find_element(By.CSS_SELECTOR, 'input[value="16"]')
copenhagen_hnefatafl_11x11.click()

# Find the "Next" button and click it
next_button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'button[type="submit"] b'))
)
next_button.click()

# Extract all game rows
games = driver.find_elements(By.XPATH, "//tr")

game_data = []

for game in games:
    try:
        # Check if the game is completed (Black Text)
        color = game.value_of_css_property("color")
        if "rgb(0, 0, 0)" not in color:  # Skip ongoing games (green text)
            continue

        # Extract Game Info
        cells = game.find_elements(By.TAG_NAME, "td")
        if len(cells) < 2:
            continue

        game_title = cells[1].text
        if "Copenhagen Hnefatafl 11x11" not in game_title:
            continue  # Only scrape 11x11 format

        print(f"Scraping: {game_title}")

        # Click "List" button to enter the game details
        list_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]:has(b:contains("List"))')
        list_button.click()
        time.sleep(random.uniform(2, 4))  # Random sleep to avoid detection

        # Extract moves and results
        move_elements = driver.find_elements(By.XPATH, "//tr")
        moves = []
        for move in move_elements:
            cols = move.find_elements(By.TAG_NAME, "td")
            if len(cols) == 2:  # Ensure correct format
                moves.append({
                    "black": cols[0].text.strip(),
                    "white": cols[1].text.strip()
                })

        # Determine winner
        result_text = driver.find_element(By.XPATH, "//b[contains(text(), 'won')]").text
        winner = "White" if "White won" in result_text else "Black"

        # Capture board state
        board = []
        rows = driver.find_elements(By.XPATH, "//tr[contains(@class, 'boardrow')]")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            board_row = []
            for cell in cells:
                class_name = cell.get_attribute("class")
                if "empty" in class_name:
                    board_row.append(0)
                elif "attacker" in class_name:
                    board_row.append(1)
                elif "defender" in class_name:
                    board_row.append(2)
                elif "king" in class_name:
                    board_row.append(3)
            board.append(board_row)

        # Store game data
        game_data.append({
            "title": game_title,
            "winner": winner,
            "moves": moves,
            "final_board": board
        })

        # Go back to the archive page
        driver.back()
        time.sleep(random.uniform(2, 4))

    except Exception as e:
        print(f"Error processing game: {e}")

# Close WebDriver
driver.quit()

# Save data to JSON
with open("hnefatafl_games.json", "w") as f:
    json.dump(game_data, f, indent=4)

# Save to CSV
df = pd.DataFrame(game_data)
df.to_csv("hnefatafl_games.csv", index=False)

print("✅ Scraping Complete! Data saved as JSON and CSV.")
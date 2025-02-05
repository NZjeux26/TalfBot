import csv
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ---------------------------
# Configuration for Brave and Chromedriver
brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
chrome_driver_path = "/opt/homebrew/bin/chromedriver"
#Run Headless
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.binary_location = brave_path

service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# ---------------------------
# Step 1. Open the games archive search page.
driver.get("https://aagenielsen.dk/visallespil_soeg.php")

# Step 2. Select the "Copenhagen Hnefatafl 11x11" radio button and click "Next"
cph_radio = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[value="16"]'))
)
cph_radio.click()

next_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"] b'))
)
next_button.click()

# ---------------------------
# Step 3. On the results page, locate completed games.
# We filter for rows that:
#   - Contain "Copenhagen Hnefatafl 11x11"
#   - Do NOT contain "ongoing"
#   - Contain "won"
#
# We'll click the "List" button in each matching row to extract game moves.

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "tr")))
time.sleep(1)

rows = driver.find_elements(By.XPATH, "//tr")
csv_data = []  # Each entry will be [game_summary, game_moves]
processed_games = 0
max_games = 100  # For testing, only process 5 games

for idx in range(len(rows)):
    if processed_games >= max_games:
        break

    # Refresh the list each iteration since the page reloads after clicking "List"
    rows = driver.find_elements(By.XPATH, "//tr")
    row = rows[idx]
    row_text = row.text.lower()

    if ("copenhagen hnefatafl 11x11" in row_text and
        "ongoing" not in row_text and
        "won" in row_text):

        game_summary = row.text.strip()
        print("Found completed game:")
        print(game_summary)

        try:
            list_button = row.find_element(By.XPATH, ".//form[contains(@action, 'visspil.php')]//button")
        except Exception as e:
            print("Could not locate the List button in this row.", e)
            continue

        list_button.click()

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        moves_page = driver.page_source
        soup_moves = BeautifulSoup(moves_page, "html.parser")
        game_moves = soup_moves.get_text(separator=" ", strip=True)
        print("Extracted moves (preview):")
        print(game_moves[:200], "...\n")

        csv_data.append([game_summary, game_moves])
        processed_games += 1

        driver.back()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//tr")))
        time.sleep(2)

# ---------------------------
# Step 4. Process and write the data to CSV.
# We want our CSV to have three columns:
#   1. Winning Colour (e.g. "white won" or "black won")
#   2. Number of moves (e.g. "32")
#   3. List of moves (starting at the first move, e.g. "1. k8-i8 ...")
processed_csv = []

for game_summary, game_moves in csv_data:
    # --- Extract winning colour from the game summary ---
    win_match = re.search(r'\b(white|black)\s+won\b', game_summary, re.IGNORECASE)
    winning_colour = win_match.group(0).strip() if win_match else ""

    # --- Extract number of moves from the game summary ---
    moves_match = re.search(r'(\d+)\s+moves', game_summary, re.IGNORECASE)
    moves_count = moves_match.group(1).strip() if moves_match else ""

    # --- Extract list of moves from game_moves ---
    # Locate the first occurrence of a digit followed by a period (e.g. "1.")
    move_start = re.search(r'\d+\.', game_moves)
    moves_list = game_moves[move_start.start():].strip() if move_start else game_moves.strip()

    ## Remove trailing content starting at "Black captured" (everything after this is ignored)
    pattern = re.compile(r'(.*?)\s*Black captured', re.IGNORECASE)
    match = pattern.search(moves_list)
    if match:
        moves_list = match.group(1).strip()

    processed_csv.append([winning_colour, moves_count, moves_list])

csv_filename = "game_moves.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Winning Colour", "Moves Count", "Moves List"])
    writer.writerows(processed_csv)

print(f"Data extraction complete. {len(processed_csv)} games written to {csv_filename}")

driver.quit()

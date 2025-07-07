from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

driver = webdriver.Safari()
driver.get("https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr")

wait = WebDriverWait(driver, 20)

columns_to_select = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED"
]

for col in columns_to_select:
    try:
        checkbox = wait.until(EC.element_to_be_clickable((By.ID, col)))
        if not checkbox.is_selected():
            checkbox.click()
            print(f"Selected {col}")
    except Exception as e:
        print(f"Failed to select {col}: {e}")

# Wait for the submit/download button to be clickable
submit_button = wait.until(
    EC.element_to_be_clickable((By.ID, "btnDownload"))
)

# Click the submit/download button
submit_button.click()
time.sleep(30)

def wait_for_download(download_dir, timeout=60):
    import time
    start = time.time()
    while True:
        files = [f for f in os.listdir(download_dir) if f.endswith(".zip") or f.endswith(".csv")]
        if files:
            print(f"Download detected: {files}")
            return files[0]
        elif time.time() - start > timeout:
            print("Download timeout.")
            return None
        time.sleep(1)

download_folder = os.path.expanduser("~/Downloads")
filename = wait_for_download(download_folder, timeout=60)
if filename:
    print(f"Download complete: {filename}")
else:
    print("Download not detected within timeout.")


print("Clicked download button, waiting for file to download...")

driver.quit()
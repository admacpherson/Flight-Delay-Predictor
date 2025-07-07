from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def download_bts_data():
    driver = webdriver.Safari()

    try:
        url = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr"
        driver.get(url)

        wait = WebDriverWait(driver, 20)

        # Wait for the submit/download button to be clickable
        submit_button = wait.until(
            EC.element_to_be_clickable((By.ID, "btnDownload"))
        )

        # Click the submit/download button
        submit_button.click()

        print("Clicked download button, waiting for file to download...")

        # Wait some seconds to allow download (adjust if needed)
        time.sleep(30)

    finally:
        driver.quit()

if __name__ == "__main__":
    download_bts_data()

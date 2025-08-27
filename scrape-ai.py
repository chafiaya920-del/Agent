import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from tenacity import retry, stop_after_attempt, wait_exponential
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.common.exceptions import TimeoutException, WebDriverException

# ---- Configuration ----
BASE_URL = "https://www.amazon.com/s?k=laptops&page={}"
MAX_PAGES = 5
MAX_RETRIES = 3

# Robust selectors for Amazon product data
SELECTORS = {
    "product_card": "div[data-component-type='s-search-result']",
    "title": "h2 a span",
    "price": "span.a-price .a-offscreen",
    "rating": "span.a-icon-alt",
    "link": "h2 a",
    "availability": "span.a-color-price",
    "prime": "i.a-icon-prime",
    "reviews_count": "span.a-size-base.s-underline-text"
}

# List of proxies (replace with your actual proxies)
PROXIES = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080",
    # Add more proxies as needed
]

def get_random_proxy():
    """Get a random proxy from the list"""
    return random.choice(PROXIES) if PROXIES else None

def setup_driver(proxy=None):
    """Setup Chrome driver with proxy support and anti-detection measures"""
    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    
    # Add random user agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
    ]
    options.add_argument(f"--user-agent={random.choice(user_agents)}")
    
    if proxy:
        prox = Proxy()
        prox.proxy_type = ProxyType.MANUAL
        prox.http_proxy = proxy
        prox.ssl_proxy = proxy
        capabilities = webdriver.DesiredCapabilities.CHROME
        prox.add_to_capabilities(capabilities)
        driver = uc.Chrome(options=options, desired_capabilities=capabilities)
    else:
        driver = uc.Chrome(options=options)
    
    return driver

def get_random_delay():
    """Get a random delay between requests"""
    return random.uniform(3, 7)

def scroll_page(driver):
    """Scroll the page randomly to simulate human behavior"""
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")
    current_position = 0
    
    while current_position < total_height:
        scroll_amount = random.randint(100, 300)
        current_position += scroll_amount
        driver.execute_script(f"window.scrollTo(0, {current_position});")
        time.sleep(random.uniform(0.1, 0.3))

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def scrape_page(driver, url):
    """Scrape a single page with retry mechanism"""
    try:
        driver.get(url)
        
        # Wait for the product cards to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["product_card"]))
        )
        
        # Scroll the page to simulate human behavior
        scroll_page(driver)
        
        # Add a small random delay
        time.sleep(random.uniform(1, 2))
        
        return driver.page_source
    except TimeoutException:
        print("Timeout waiting for page to load. Retrying...")
        raise
    except WebDriverException as e:
        print(f"WebDriver error: {e}. Retrying...")
        raise

def extract_product_data(card):
    """Extract product data from a card element"""
    try:
        # Get product title
        title_elem = card.select_one(SELECTORS["title"])
        title = title_elem.get_text(strip=True) if title_elem else "N/A"

        # Get price
        price_elem = card.select_one(SELECTORS["price"])
        price = price_elem.get_text(strip=True) if price_elem else "N/A"

        # Get product link
        link_elem = card.select_one(SELECTORS["link"])
        link = "https://www.amazon.com" + link_elem["href"] if link_elem and "href" in link_elem.attrs else "N/A"

        # Get rating
        rating_elem = card.select_one(SELECTORS["rating"])
        rating = rating_elem.get_text(strip=True) if rating_elem else "N/A"

        # Get availability
        availability_elem = card.select_one(SELECTORS["availability"])
        availability = availability_elem.get_text(strip=True) if availability_elem else "In Stock"

        # Get Prime status
        prime_elem = card.select_one(SELECTORS["prime"])
        prime = "Yes" if prime_elem else "No"

        # Get number of reviews
        reviews_elem = card.select_one(SELECTORS["reviews_count"])
        reviews_count = reviews_elem.get_text(strip=True) if reviews_elem else "0"

        return {
            "title": title,
            "price": price,
            "rating": rating,
            "link": link,
            "availability": availability,
            "prime": prime,
            "reviews_count": reviews_count
        }
    except Exception as e:
        print(f"Error extracting product data: {e}")
        return None

# ---- Main scraping logic ----
product_data = []

try:
    # Initialize driver with a random proxy
    proxy = get_random_proxy()
    driver = setup_driver(proxy)
    print(f"✅ Chrome driver initialized successfully with proxy: {proxy if proxy else 'No proxy'}")

    # ---- Scraping loop ----
    for page_num in range(1, MAX_PAGES + 1):
        url = BASE_URL.format(page_num)
        print(f"Scraping page {page_num}: {url}")
        
        try:
            # Scrape the page with retry mechanism
            page_source = scrape_page(driver, url)
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Get all product cards
            product_cards = soup.select(SELECTORS["product_card"])
            
            if not product_cards:
                print("⚠️ No products found. Amazon might be blocking our requests.")
                break

            # Extract data from each product card
            for card in product_cards:
                product_info = extract_product_data(card)
                if product_info and product_info["title"] != "N/A":
                    product_data.append(product_info)
                    print(f"Found product: {product_info['title']}")

            # Random delay between pages
            delay = get_random_delay()
            print(f"Waiting {delay:.1f} seconds before next page...")
            time.sleep(delay)

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            continue

finally:
    # Always close the driver
    try:
        driver.quit()
        print("✅ Chrome driver closed successfully")
    except:
        pass

# ---- Save to CSV ----
if product_data:
    df = pd.DataFrame(product_data)
    df.to_csv("amazon_products.csv", index=False)
    print(f"✅ Done. Saved {len(product_data)} products to amazon_products.csv")
else:
    print("❌ No products were found. Amazon might be blocking our requests.")

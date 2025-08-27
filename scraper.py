import requests
from selectorlib import Extractor
from fake_useragent import UserAgent
import json
import time
import random
import argparse
import logging
from typing import Optional

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Initialisation de l’extracteur YAML
extractor = Extractor.from_yaml_file("selectors.yml")

def get_random_headers() -> dict:
    """Génère un User-Agent aléatoire"""
    ua = UserAgent()
    return {
        "User-Agent": ua.random,
        "Accept-Language": "en-US,en;q=0.9",
    }

def scrape_amazon_product(url: str, proxy: Optional[str] = None) -> dict:
    """Scrape les données d’un produit Amazon à partir de l’URL"""
    logging.info(f"Scraping URL: {url}")
    headers = get_random_headers()

    proxies = {"http": proxy, "https": proxy} if proxy else None

    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        if "captcha" in response.text.lower():
            logging.warning("Captcha détecté dans la page")
            return {"error": "Captcha detected"}
        if response.status_code != 200:
            logging.error(f"Erreur HTTP {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}

        data = extractor.extract(response.text)
        return data

    except requests.RequestException as e:
        logging.error(f"Erreur réseau: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL du produit Amazon")
    parser.add_argument("--proxy", help="Adresse proxy HTTP (optionnel)")
    args = parser.parse_args()

    result = scrape_amazon_product(args.url, proxy=args.proxy)
    print(json.dumps(result, indent=2))

    # Random delay to avoid bot detection
    delay = random.uniform(2.0, 5.0)
    logging.info(f"Pause de {delay:.2f} secondes pour éviter la détection")
    time.sleep(delay)

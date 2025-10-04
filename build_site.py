import os
import sqlite3
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import shutil
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader

# --- Configuration --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Source Data URL and Paths (Historical DB is removed)
JSON_URL = "https://www.milkywayidle.com/game_data/marketplace.json"
JSON_PATH = "marketplace.json"
CATEGORY_FILE_PATH = "cata.txt"

# Output directory for static files
OUTPUT_DIR = "."
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")

# --- Category Parsing ---
def parse_categories(filepath):
    """Parses the cata.txt file into a dictionary {item_name: category}."""
    categories = {}
    current_main_category = "Unknown"
    current_sub_category = None
    current_display_category = "Unknown"

    known_main_categories = [
        "Currencies", "Loots", "Resources", "Consumables", "Books", "Keys",
        "Equipment", "Jewelry", "Trinket", "Tools"
    ]
    equipment_subcategories = [
        "Main Hand", "Off Hand", "Head", "Body", "Legs", "Hands",
        "Feet", "Back", "Pouch"
    ]
    tool_subcategories = [
        "Milking", "Foraging", "Woodcutting", "Cheesesmithing", "Crafting",
        "Tailoring", "Cooking", "Brewing", "Alchemy", "Enhancing"
    ]

    logging.info(f"Attempting to parse category file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line == '•': continue

                if line in known_main_categories:
                    current_main_category = line
                    current_sub_category = None
                    current_display_category = current_main_category
                    continue

                if current_main_category == "Equipment" and line in equipment_subcategories:
                    current_sub_category = line
                    current_display_category = f"Equipment / {current_sub_category}"
                    continue

                if current_main_category == "Tools" and line in tool_subcategories:
                    current_sub_category = line
                    current_display_category = f"Tools / {current_sub_category}"
                    continue

                item_names = [name.strip() for name in line.split(',') if name.strip()]
                for item_name in item_names:
                    if item_name:
                        if current_display_category == "Unknown" and current_main_category != "Unknown":
                            current_display_category = current_main_category
                        categories[item_name] = current_display_category

        logging.info(f"Parsed {len(categories)} items from category file.")
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}. Categories will be missing.")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error parsing category file {filepath}: {e}", exc_info=True)
        return {}

# --- Data Fetching ---
def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    logging.info(f"Attempting to download {url} to {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded {local_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error downloading {url}: {e}", exc_info=True)
        return False

# --- Data Loading and Processing ---
def load_live_data():
    """Loads live market data and vendor prices from the JSON file."""
    logging.info(f"Loading live data from {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        logging.warning(f"{JSON_PATH} not found.")
        return pd.DataFrame(), {}

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # CORRECTED: Properly load vendor prices from the JSON
        vendor_prices = {}
        if 'vendorPrice' in data and isinstance(data['vendorPrice'], dict):
            vendor_prices = data['vendorPrice']
            logging.info(f"Loaded {len(vendor_prices)} vendor prices.")
        else:
            logging.warning("'vendorPrice' key not found in JSON. Vendor prices will be missing.")

        if 'marketData' not in data or not isinstance(data['marketData'], dict):
            logging.error(f"Invalid JSON structure: 'marketData' key missing or not a dictionary.")
            return pd.DataFrame(), vendor_prices

        live_records = []
        market_data = data['marketData']
        current_time = datetime.now(timezone.utc)

        for product_path, tiers in market_data.items():
            if not isinstance(tiers, dict): continue
            for tier_str, prices in tiers.items():
                if not isinstance(prices, dict): continue

                # Use .get() with a default to avoid errors if keys are missing
                ask_price = prices.get('a', -1)
                buy_price = prices.get('b', -1)

                live_records.append({
                    'product': product_path,
                    'buy': pd.NA if buy_price == -1 else buy_price,
                    'ask': pd.NA if ask_price == -1 else ask_price,
                    'timestamp': current_time
                })

        if not live_records:
            logging.warning("No market records found in the JSON data.")
            return pd.DataFrame(), vendor_prices

        live_df = pd.DataFrame(live_records)
        live_df['buy'] = pd.to_numeric(live_df['buy'], errors='coerce')
        live_df['ask'] = pd.to_numeric(live_df['ask'], errors='coerce')
        live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], errors='coerce')

        logging.info(f"Loaded {len(live_df)} live market records.")
        return live_df, vendor_prices

    except Exception as e:
        logging.error(f"Unexpected error loading live data from {JSON_PATH}: {e}", exc_info=True)
        return pd.DataFrame(), {}

# --- Main Execution ---
def main():
    """Main build process."""
    logging.info("--- Starting Static Site Build ---")
    try:
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

        # --- Parse Categories ---
        item_categories = parse_categories(CATEGORY_FILE_PATH)
        if not item_categories:
            logging.warning("Category data is empty. Categories will be missing in output.")

        # --- Download and Load Live Data ---
        logging.info("--- Downloading Data ---")
        json_ok = download_file(JSON_URL, JSON_PATH)
        if not json_ok:
            logging.error("Failed to download critical live data (marketplace.json). Cannot proceed.")
            return

        logging.info("--- Loading Data ---")
        market_df, vendor_prices = load_live_data()

        if market_df.empty:
            logging.error("Live market data is empty. Cannot build site.")
            return

        # --- Filter Products ---
        all_products = sorted(list(market_df['product'].unique()))
        logging.info(f"Applying vendor price filter to {len(all_products)} products...")
        filtered_products = [
            p for p in all_products
            if p == "Bag Of 10 Cowbells" or ((vp := vendor_prices.get(p)) is not None and vp > 0)
        ]
        logging.info(f"Filtered down to {len(filtered_products)} products.")

        # --- Generate Market Summary JSON ---
        logging.info("--- Generating market_summary.json ---")
        market_summary = []
        for product_name in filtered_products:
            # Get the data for the current product
            product_data = market_df[market_df['product'] == product_name].iloc[0]
            market_summary.append({
                'name': product_name,
                'category': item_categories.get(product_name, 'Unknown'),
                'buy': product_data['buy'] if pd.notna(product_data['buy']) else None,
                'ask': product_data['ask'] if pd.notna(product_data['ask']) else None,
                'vendor': vendor_prices.get(product_name)
            })

        summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f, allow_nan=False, default=str)
        logging.info(f"Saved market summary ({len(market_summary)} items) to {summary_path}")

        # The rest of the script (generating HTML, etc.) would go here.
        # This corrected version focuses on fixing the data processing pipeline.

        logging.info("--- Static Site Build Finished Successfully ---")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the main build process: {e}", exc_info=True)

if __name__ == '__main__':
    main()

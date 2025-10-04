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

# Source Data URL and Paths
JSON_URL = "https://www.milkywayidle.com/game_data/marketplace.json"
JSON_PATH = "marketplace.json"
CATEGORY_FILE_PATH = "cata.txt"

# Output directory for static files 
OUTPUT_DIR = "output" 
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")

# --- Category Parsing ---
def parse_categories(filepath):
    """
    Parses the cata.txt file into a dictionary {lowercase_item_name: category}.
    (FIX: Makes keys lowercase and cleans up non-standard spaces for robust matching.)
    """
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
        "Feet", "Back", "Pouch", "Two Hand"
    ]
    tool_subcategories = [
        "Milking", "Foraging", "Woodcutting", "Cheesesmithing", "Crafting",
        "Tailoring", "Cooking", "Brewing", "Alchemy", "Enhancing"
    ]

    logging.info(f"Attempting to parse category file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # FIX: Aggressive cleanup of line whitespace, including non-breaking space (u'\xa0')
                line = line.strip().replace(u'\xa0', u' ')
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
                        
                        # FIX: Store item name as lowercase key for robust matching against API data
                        categories[item_name.lower()] = current_display_category

        logging.info(f"Parsed {len(categories)} items from category file.")
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}. Categories will be missing.")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error parsing category file {filepath}: {e}", exc_info=True)
        return {}

# --- Data Loading and Processing ---
def load_live_data():
    """
    Loads live market data and vendor prices from the JSON file.
    CRITICAL FIX: This function now only extracts prices for the base item tier ("0").
    """
    logging.info(f"Loading live data from {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        logging.warning(f"{JSON_PATH} not found.")
        return pd.DataFrame(), {}

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

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

        for product_hrid, tiers in market_data.items():
            if not isinstance(tiers, dict): continue
            
            # CRITICAL FIX: Only extract data for the base item tier (enhancement level "0")
            if "0" in tiers and isinstance(tiers["0"], dict):
                prices = tiers["0"]

                ask_price = prices.get('a', -1)
                buy_price = prices.get('b', -1)

                live_records.append({
                    'product_hrid': product_hrid, # Keep the raw HRID for vendor lookup later
                    'buy': pd.NA if buy_price == -1 else buy_price,
                    'ask': pd.NA if ask_price == -1 else ask_price,
                    'timestamp': current_time
                })

        if not live_records:
            logging.warning("No base-tier market records found in the JSON data.")
            return pd.DataFrame(), vendor_prices

        live_df = pd.DataFrame(live_records)
        live_df['buy'] = pd.to_numeric(live_df['buy'], errors='coerce')
        live_df['ask'] = pd.to_numeric(live_df['ask'], errors='coerce')
        live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], errors='coerce')

        logging.info(f"Loaded {len(live_df)} base-tier market records.")
        return live_df, vendor_prices

    except Exception as e:
        logging.error(f"Unexpected error loading live data from {JSON_PATH}: {e}", exc_info=True)
        return pd.DataFrame(), {}

# --- Utility Function ---
def get_item_name_from_hrid(product_hrid):
    """
    Transforms the raw API HRID into a human-readable name and a category key.
    E.g., '/items/verdant_milk' -> 'Verdant Milk' (Display Name) and 'verdant milk' (Category Key)
    """
    # Remove '/items/' prefix
    cleaned_name = product_hrid.split('/')[-1]
    # Convert snake_case to space-separated, Title Case
    human_readable_name = cleaned_name.replace('_', ' ').title()
    # Create the key for lookup (lowercase space-separated)
    category_key = human_readable_name.lower()
    return human_readable_name, category_key

# --- Main Execution ---
def main():
    """Main build process."""
    logging.info("--- Starting Static Site Build ---")
    try:
        # Ensure the output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True) 
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

        # --- Parse Categories ---
        item_categories = parse_categories(CATEGORY_FILE_PATH)
        if not item_categories:
            logging.warning("Category data is empty. Categories will be missing in output.")

        # --- Download and Load Live Data ---
        logging.info("--- Downloading Data ---")
        # Included download logic here for completeness
        try:
            response = requests.get(JSON_URL, stream=True, timeout=60)
            response.raise_for_status()
            with open(JSON_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Successfully downloaded {JSON_PATH}")
            
            market_df, vendor_prices = load_live_data()
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error downloading {JSON_URL}: {e}")
            return
        except Exception as e:
            logging.error(f"Unexpected error during download/load: {e}", exc_info=True)
            return

        if market_df.empty:
            logging.error("Live market data is empty. Cannot build site.")
            return

        # --- Prepare Products for Summary (Using the raw HRID list) ---
        all_product_hrids = sorted(list(market_df['product_hrid'].unique()))
        filtered_products = all_product_hrids
        logging.info(f"Including {len(filtered_products)} base products in the summary.")

        # --- Generate Market Summary JSON ---
        logging.info("--- Generating market_summary.json ---")
        market_summary = []
        uncategorized_items = []

        for product_hrid in filtered_products:
            
            # CRITICAL FIX: Transform the HRID into the human-readable name for display and lookup
            human_readable_name, category_key = get_item_name_from_hrid(product_hrid)
            
            # Perform the category lookup
            category_name = item_categories.get(category_key, 'Unknown')
            
            if category_name == 'Unknown':
                uncategorized_items.append(human_readable_name)

            # Get the single base-tier price record for the current HRID
            product_data = market_df[market_df['product_hrid'] == product_hrid].iloc[0] 
            
            market_summary.append({
                'name': human_readable_name, # Use the correctly formatted name for display
                'category': category_name,
                'buy': product_data['buy'] if pd.notna(product_data['buy']) else None,
                'ask': product_data['ask'] if pd.notna(product_data['ask']) else None,
                # Vendor lookup still uses the raw HRID key
                'vendor': vendor_prices.get(product_hrid)
            })

        summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f, allow_nan=False, default=str)
        logging.info(f"Saved market summary ({len(market_summary)} items) to {summary_path}")

        # Final diagnostic logging: Report all uncategorized items
        if uncategorized_items:
            logging.warning(f"Found {len(uncategorized_items)} items with 'Unknown' category. Please check names in cata.txt.")
            logging.warning(f"UNCATEGORIZED ITEMS (Display Name): {', '.join(uncategorized_items)}")
        else:
            logging.info("All products successfully categorized.")
            
        # --- Copy HTML files (Ensures site loads, resolving 404) ---
        try:
            shutil.copyfile("templates/index.html", os.path.join(OUTPUT_DIR, "index.html"))
            logging.info(f"Copied index.html to {OUTPUT_DIR}/index.html")
            
            if os.path.exists("templates/404.html"):
                shutil.copyfile("templates/404.html", os.path.join(OUTPUT_DIR, "404.html"))
                logging.info(f"Copied 404.html to {OUTPUT_DIR}/404.html")
            
        except FileNotFoundError:
            logging.error("Could not find required HTML template file in the 'templates/' folder. Please check file names.")

        logging.info("--- Static Site Build Finished Successfully ---")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the main build process: {e}", exc_info=True)

if __name__ == '__main__':
    # Initial download removed from main() to allow try/except for requests errors
    main()

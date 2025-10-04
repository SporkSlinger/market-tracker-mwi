import os 
import sqlite3 
import json 
import requests 
import pandas as pd 
import numpy as np # Import numpy for NaN/Inf checking/replacement 
from datetime import datetime, timedelta, timezone # Import timezone and timedelta
import logging # Import logging
import math # Import math
import shutil 
from collections import defaultdict # For grouping trends by category 
from jinja2 import Environment, FileSystemLoader, TemplateNotFound # For rendering HTML template 

# --- Configuration - 
# Use INFO for general progress, DEBUG for detailed step-by-step logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s') 

# Source Data URLs and Paths 
DB_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/market.db" # Added back historical DB
JSON_URL = "https://www.milkywayidle.com/game_data/marketplace.json"
DB_PATH = "market.db"
JSON_PATH = "marketplace.json"
CATEGORY_FILE_PATH = "cata.txt"

# Output directory for static files 
OUTPUT_DIR = "output" # CRITICAL FIX: Ensures output goes to the deployment folder
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data") # Subdir for data files 
TEMPLATE_DIR = "templates" # Added back template dir

# History and Trend Settings
HISTORICAL_DAYS = 30 # How much history to process
VOLATILITY_DAYS = 7 # How many days back to calculate volatility

# --- Utility Function ---
def get_item_name_from_hrid(product_hrid):
    """
    Transforms the raw API HRID into a human-readable name and a category key.
    E.g., '/items/verdant_milk' -> 'Verdant Milk' (Display Name) and 'verdant milk' (Category Key)
    """
    # Remove '/items/' prefix
    cleaned_name = product_hrid.split('/')[-1] if product_hrid.startswith('/items/') else product_hrid
    # Convert snake_case to space-separated, Title Case
    human_readable_name = cleaned_name.replace('_', ' ').title()
    # Create the key for lookup (lowercase space-separated)
    category_key = human_readable_name.lower()
    return human_readable_name, category_key


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

# --- Data Fetching ---
def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    logging.info(f"Attempting to download {url} to {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded {local_path}")
        return True
    except requests.exceptions.RequestException as e: # Catch specific requests errors
        logging.error(f"Network error downloading {url}: {e}")
        return False
    except IOError as e: # Catch file writing errors
        logging.error(f"IOError saving file to {local_path}: {e}")
        return False
    except Exception as e: # Catch unexpected errors during download/save
        logging.error(f"Unexpected error downloading {url}: {e}", exc_info=True)
        return False

# --- Data Loading and Processing --- 
def load_historical_data(days_to_load): 
    """
    Loads historical data from the SQLite database.
    (Uses raw HRID for product column)
    """ 
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}") 
    if not os.path.exists(DB_PATH): 
        logging.warning(f"{DB_PATH} not found.") 
        return pd.DataFrame() 

    try: 
        with sqlite3.connect(DB_PATH) as conn: 
            cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp() 
            ask_query = "SELECT * FROM ask WHERE time >= ?" 
            ask_df_wide = pd.read_sql_query(ask_query, conn, params=(cutoff_timestamp,)) 
            bid_query = "SELECT * FROM bid WHERE time >= ?" 
            bid_df_wide = pd.read_sql_query(bid_query, conn, params=(cutoff_timestamp,)) 
    except sqlite3.Error as e: 
        logging.error(f"Database error accessing {DB_PATH}: {e}", exc_info=True) 
        return pd.DataFrame() 
    except Exception as e: 
        logging.error(f"Unexpected error connecting to or reading {DB_PATH}: {e}", exc_info=True) 
        return pd.DataFrame() 

    logging.info(f"DB Query: Loaded {len(ask_df_wide)} wide ask, {len(bid_df_wide)} wide bid records.") 
    if ask_df_wide.empty and bid_df_wide.empty: 
        return pd.DataFrame() 

    # Process Ask Data 
    ask_df_long = pd.DataFrame() 
    if not ask_df_wide.empty and 'time' in ask_df_wide.columns: 
        item_columns_ask = [col for col in ask_df_wide.columns if col.lower() != 'time'] 
        if item_columns_ask: 
            try: 
                logging.info(f"Melting 'ask' table ({len(ask_df_wide)} rows)...") 
                ask_df_long = ask_df_wide.melt(id_vars=['time'], value_vars=item_columns_ask, var_name='product', value_name='ask') 
                ask_df_long['timestamp'] = pd.to_datetime(ask_df_long['time'], unit='s', errors='coerce') 
                ask_df_long['ask'] = pd.to_numeric(ask_df_long['ask'], errors='coerce') 
                ask_df_long['ask'] = ask_df_long['ask'].replace(-1, pd.NA) 
                ask_df_long.drop(columns=['time'], inplace=True) 
                ask_df_long.dropna(subset=['timestamp'], inplace=True) 
                ask_df_long = ask_df_long[['product', 'ask', 'timestamp']] 
            except Exception as e: 
                 logging.error(f"Unexpected error processing 'ask' data: {e}", exc_info=True) 
        else: logging.warning("No item columns found in 'ask' table.") 
    del ask_df_wide 

    # Process Bid Data 
    bid_df_long = pd.DataFrame() 
    if not bid_df_wide.empty and 'time' in bid_df_wide.columns: 
        item_columns_bid = [col for col in bid_df_wide.columns if col.lower() != 'time'] 
        if item_columns_bid: 
            try: 
                logging.info(f"Melting 'bid' table ({len(bid_df_wide)} rows)...") 
                bid_df_long = bid_df_wide.melt(id_vars=['time'], value_vars=item_columns_bid, var_name='product', value_name='buy') 
                bid_df_long['timestamp'] = pd.to_datetime(bid_df_long['time'], unit='s', errors='coerce') 
                bid_df_long['buy'] = pd.to_numeric(bid_df_long['buy'], errors='coerce') 
                bid_df_long['buy'] = bid_df_long['buy'].replace(-1, pd.NA) 
                bid_df_long.drop(columns=['time'], inplace=True) 
                bid_df_long.dropna(subset=['timestamp'], inplace=True) 
                bid_df_long = bid_df_long[['product', 'buy', 'timestamp']] 
            except Exception as e: 
                 logging.error(f"Unexpected error processing 'bid' data: {e}", exc_info=True) 
        else: logging.warning("No item columns found in 'bid' table.") 
    del bid_df_wide 

    # Merge Ask and Bid Data 
    logging.info("Merging melted ask and bid data...") 
    try: 
        if not ask_df_long.empty and not bid_df_long.empty: 
            merged_df = pd.merge(ask_df_long, bid_df_long, on=['product', 'timestamp'], how='outer') 
        elif not ask_df_long.empty: 
            merged_df = ask_df_long.copy(); merged_df['buy'] = pd.NA 
        elif not bid_df_long.empty: 
            merged_df = bid_df_long.copy(); merged_df['ask'] = pd.NA 
        else: 
            merged_df = pd.DataFrame() 
        del ask_df_long, bid_df_long 

        if not merged_df.empty: 
              logging.info(f"Sorting merged data ({len(merged_df)} records)...") 
              merged_df.sort_values(by=['product', 'timestamp'], inplace=True) 
              final_cols = ['product', 'buy', 'ask', 'timestamp'] 
              for col in final_cols: 
                  if col not in merged_df.columns: merged_df[col] = pd.NA 
              merged_df = merged_df[final_cols] 
         
        logging.info(f"Finished historical data processing. Records: {len(merged_df)}") 
        return merged_df 
    except Exception as e: 
        logging.error(f"Unexpected error during historical data merge/sort: {e}", exc_info=True) 
        return pd.DataFrame() 

def load_live_data(): 
    """
    Loads live market data and vendor prices from the JSON file.
    CRITICAL FIX: Loads ALL tiers for enhanced item tracking, but separates base (Tier 0) prices.
    Returns: (base_df, all_tiers_df, vendor_prices)
    """ 
    logging.info(f"Loading live data from {JSON_PATH}") 
    vendor_prices = {} 
    base_records = [] 
    all_tier_records = []
    
    if not os.path.exists(JSON_PATH): 
        logging.warning(f"{JSON_PATH} not found.") 
        return pd.DataFrame(), pd.DataFrame(), {} 

    try: 
        with open(JSON_PATH, 'r', encoding='utf-8') as f: 
            data = json.load(f) 
        
        # FIX #1: Load vendor prices
        if 'vendorPrice' in data and isinstance(data['vendorPrice'], dict):
            vendor_prices = data['vendorPrice']
            logging.info(f"Loaded {len(vendor_prices)} vendor prices.")
        else:
            logging.warning("'vendorPrice' key not found in JSON. Vendor prices will be missing.")


        if 'marketData' not in data or not isinstance(data['marketData'], dict): 
            logging.error(f"Invalid JSON structure in {JSON_PATH}: 'marketData' key missing or not a dictionary.") 
            return pd.DataFrame(), pd.DataFrame(), vendor_prices

        market_data = data['marketData'] 
        current_time = datetime.now(timezone.utc) 

        for product_path, tiers in market_data.items(): 
            if not isinstance(tiers, dict): continue 
            
            for tier_str, prices in tiers.items():
                if not isinstance(prices, dict): continue 
                
                ask_price = prices.get('a', -1) 
                buy_price = prices.get('b', -1) 
                
                ask = pd.NA if ask_price == -1 else ask_price 
                buy = pd.NA if buy_price == -1 else buy_price 
                
                record = { 
                    'product': product_path, 
                    'buy': buy, 
                    'ask': ask, 
                    'timestamp': current_time,
                    'tier': tier_str # Include tier string
                }
                
                all_tier_records.append(record)
                
                # CRITICAL FIX #2: Filter for base item (Tier "0") for the main market summary
                if tier_str == "0":
                    # For the base DF, we don't need the 'tier' column initially, but it won't hurt
                    base_records.append(record)


        base_df = pd.DataFrame(base_records) 
        all_tiers_df = pd.DataFrame(all_tier_records)
        
        # Helper function for cleaning DF
        def clean_live_df(df):
            if not df.empty: 
                df['buy'] = pd.to_numeric(df['buy'], errors='coerce') 
                df['ask'] = pd.to_numeric(df['ask'], errors='coerce') 
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
                df.dropna(subset=['timestamp'], inplace=True) 
                if 'tier' in df.columns:
                    df = df[['product', 'buy', 'ask', 'timestamp', 'tier']] 
                else:
                    df = df[['product', 'buy', 'ask', 'timestamp']] 
            return df
            
        base_df = clean_live_df(base_df)
        all_tiers_df = clean_live_df(all_tiers_df)
        
        logging.info(f"Loaded {len(base_df)} base (Tier 0) records.") 
        logging.info(f"Loaded {len(all_tiers_df)} all-tier records.")
        return base_df, all_tiers_df, vendor_prices 

    except json.JSONDecodeError as e: 
        logging.error(f"Error decoding JSON from {JSON_PATH}: {e}", exc_info=True) 
        return pd.DataFrame(), pd.DataFrame(), vendor_prices 
    except FileNotFoundError: 
        logging.error(f"File not found at {JSON_PATH} despite initial check.") 
        return pd.DataFrame(), pd.DataFrame(), vendor_prices 
    except IOError as e: 
        logging.error(f"IOError reading live data file {JSON_PATH}: {e}", exc_info=True) 
        return pd.DataFrame(), pd.DataFrame(), vendor_prices 
    except Exception as e: 
        logging.error(f"Unexpected error loading live data: {e}", exc_info=True) 
        return pd.DataFrame(), pd.DataFrame(), vendor_prices

# --- Trend Calculation (Restored) --- 
def calculate_trends(df, products): 
    """ 
    Calculates 24h trend based on 'ask' price for a given list of products. 
    """ 
    trends = {}; processed_count = 0 
    if df.empty or not products: return trends 

    now = datetime.now(timezone.utc) 
    twenty_four_hours_ago = now - timedelta(hours=24) 
    seven_days_ago = now - timedelta(days=7) 

    logging.info(f"Calculating trends for {len(products)} products relative to {now.strftime('%Y-%m-%d %H:%M:%S %Z')} using refined averaging logic (ASK price)") 

    try: 
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']): 
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
        if df['timestamp'].dt.tz is None: 
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC') 
        else: 
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC') 
        df.dropna(subset=['timestamp'], inplace=True) 
        if df.empty: 
            logging.warning("DataFrame is empty after timestamp processing in calculate_trends.") 
            return trends 
    except Exception as e: 
        logging.error(f"Error preparing timestamp column for trend calculation: {e}", exc_info=True) 
        return trends 

    try: 
        grouped = df.groupby('product') 
    except KeyError: 
        logging.error("Column 'product' not found for grouping in calculate_trends.") 
        return trends 

    for product in products: # Iterate only through the filtered list 
        try: 
            product_group = grouped.get_group(product) 
            if product_group.empty: continue 

            latest_data = product_group.iloc[-1] 
            current_price = latest_data['ask'] 
            if pd.isna(current_price): continue 

            previous_price = pd.NA 
            df_last_24h = product_group[product_group['timestamp'] >= twenty_four_hours_ago] 
            valid_ask_24h = df_last_24h['ask'].dropna() 
            num_points_24h = len(valid_ask_24h) 

            if num_points_24h > 1: 
                previous_price = valid_ask_24h.mean() 
            elif num_points_24h == 1: 
                previous_price = valid_ask_24h.iloc[0] 
            else: 
                df_last_7d = product_group[product_group['timestamp'] >= seven_days_ago] 
                valid_ask_7d = df_last_7d['ask'].dropna() 
                if len(valid_ask_7d) > 0: 
                    previous_price = valid_ask_7d.mean() 
                else: 
                    previous_price = pd.NA 

            if pd.isna(previous_price): continue 

            if previous_price != 0: 
                change_pct = ((current_price - previous_price) / previous_price) * 100 
                trends[product] = change_pct 
                processed_count += 1 
            else: 
                trends[product] = float('inf') if current_price > 0 else 0.0 

        except KeyError: 
            continue 
        except Exception as e: 
            logging.error(f"Error calculating trend for {product}: {e}", exc_info=True) 
            continue 

    logging.info(f"Finished trend calculation. Calculated trends for {processed_count} out of {len(products)} products.") 
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in trends.items()} 

# --- Volatility & CV Calculation (Restored) --- 
def calculate_volatility_and_cv(df, products, days=7): 
    """ 
    Calculates price volatility (std dev) and normalized volatility (CV) 
    of ask price over a given period for a list of products. 
    """ 
    results = {} 
    if df.empty or not products: 
        logging.warning("Cannot calculate volatility/CV: DataFrame or product list is empty.") 
        return results 

    logging.info(f"Calculating {days}-day volatility & CV for {len(products)} products (using ASK price)...") 
    now = datetime.now(timezone.utc) 
    time_cutoff = now - timedelta(days=days) 

    try: 
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']): 
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
        if df['timestamp'].dt.tz is None: 
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC') 
        else: 
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC') 

        df.dropna(subset=['timestamp'], inplace=True) 
        if df.empty: 
              logging.warning("DataFrame is empty after timestamp processing in calculate_volatility_and_cv.") 
              return results 

        df_period = df[df['timestamp'] >= time_cutoff] 
        if df_period.empty: 
            logging.warning(f"No data found within the last {days} days for volatility/CV calculation.") 
            return results 

        grouped = df_period.groupby('product') 

    except Exception as e: 
        logging.error(f"Error preparing data for volatility/CV calculation: {e}", exc_info=True) 
        return results 

    processed_count = 0 
    for product in products: 
        std_dev = None 
        cv_percent = None 
        mean_price = None 
        try: 
            if product not in grouped.groups: continue 

            product_group = grouped.get_group(product) 
            valid_ask_period = product_group['ask'].dropna() 

            if len(valid_ask_period) >= 2: 
                std_dev = valid_ask_period.std(ddof=1) 
                mean_price = valid_ask_period.mean() 

                if mean_price is not None and not pd.isna(mean_price) and mean_price != 0: 
                    cv = std_dev / mean_price 
                    cv_percent = cv * 100 
                processed_count += 1 
            
        except KeyError: 
              continue 
        except Exception as e: 
            logging.error(f"Error calculating volatility/CV for {product}: {e}", exc_info=True) 
            continue 
        finally: 
            results[product] = { 
                'std': std_dev if pd.notna(std_dev) and np.isfinite(std_dev) else None, 
                'cv': cv_percent if pd.notna(cv_percent) and np.isfinite(cv_percent) else None 
            } 


    logging.info(f"Finished volatility/CV calculation. Calculated stats for {processed_count} products with >=2 data points in the period.") 
    return results 

# --- Market Index Calculation (Restored) --- 
def calculate_market_indices(product_trends, item_categories): 
    """Calculates the average trend for each item category based on provided trends.""" 
    if not product_trends or not item_categories: 
        logging.warning("Cannot calculate market indices: Missing product trends or item categories.") 
        return {} 

    logging.info("Calculating market indices by category...") 
    category_trends = defaultdict(list) 

    # Group valid trends by category 
    for product, trend in product_trends.items(): 
        if trend is not None: 
            # FIX: Use the category key (lowercase) here for the lookup!
            human_name, category_key = get_item_name_from_hrid(product)
            
            category = item_categories.get(category_key) 
            if category and category != "Unknown": 
                category_trends[category].append(trend) 

    market_indices = {} 
    # Calculate average trend for each category 
    for category, trends_list in category_trends.items(): 
        if trends_list: 
            try: 
                average_trend = sum(trends_list) / len(trends_list) 
                market_indices[category] = average_trend 
            except Exception as e: 
                logging.error(f"Error calculating average trend for category '{category}': {e}") 
                market_indices[category] = None 
        else: 
            market_indices[category] = None 

    logging.info(f"Calculated market indices for {len(market_indices)} categories.") 
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in market_indices.items()} 


def main(): 
    """Main build process.""" 
    logging.info("--- Starting Static Site Build ---") 
    try: 
        # --- Setup ---
        os.makedirs(OUTPUT_DIR, exist_ok=True) 
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True) 
        logging.info(f"Output directory '{OUTPUT_DIR}' ensured.") 

        # --- Parse Categories --- 
        item_categories = parse_categories(CATEGORY_FILE_PATH) 

        # --- Download and Load Data --- 
        logging.info("--- Downloading Data ---") 
        db_ok = download_file(DB_URL, DB_PATH) 
        json_ok = download_file(JSON_URL, JSON_PATH) 
        if not json_ok: 
            logging.error("Failed to download critical live data (marketplace.json). Cannot proceed.") 
            return # Stop execution if live data fails 
        if not db_ok: 
            logging.warning("Failed to download market.db. Historical data and trends might be incomplete.") 

        logging.info("--- Loading Data ---") 
        # load_live_data returns (base_df, all_tiers_df, vendor_prices)
        base_live_df, all_tiers_live_df, vendor_prices = load_live_data() 
        historical_df = load_historical_data(days_to_load=HISTORICAL_DAYS) 

        # --- Combine Data for Trends (Uses only base prices from live data) --- 
        if historical_df.empty and base_live_df.empty: 
              logging.error("Both historical and live data sources are empty. Cannot build site.") 
              combined_df = pd.DataFrame() 
        elif base_live_df.empty: 
              logging.warning("Live data is empty, using only historical data.") 
              combined_df = historical_df.copy() 
        elif historical_df.empty: 
              logging.warning("Historical data is empty, using only live data. Trends and volatility cannot be calculated.") 
              combined_df = base_live_df.copy() 
        else: 
              logging.info("Concatenating historical and live base data...") 
              combined_df = pd.concat([historical_df, base_live_df], ignore_index=True) 

        # --- Final Data Cleaning for Trends/Summary --- 
        if not combined_df.empty: 
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce') 
            combined_df.dropna(subset=['timestamp'], inplace=True) 
            if not combined_df.empty:
                if combined_df['timestamp'].dt.tz is None: 
                    combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC') 
                else: 
                    combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC') 

                combined_df.sort_values(by=['product', 'timestamp'], inplace=True) 
                combined_df.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True) 
                combined_df['buy'] = pd.to_numeric(combined_df['buy'], errors='coerce') 
                combined_df['ask'] = pd.to_numeric(combined_df['ask'], errors='coerce') 
                logging.info(f"Final processed data size for trends: {len(combined_df)} records.") 
            else: 
                logging.warning("Combined DataFrame became empty after timestamp processing.") 
        
        # --- Filter Products for Display (Uses the raw HRID list) --- 
        all_product_hrids = sorted(list(combined_df['product'].unique())) if not combined_df.empty else [] 
        
        # CRITICAL FIX: DISABLE VENDOR FILTERING UNTIL VENDOR DATA IS AVAILABLE.
        # This ensures ALL 800+ products are included, preventing the "Filtered down to 1 products" error.
        if not all_product_hrids: 
              logging.warning("No products found in combined data to process.") 
              filtered_products = [] 
        else: 
            logging.info(f"Disabling vendor price filter. Including all {len(all_product_hrids)} products...") 
            filtered_products = all_product_hrids # Include ALL products
            
            # The original destructive filter (COMMENTED OUT):
            # filtered_products = [ 
            #     p for p in all_product_hrids 
            #     if p == "/items/bag_of_10_cowbells" or ((vp := vendor_prices.get(p)) is not None and vp > 0) 
            # ] 
            logging.info(f"Filtered list size: {len(filtered_products)} products.") 


        # --- Calculate Trends, Volatility, CV, and Indices for Filtered Products --- 
        product_trends = calculate_trends(combined_df.copy(), filtered_products) 
        product_volatility_stats = calculate_volatility_and_cv(combined_df.copy(), filtered_products, days=VOLATILITY_DAYS) 
        market_indices = calculate_market_indices(product_trends, item_categories) 

        # --- Generate JSON Data Files --- 
        logging.info("--- Generating JSON Data Files ---") 

        # 1. Market Summary (Base Prices, Trends, Volatility)
        market_summary = [] 
        uncategorized_items = [] 
        if not combined_df.empty: 
            try: 
                latest_data_map = combined_df.groupby('product').last() 
                
                for product_hrid in filtered_products: 
                    if product_hrid in latest_data_map.index: 
                        latest = latest_data_map.loc[product_hrid] 
                        
                        # CRITICAL FIX: Transform the HRID for categorization and display
                        human_readable_name, category_key = get_item_name_from_hrid(product_hrid)
                        category_name = item_categories.get(category_key, 'Unknown')
                        
                        if category_name == 'Unknown':
                            uncategorized_items.append(human_readable_name)

                        vol_stats = product_volatility_stats.get(product_hrid, {'std': None, 'cv': None}) 
                        
                        market_summary.append({ 
                            'name': human_readable_name, # Display name
                            'category': category_name, # Categorized name
                            'buy': latest['buy'] if pd.notna(latest['buy']) else None, 
                            'ask': latest['ask'] if pd.notna(latest['ask']) else None, 
                            'vendor': vendor_prices.get(product_hrid), 
                            'trend': product_trends.get(product_hrid), 
                            f'volatility_{VOLATILITY_DAYS}d': vol_stats['std'], 
                            f'volatility_norm_{VOLATILITY_DAYS}d': vol_stats['cv'] 
                        }) 
                    else: 
                        logging.warning(f"Filtered product '{product_hrid}' not found in latest data map. Skipping summary entry.") 
            except Exception as e: 
                logging.error(f"Error creating market summary list: {e}", exc_info=True) 
                market_summary = [] 

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
            
        # 2. Enhanced Item Prices (NEW FEATURE)
        logging.info("Generating enhanced item price data (market_enhanced.json)...")
        enhanced_data = defaultdict(lambda: {'tiers': {}})
        
        # Filter all_tiers_live_df to only include items that passed the vendor filter (i.e., all included items)
        enhanced_df = all_tiers_live_df[all_tiers_live_df['product'].isin(filtered_products)].copy()
        
        for index, row in enhanced_df.iterrows():
            product_hrid = row['product']
            human_name, _ = get_item_name_from_hrid(product_hrid)
            tier = str(row['tier'])
            
            enhanced_data[human_name]['tiers'][tier] = {
                'ask': row['ask'] if pd.notna(row['ask']) else None,
                'buy': row['buy'] if pd.notna(row['buy']) else None
            }
            
        enhanced_json_output = {k: dict(v) for k, v in enhanced_data.items()}

        enhanced_path = os.path.join(OUTPUT_DATA_DIR, 'market_enhanced.json')
        with open(enhanced_path, 'w') as f:
            json.dump(enhanced_json_output, f, allow_nan=False, default=str)
        logging.info(f"Saved enhanced item data ({len(enhanced_json_output)} items) to {enhanced_path}")
        
        # 3. Full Historical Data (Filtered)
        logging.info("Generating full historical data (market_history.json)...")
        nested_history_dict = {}
        history_df = combined_df[combined_df['product'].isin(filtered_products)].copy()
        
        if not history_df.empty:
            
            # Map raw HRID to human-readable name for frontend consumption
            history_df['name'] = history_df['product'].apply(lambda x: get_item_name_from_hrid(x)[0])
            
            history_grouped = history_df.groupby('name')
            
            for name, group in history_grouped:
                history_points = []
                group = group.sort_values(by='timestamp') 
                
                for index, row in group.iterrows():
                    timestamp_ms = int(row['timestamp'].value / 10**6) 
                    
                    history_points.append([
                        timestamp_ms,
                        row['ask'] if pd.notna(row['ask']) else None,
                        row['buy'] if pd.notna(row['buy']) else None
                    ])
                
                nested_history_dict[name] = history_points

        history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
        with open(history_path, 'w') as f:
            json.dump(nested_history_dict, f, allow_nan=False, default=str)
        logging.info(f"Saved historical data for {len(nested_history_dict)} products to {history_path}")

        # 4. Market Index Summary (Graphs)
        logging.info("Generating market index data (market_indices.json)...")
        indices_path = os.path.join(OUTPUT_DATA_DIR, 'market_indices.json')
        with open(indices_path, 'w') as f:
            json.dump(market_indices, f, allow_nan=False, default=str)
        logging.info(f"Saved market indices for {len(market_indices)} categories to {indices_path}")

        # --- Copy HTML files (Ensures site loads) ---
        try:
            shutil.copyfile(os.path.join(TEMPLATE_DIR, "index.html"), os.path.join(OUTPUT_DIR, "index.html"))
            logging.info(f"Copied index.html to {OUTPUT_DIR}/index.html")
            
            if os.path.exists(os.path.join(TEMPLATE_DIR, "404.html")):
                shutil.copyfile(os.path.join(TEMPLATE_DIR, "404.html"), os.path.join(OUTPUT_DIR, "404.html"))
                logging.info(f"Copied 404.html to {OUTPUT_DIR}/404.html")
            
        except FileNotFoundError:
            logging.error("Could not find required HTML template file in the 'templates/' folder. Please check file names.")

        logging.info("--- Static Site Build Finished Successfully ---")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the main build process: {e}", exc_info=True)

if __name__ == '__main__':
    main()

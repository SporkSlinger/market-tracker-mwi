import os
import sqlite3
import json
import requests
import pandas as pd
import numpy as np # Import numpy for NaN/Inf checking/replacement
from datetime import datetime, timedelta, timezone # Import timezone
import logging
import math
import shutil
from collections import defaultdict # For grouping trends by category
from jinja2 import Environment, FileSystemLoader, TemplateNotFound # For rendering HTML template

# --- Configuration --
# Use INFO for general progress, DEBUG for detailed step-by-step logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Source Data URLs and Paths
DB_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/market.db"
JSON_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/milkyapi.json"
DB_PATH = "market.db" # Downloaded file paths
JSON_PATH = "milkyapi.json"
CATEGORY_FILE_PATH = "cata.txt" # Path to your category file

# Output directory for static files
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data") # Subdir for data files
TEMPLATE_DIR = "templates" # Directory for Jinja templates

# History and Trend Settings
HISTORICAL_DAYS = 30 # How much history to process
VOLATILITY_DAYS = 7 # How many days back to calculate volatility

# --- Category Parsing ---
def parse_categories(filepath):
    """Parses the cata.txt file into a dictionary {item_name: category}."""
    categories = {}
    current_main_category = "Unknown"
    current_sub_category = None
    current_display_category = "Unknown" # Category name to assign to items

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
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line == '•': continue
                logging.debug(f"Processing Line {line_num}: '{line}'")

                if line in known_main_categories:
                    current_main_category = line
                    current_sub_category = None
                    current_display_category = current_main_category
                    logging.debug(f"  -> Matched Main Category: {current_main_category}")
                    continue

                if current_main_category == "Equipment" and line in equipment_subcategories:
                    current_sub_category = line
                    current_display_category = f"Equipment / {current_sub_category}"
                    logging.debug(f"  -> Matched Equipment Sub Category: {current_sub_category} -> Display: {current_display_category}")
                    continue

                if current_main_category == "Tools" and line in tool_subcategories:
                    current_sub_category = line
                    current_display_category = f"Tools / {current_sub_category}"
                    logging.debug(f"  -> Matched Tool Sub Category: {current_sub_category} -> Display: {current_display_category}")
                    continue

                item_names = [name.strip() for name in line.split(',') if name.strip()]
                if not item_names:
                    logging.warning(f"  -> Line {line_num} parsed as empty item list: '{line}'")
                    continue

                for item_name in item_names:
                    if item_name:
                        if current_display_category == "Unknown" and current_main_category != "Unknown":
                             current_display_category = current_main_category
                        categories[item_name] = current_display_category
                        logging.debug(f"  -> Found Item: '{item_name}' -> '{current_display_category}'")
                    else:
                         logging.warning(f"  -> Found empty item name after split/strip on line {line_num}: '{line}'")

        logging.info(f"Parsed {len(categories)} items from category file.")
        sample_items = list(categories.items())[:5] + list(categories.items())[-5:]
        logging.debug(f"Category parsing sample (first/last 5): {sample_items}")
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}. Categories will be missing.")
        return {}
    except IOError as e: # More specific IO error
        logging.error(f"IOError reading category file {filepath}: {e}", exc_info=True)
        return {}
    except Exception as e: # Catch-all for other parsing issues
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
    """Loads historical data from the SQLite database."""
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.warning(f"{DB_PATH} not found.")
        return pd.DataFrame()

    # Use 'with' statement for automatic connection closing
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp()
            # Load ask data
            ask_query = "SELECT * FROM ask WHERE time >= ?"
            ask_df_wide = pd.read_sql_query(ask_query, conn, params=(cutoff_timestamp,))
            # Load bid data
            bid_query = "SELECT * FROM bid WHERE time >= ?"
            bid_df_wide = pd.read_sql_query(bid_query, conn, params=(cutoff_timestamp,))
    except sqlite3.Error as e: # Catch specific SQLite errors
        logging.error(f"Database error accessing {DB_PATH}: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e: # Catch other potential errors during DB access
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
                ask_df_long['ask'] = ask_df_long['ask'].replace(-1, pd.NA) # Replace -1 after numeric conversion
                ask_df_long.drop(columns=['time'], inplace=True)
                ask_df_long.dropna(subset=['timestamp'], inplace=True) # Only drop rows with invalid timestamps
                ask_df_long = ask_df_long[['product', 'ask', 'timestamp']]
            except (TypeError, ValueError, KeyError, AttributeError) as melt_error: # More specific pandas/processing errors
                logging.error(f"Error processing 'ask' data during melt/conversion: {melt_error}", exc_info=True)
            except Exception as e:
                 logging.error(f"Unexpected error processing 'ask' data: {e}", exc_info=True)
        else: logging.warning("No item columns found in 'ask' table.")
    del ask_df_wide # Free memory

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
                bid_df_long['buy'] = bid_df_long['buy'].replace(-1, pd.NA) # Replace -1 after numeric conversion
                bid_df_long.drop(columns=['time'], inplace=True)
                bid_df_long.dropna(subset=['timestamp'], inplace=True) # Only drop rows with invalid timestamps
                bid_df_long = bid_df_long[['product', 'buy', 'timestamp']]
            except (TypeError, ValueError, KeyError, AttributeError) as melt_error: # More specific pandas/processing errors
                logging.error(f"Error processing 'bid' data during melt/conversion: {melt_error}", exc_info=True)
            except Exception as e:
                 logging.error(f"Unexpected error processing 'bid' data: {e}", exc_info=True)
        else: logging.warning("No item columns found in 'bid' table.")
    del bid_df_wide # Free memory

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
        del ask_df_long, bid_df_long # Free memory

        if not merged_df.empty:
             logging.info(f"Sorting merged data ({len(merged_df)} records)...")
             merged_df.sort_values(by=['product', 'timestamp'], inplace=True)
             # Ensure essential columns exist
             final_cols = ['product', 'buy', 'ask', 'timestamp']
             for col in final_cols:
                 if col not in merged_df.columns: merged_df[col] = pd.NA
             merged_df = merged_df[final_cols]
        logging.info(f"Finished historical data processing. Records: {len(merged_df)}")
        return merged_df
    except (pd.errors.MergeError, KeyError, ValueError) as merge_error:
        logging.error(f"Error merging historical dataframes: {merge_error}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error during historical data merge/sort: {e}", exc_info=True)
        return pd.DataFrame()

def load_live_data():
    """Loads live market data and vendor prices from the JSON file."""
    logging.info(f"Loading live data from {JSON_PATH}")
    vendor_prices = {}; live_records = []
    if not os.path.exists(JSON_PATH):
        logging.warning(f"{JSON_PATH} not found.")
        return pd.DataFrame(), {}

    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)

        if 'market' not in data or not isinstance(data['market'], dict):
            logging.error(f"Invalid JSON structure in {JSON_PATH}: 'market' key missing or not a dictionary.")
            return pd.DataFrame(), {}

        market_data = data['market']
        current_time = datetime.now(timezone.utc) # Use timezone-aware timestamp

        for product_name, price_info in market_data.items():
            if isinstance(price_info, dict) and 'ask' in price_info and 'bid' in price_info:
                # Convert -1 to pd.NA for internal processing
                ask_price = pd.NA if price_info['ask'] == -1 else price_info['ask']
                buy_price = pd.NA if price_info['bid'] == -1 else price_info['bid']
                vendor_price = price_info.get('vendor', pd.NA) # Use NA for missing vendor price

                live_records.append({
                    'product': product_name,
                    'buy': buy_price,
                    'ask': ask_price,
                    'timestamp': current_time
                })

                # Store vendor price (handle -1 and non-integer values)
                if vendor_price == -1 or vendor_price is None or pd.isna(vendor_price):
                    vendor_prices[product_name] = None
                else:
                    try:
                        vendor_prices[product_name] = int(vendor_price)
                    except (ValueError, TypeError):
                        logging.warning(f"Could not convert vendor price '{vendor_price}' to int for product '{product_name}'. Setting to None.")
                        vendor_prices[product_name] = None
            else:
                logging.warning(f"Skipping invalid/incomplete market data item in JSON: '{product_name}'")

        live_records_df = pd.DataFrame(live_records)
        if not live_records_df.empty:
            # Convert types after DataFrame creation
            live_records_df['buy'] = pd.to_numeric(live_records_df['buy'], errors='coerce')
            live_records_df['ask'] = pd.to_numeric(live_records_df['ask'], errors='coerce')
            live_records_df['timestamp'] = pd.to_datetime(live_records_df['timestamp'], errors='coerce')
            live_records_df.dropna(subset=['timestamp'], inplace=True) # Ensure timestamp is valid
            live_records_df = live_records_df[['product', 'buy', 'ask', 'timestamp']] # Ensure column order

        logging.info(f"Loaded {len(live_records_df)} live records and {len(vendor_prices)} vendor prices.")
        return live_records_df, vendor_prices

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {JSON_PATH}: {e}", exc_info=True)
        return pd.DataFrame(), {}
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        logging.error(f"File not found at {JSON_PATH} despite initial check.")
        return pd.DataFrame(), {}
    except IOError as e:
        logging.error(f"IOError reading live data file {JSON_PATH}: {e}", exc_info=True)
        return pd.DataFrame(), {}
    except Exception as e: # Catch-all for other issues (e.g., unexpected structure)
        logging.error(f"Unexpected error loading live data: {e}", exc_info=True)
        return pd.DataFrame(), {}

# --- Trend Calculation ---
def calculate_trends(df, products):
    """
    Calculates 24h trend based on 'ask' price for a given list of products.
    Refined Logic:
    - If >1 point in last 24h: Use 24h average 'ask'.
    - If =1 point in last 24h: Use that single 'ask' price.
    - If 0 points in last 24h: Use 7d average 'ask'.
    - If 0 points in last 7d: No trend.
    """
    trends = {}; processed_count = 0
    if df.empty or not products: return trends

    now = datetime.now(timezone.utc)
    twenty_four_hours_ago = now - timedelta(hours=24)
    seven_days_ago = now - timedelta(days=7)

    logging.info(f"Calculating trends for {len(products)} products relative to {now.strftime('%Y-%m-%d %H:%M:%S %Z')} using refined averaging logic (ASK price)")

    try:
        # Ensure timestamp column is datetime objects and timezone-aware (UTC)
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
                logging.debug(f"Trend for '{product}': Using 24h avg ({num_points_24h} points) = {previous_price}")
            elif num_points_24h == 1:
                previous_price = valid_ask_24h.iloc[0]
                logging.debug(f"Trend for '{product}': Using single 24h point = {previous_price}")
            else:
                df_last_7d = product_group[product_group['timestamp'] >= seven_days_ago]
                valid_ask_7d = df_last_7d['ask'].dropna()
                if len(valid_ask_7d) > 0:
                    previous_price = valid_ask_7d.mean()
                    logging.debug(f"Trend for '{product}': Using 7d avg ({len(valid_ask_7d)} points) = {previous_price}")
                else:
                    logging.debug(f"Trend for '{product}': No valid ask price found in last 7 days.")
                    previous_price = pd.NA

            if pd.isna(previous_price): continue

            if previous_price != 0:
                change_pct = ((current_price - previous_price) / previous_price) * 100
                trends[product] = change_pct
                processed_count += 1
            else:
                trends[product] = float('inf') if current_price > 0 else 0.0

        except KeyError:
            # This might happen if a product is in the filtered list but somehow missing from the grouped object
            logging.warning(f"KeyError processing trend for filtered product {product}. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error calculating trend for {product}: {e}", exc_info=True)
            continue

    logging.info(f"Finished trend calculation. Calculated trends for {processed_count} out of {len(products)} products.")
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in trends.items()}

# --- Volatility Calculation ---
def calculate_volatility(df, products, days=7):
    """Calculates price volatility (standard deviation of ask price) over a given period for a list of products."""
    volatility = {}
    if df.empty or not products:
        logging.warning("Cannot calculate volatility: DataFrame or product list is empty.")
        return volatility

    logging.info(f"Calculating {days}-day volatility for {len(products)} products (using ASK price)...")
    now = datetime.now(timezone.utc)
    time_cutoff = now - timedelta(days=days)

    try:
        # Ensure timestamp column is datetime objects and timezone-aware (UTC)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
             logging.warning("DataFrame is empty after timestamp processing in calculate_volatility.")
             return volatility

        # Filter data for the volatility period
        df_period = df[df['timestamp'] >= time_cutoff]
        if df_period.empty:
            logging.warning(f"No data found within the last {days} days for volatility calculation.")
            return volatility

        grouped = df_period.groupby('product')

    except Exception as e:
        logging.error(f"Error preparing data for volatility calculation: {e}", exc_info=True)
        return volatility

    processed_count = 0
    for product in products: # Iterate only through the filtered list
        try:
            if product not in grouped.groups: continue # Skip if no data for this product in the period

            product_group = grouped.get_group(product)
            valid_ask_period = product_group['ask'].dropna()

            # Standard deviation requires at least 2 data points
            if len(valid_ask_period) >= 2:
                std_dev = valid_ask_period.std(ddof=1) # ddof=1 for sample standard deviation
                volatility[product] = std_dev
                processed_count += 1
            else:
                volatility[product] = None # Not enough data points to calculate std dev

        except KeyError:
             logging.warning(f"KeyError processing volatility for filtered product {product}. Skipping.")
             continue
        except Exception as e:
            logging.error(f"Error calculating volatility for {product}: {e}", exc_info=True)
            volatility[product] = None # Assign None on error
            continue

    logging.info(f"Finished volatility calculation. Calculated volatility for {processed_count} products.")
    # Ensure NaN/Inf are replaced with None for JSON compatibility
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in volatility.items()}

# --- Market Index Calculation ---
def calculate_market_indices(product_trends, item_categories):
    """Calculates the average trend for each item category based on provided trends."""
    # Note: product_trends should already be filtered to include only desired items
    if not product_trends or not item_categories:
        logging.warning("Cannot calculate market indices: Missing product trends or item categories.")
        return {}

    logging.info("Calculating market indices by category...")
    category_trends = defaultdict(list) # Use defaultdict for easier appending

    # Group valid trends by category
    for product, trend in product_trends.items():
        if trend is not None: # Only consider items with a valid trend
            category = item_categories.get(product)
            if category and category != "Unknown": # Ensure category is known
                category_trends[category].append(trend)

    market_indices = {}
    # Calculate average trend for each category
    for category, trends_list in category_trends.items():
        if trends_list: # Ensure list is not empty
            try:
                average_trend = sum(trends_list) / len(trends_list)
                market_indices[category] = average_trend
            except ZeroDivisionError:
                 market_indices[category] = None
            except Exception as e:
                 logging.error(f"Error calculating average trend for category '{category}': {e}")
                 market_indices[category] = None
        else:
            market_indices[category] = None # No items with valid trends in this category

    logging.info(f"Calculated market indices for {len(market_indices)} categories.")
    # Ensure NaN/Inf are replaced with None for JSON compatibility
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in market_indices.items()}


def main():
    """Main build process."""
    logging.info("--- Starting Static Site Build ---")
    try: # Wrap main logic for better top-level error catching
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

        # --- Parse Categories ---
        item_categories = parse_categories(CATEGORY_FILE_PATH)
        if not item_categories:
            logging.warning("Category data is empty or failed to load. Categories will be missing in output.")

        # --- Download and Load Data ---
        logging.info("--- Downloading Data ---")
        db_ok = download_file(DB_URL, DB_PATH)
        json_ok = download_file(JSON_URL, JSON_PATH)
        if not json_ok:
            logging.error("Failed to download critical live data (milkyapi.json). Cannot proceed.")
            return # Stop execution if live data fails
        if not db_ok:
            logging.warning("Failed to download market.db. Historical data and trends might be incomplete.")

        logging.info("--- Loading Data ---")
        historical_df = load_historical_data(days_to_load=HISTORICAL_DAYS)
        live_df, vendor_prices = load_live_data()

        if historical_df.empty and live_df.empty:
             logging.error("Both historical and live data sources are empty. Cannot build site.")
             combined_df = pd.DataFrame() # Ensure combined_df exists even if empty
        elif live_df.empty:
             logging.warning("Live data is empty, using only historical data. Trends might be inaccurate.")
             combined_df = historical_df.copy()
        elif historical_df.empty:
             logging.warning("Historical data is empty, using only live data. Trends and volatility cannot be calculated.")
             combined_df = live_df.copy() # Allow summary generation at least
        else:
             logging.info("Concatenating historical and live data...")
             combined_df = pd.concat([historical_df, live_df], ignore_index=True)

        # --- Process Combined Data ---
        if not combined_df.empty:
            logging.info(f"Processing combined data ({len(combined_df)} records)...")
            try:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                combined_df.dropna(subset=['timestamp'], inplace=True)

                if not combined_df.empty: # Check again after potential drops
                    if combined_df['timestamp'].dt.tz is None:
                        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC')
                    else:
                        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC')

                    combined_df.sort_values(by=['product', 'timestamp'], inplace=True)
                    combined_df.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True)
                    combined_df['buy'] = pd.to_numeric(combined_df['buy'], errors='coerce')
                    combined_df['ask'] = pd.to_numeric(combined_df['ask'], errors='coerce')
                    logging.info(f"Final processed data size: {len(combined_df)} records.")
                else:
                    logging.warning("Combined DataFrame became empty after timestamp processing.")

            except Exception as e:
                logging.error(f"Error during combined data processing (timestamps, types, duplicates): {e}", exc_info=True)
                combined_df = pd.DataFrame() # Reset on error

        else:
            logging.warning("Combined data is empty before processing.")

        # --- Filter Products Based on Vendor Price ---
        all_products_initial = sorted(list(combined_df['product'].unique())) if not combined_df.empty else []
        if not all_products_initial:
             logging.warning("No products found in combined data to process.")
             filtered_products = []
        else:
            logging.info(f"Applying vendor price filter to {len(all_products_initial)} initial products...")
            filtered_products = [
                p for p in all_products_initial
                if p == "Bag Of 10 Cowbells" or (vp := vendor_prices.get(p)) is None or vp > 0
                # Condition: Keep if name is exception OR vendor price is missing OR vendor price is > 0
            ]
            logging.info(f"Filtered down to {len(filtered_products)} products.")


        # --- Calculate Trends, Volatility, and Indices for Filtered Products ---
        # Pass copies to calculation functions as they might modify timezone info or require specific states
        df_copy_for_calc = combined_df.copy() if not combined_df.empty else pd.DataFrame()
        # Only calculate for filtered products
        product_trends = calculate_trends(df_copy_for_calc.copy(), filtered_products)
        product_volatility = calculate_volatility(df_copy_for_calc.copy(), filtered_products, days=VOLATILITY_DAYS)
        # Indices are based on the trends of the filtered products
        market_indices = calculate_market_indices(product_trends, item_categories)

        # --- Generate JSON Data Files ---
        logging.info("--- Generating JSON Data Files ---")

        # 1. Market Summary (Iterate through FILTERED products)
        market_summary = []
        if not combined_df.empty:
            try:
                latest_data_map = combined_df.groupby('product').last()
                for product in filtered_products: # Use the filtered list here
                    if product in latest_data_map.index:
                        latest = latest_data_map.loc[product]
                        market_summary.append({
                            'name': product,
                            'category': item_categories.get(product, 'Unknown'),
                            'buy': latest['buy'] if pd.notna(latest['buy']) else None,
                            'ask': latest['ask'] if pd.notna(latest['ask']) else None,
                            'vendor': vendor_prices.get(product),
                            'trend': product_trends.get(product), # Already calculated only for filtered
                            f'volatility_{VOLATILITY_DAYS}d': product_volatility.get(product) # Already calculated only for filtered
                        })
                    else:
                         # This warning might appear if an item exists only in live_df but not historical,
                         # and gets filtered out before calculations. Should be rare.
                         logging.warning(f"Filtered product '{product}' not found in latest_data_map. Skipping summary entry.")
            except Exception as e:
                logging.error(f"Error creating market summary list: {e}", exc_info=True)
                market_summary = []

        summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(market_summary, f, allow_nan=False, default=str)
            logging.info(f"Saved market summary ({len(market_summary)} items) to {summary_path}")
        except (IOError, TypeError, ValueError) as e:
            logging.error(f"Error saving market summary JSON to {summary_path}: {e}", exc_info=True)


        # 2. Full Historical Data (Generate full, then filter output)
        nested_history_dict = {}
        if not combined_df.empty:
            logging.info("Grouping historical data by product for nested JSON...")
            try:
                # Ensure timestamp is timezone-aware UTC (should be already, but verify)
                if combined_df['timestamp'].dt.tz is None or str(combined_df['timestamp'].dt.tz) != 'UTC':
                     logging.warning("Timestamps in combined_df are not UTC timezone-aware before history generation. Re-converting...")
                     if combined_df['timestamp'].dt.tz is None:
                         combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC')
                     else:
                         combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC')

                grouped_history = combined_df.groupby('product')

                for product_name, group_df in grouped_history:
                    # Generate lists for all products initially
                    buy_data = group_df[['timestamp', 'buy']]
                    buy_list = [
                        {"timestamp": int(ts.timestamp()), "price": price if pd.notna(price) else None}
                        for ts, price in zip(buy_data['timestamp'], buy_data['buy'])
                        if pd.isna(price) or (pd.notna(price) and np.isfinite(price))
                    ]
                    ask_data = group_df[['timestamp', 'ask']]
                    ask_list = [
                        {"timestamp": int(ts.timestamp()), "price": price if pd.notna(price) else None}
                        for ts, price in zip(ask_data['timestamp'], ask_data['ask'])
                        if pd.isna(price) or (pd.notna(price) and np.isfinite(price))
                    ]
                    if buy_list or ask_list:
                         nested_history_dict[product_name] = {"buy": buy_list, "ask": ask_list}

            except Exception as e:
                logging.error(f"Error processing data for market history JSON: {e}", exc_info=True)
                nested_history_dict = {} # Reset on error

        # Filter the generated history dictionary based on filtered_products
        filtered_nested_history = {
            p: data for p, data in nested_history_dict.items() if p in filtered_products
        }

        # Save the FILTERED nested dictionary
        history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
        try:
            logging.info(f"Attempting to save FILTERED nested market history JSON ({len(filtered_nested_history)} products)...")
            with open(history_path, 'w') as f:
                json.dump(filtered_nested_history, f, allow_nan=False)
            logging.info(f"Saved filtered market history to {history_path}")
        except (IOError, TypeError, ValueError) as e:
            logging.error(f"Error saving market history JSON to {history_path}: {e}", exc_info=True)

        # 3. Market Indices JSON (Already based on filtered trends)
        indices_path = os.path.join(OUTPUT_DATA_DIR, 'market_indices.json')
        try:
            with open(indices_path, 'w') as f:
                json.dump(market_indices, f, allow_nan=False, default=str)
            logging.info(f"Saved market indices ({len(market_indices)} categories) to {indices_path}")
        except (IOError, TypeError, ValueError) as e:
            logging.error(f"Error saving market indices JSON to {indices_path}: {e}", exc_info=True)


        # --- Render HTML Template ---
        logging.info("--- Rendering HTML Template ---")
        try:
            unique_categories = sorted(list(set(item_categories.values()))) if item_categories else []

            if not os.path.isdir(TEMPLATE_DIR):
                 logging.error(f"Template directory '{TEMPLATE_DIR}' not found!")
            else:
                template_file_path = os.path.join(TEMPLATE_DIR, 'index.html')
                if not os.path.isfile(template_file_path):
                    logging.warning(f"HTML template file '{template_file_path}' not found! Skipping HTML rendering.")
                else:
                    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
                    template = env.get_template('index.html')
                    html_context = {
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'categories': unique_categories
                        }
                    html_content = template.render(html_context)
                    html_path = os.path.join(OUTPUT_DIR, 'index.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logging.info(f"Saved main HTML to {html_path}")

        except TemplateNotFound as e:
             logging.error(f"Jinja2 template not found: {e}", exc_info=True)
        except IOError as e:
             logging.error(f"IOError rendering or saving HTML template: {e}", exc_info=True)
        except Exception as e: # Catch other Jinja or file writing errors
            logging.error(f"Unexpected error rendering or saving HTML template: {e}", exc_info=True)

    except Exception as e: # Catch any unexpected error in the main execution flow
         logging.critical(f"An critical error occurred during the main build process: {e}", exc_info=True)
    finally:
        logging.info("--- Static Site Build Finished ---")


if __name__ == '__main__':
    main()

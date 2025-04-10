import os
import sqlite3
import json
import requests
import pandas as pd
import numpy as np # Import numpy for NaN/Inf checking/replacement
from datetime import datetime, timedelta
import logging
import math
import shutil
from jinja2 import Environment, FileSystemLoader # For rendering HTML template

# --- Configuration ---
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

# History and Trend Settings
HISTORICAL_DAYS = 30 # How much history to process
TREND_WINDOW_HOURS = 12 # +/- hours around 24h ago for trend calc

# --- Category Parsing ---
def parse_categories(filepath):
    """Parses the cata.txt file into a dictionary {item_name: category}."""
    categories = {}
    current_category = "Unknown"
    current_subcategory = None
    logging.info(f"Attempting to parse category file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line == '•':
                    continue
                # Check if line is a main category (heuristic: short, maybe all caps or single word)
                # This parsing is basic and might need adjustment based on exact file format nuances
                words = line.split()
                if len(words) < 3 and line.isupper(): # Simple check for main category header
                    current_category = line.title()
                    current_subcategory = None
                    logging.debug(f"Found Main Category: {current_category}")
                elif len(words) < 4 and line == line.title() and line not in categories: # Check for subcategory like "Main Hand"
                    # More robust check might be needed if item names can be short/titled
                    # Check if it looks like a header rather than an item name based on context
                    # For now, assume title case lines that aren't already items are subheaders
                     is_likely_item = any(char.isdigit() or char in "'’" for char in line) # Simple check if it might be an item name
                     if not is_likely_item:
                           current_subcategory = line
                           logging.debug(f"Found Sub Category: {current_subcategory}")
                     else: # Treat as item under current category
                          category_key = f"{current_category} / {current_subcategory}" if current_subcategory else current_category
                          categories[line] = category_key
                          logging.debug(f"Found Item (under main): {line} -> {category_key}")

                else: # Assume it's an item name
                    # Handle lines with '•' if items are separated differently
                    item_name = line.split('•')[0].strip()
                    if item_name:
                        category_key = f"{current_category} / {current_subcategory}" if current_subcategory else current_category
                        categories[item_name] = category_key
                        logging.debug(f"Found Item: {item_name} -> {category_key}")

        logging.info(f"Parsed {len(categories)} items from category file.")
        # Manual correction if needed
        if "Bag Of 10 Cowbells" in categories and categories["Bag Of 10 Cowbells"] == "Loots / Bag Of 10 Cowbells":
             categories["Bag Of 10 Cowbells"] = "Loots" # Example correction
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}. Categories will be missing.")
        return {}
    except Exception as e:
        logging.error(f"Error parsing category file {filepath}: {e}", exc_info=True)
        return {}


# --- Data Fetching ---
def download_file(url, local_path):
    # (download_file function remains the same)
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
        logging.error(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during download of {url}: {e}")
        return False

# --- Data Loading and Processing ---
def load_historical_data(days_to_load):
    # (load_historical_data function remains the same)
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}")
    if not os.path.exists(DB_PATH): logging.warning(f"{DB_PATH} not found."); return pd.DataFrame()
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp()
        ask_query = "SELECT * FROM ask WHERE time >= ?"; ask_df_wide = pd.read_sql_query(ask_query, conn, params=(cutoff_timestamp,))
        bid_query = "SELECT * FROM bid WHERE time >= ?"; bid_df_wide = pd.read_sql_query(bid_query, conn, params=(cutoff_timestamp,))
        conn.close(); conn = None
        logging.info(f"DB Query: Loaded {len(ask_df_wide)} wide ask, {len(bid_df_wide)} wide bid records.")
        if ask_df_wide.empty and bid_df_wide.empty: return pd.DataFrame()
        ask_df_long = pd.DataFrame()
        if not ask_df_wide.empty and 'time' in ask_df_wide.columns:
            item_columns_ask = [col for col in ask_df_wide.columns if col.lower() != 'time']
            if item_columns_ask:
                try:
                    logging.info(f"Melting 'ask' table ({len(ask_df_wide)} rows)...")
                    ask_df_long = ask_df_wide.melt(id_vars=['time'], value_vars=item_columns_ask, var_name='product', value_name='ask')
                    ask_df_long['timestamp'] = pd.to_datetime(ask_df_long['time'], unit='s', errors='coerce')
                    ask_df_long['ask'] = pd.to_numeric(ask_df_long['ask'], errors='coerce')
                    ask_df_long.drop(columns=['time'], inplace=True); ask_df_long.dropna(subset=['timestamp', 'ask'], inplace=True)
                    ask_df_long = ask_df_long[['product', 'ask', 'timestamp']]
                except Exception as melt_error: logging.error(f"Error melting 'ask' data: {melt_error}", exc_info=True)
            else: logging.warning("No item columns found in 'ask' table.")
        del ask_df_wide
        bid_df_long = pd.DataFrame()
        if not bid_df_wide.empty and 'time' in bid_df_wide.columns:
            item_columns_bid = [col for col in bid_df_wide.columns if col.lower() != 'time']
            if item_columns_bid:
                try:
                    logging.info(f"Melting 'bid' table ({len(bid_df_wide)} rows)...")
                    bid_df_long = bid_df_wide.melt(id_vars=['time'], value_vars=item_columns_bid, var_name='product', value_name='buy')
                    bid_df_long['timestamp'] = pd.to_datetime(bid_df_long['time'], unit='s', errors='coerce')
                    bid_df_long['buy'] = pd.to_numeric(bid_df_long['buy'], errors='coerce')
                    bid_df_long.drop(columns=['time'], inplace=True); bid_df_long.dropna(subset=['timestamp', 'buy'], inplace=True)
                    bid_df_long = bid_df_long[['product', 'buy', 'timestamp']]
                except Exception as melt_error: logging.error(f"Error melting 'bid' data: {melt_error}", exc_info=True)
            else: logging.warning("No item columns found in 'bid' table.")
        del bid_df_wide
        logging.info("Merging melted ask and bid data...")
        if not ask_df_long.empty and not bid_df_long.empty: merged_df = pd.merge(ask_df_long, bid_df_long, on=['product', 'timestamp'], how='outer')
        elif not ask_df_long.empty: merged_df = ask_df_long.copy(); merged_df['buy'] = pd.NA
        elif not bid_df_long.empty: merged_df = bid_df_long.copy(); merged_df['ask'] = pd.NA
        else: merged_df = pd.DataFrame()
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
        logging.error(f"Error loading/processing historical data: {e}", exc_info=True)
        if conn: conn.close()
        return pd.DataFrame()

def load_live_data():
    # (load_live_data function remains the same)
    logging.info(f"Loading live data from {JSON_PATH}")
    vendor_prices = {}; live_records_df = pd.DataFrame()
    if not os.path.exists(JSON_PATH): logging.warning(f"{JSON_PATH} not found."); return live_records_df, vendor_prices
    try:
        with open(JSON_PATH, 'r') as f: data = json.load(f)
        if 'market' not in data or not isinstance(data['market'], dict): logging.error("Invalid JSON structure."); return live_records_df, vendor_prices
        market_data = data['market']; records = []
        current_time = datetime.now()
        for product_name, price_info in market_data.items():
            if isinstance(price_info, dict) and 'ask' in price_info and 'bid' in price_info:
                 ask_price = pd.NA if price_info['ask'] == -1 else price_info['ask']
                 buy_price = pd.NA if price_info['bid'] == -1 else price_info['bid']
                 vendor_price = price_info.get('vendor', pd.NA)
                 records.append({'product': product_name, 'buy': buy_price, 'ask': ask_price, 'timestamp': current_time})
                 if vendor_price == -1 or vendor_price is None: vendor_prices[product_name] = None
                 else:
                     try: vendor_prices[product_name] = int(vendor_price)
                     except (ValueError, TypeError): vendor_prices[product_name] = None
            else: logging.warning(f"Skipping invalid JSON item: {product_name}")
        if records:
            live_records_df = pd.DataFrame(records)
            live_records_df['buy'] = pd.to_numeric(live_records_df['buy'], errors='coerce')
            live_records_df['ask'] = pd.to_numeric(live_records_df['ask'], errors='coerce')
            live_records_df['timestamp'] = pd.to_datetime(live_records_df['timestamp'], errors='coerce')
            live_records_df.dropna(subset=['timestamp'], inplace=True)
            live_records_df = live_records_df[['product', 'buy', 'ask', 'timestamp']]
        logging.info(f"Loaded {len(live_records_df)} live records and {len(vendor_prices)} vendor prices.")
        return live_records_df, vendor_prices
    except Exception as e: logging.error(f"Error loading live data: {e}", exc_info=True); return pd.DataFrame(), {}

def calculate_trends(df, products):
    # (calculate_trends function remains the same)
    trends = {}; processed_count = 0
    if df.empty or not products: return trends
    now = datetime.now(); yesterday = now - timedelta(hours=24)
    yesterday_start = yesterday - timedelta(hours=TREND_WINDOW_HOURS); yesterday_end = yesterday + timedelta(hours=TREND_WINDOW_HOURS)
    logging.info(f"Calculating trends for {len(products)} products relative to {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: return trends
    except Exception as e: logging.error(f"Error converting timestamp for trend calc: {e}"); return trends
    df['avg_price'] = df[['buy', 'ask']].mean(axis=1).fillna(df['buy']).fillna(df['ask'])
    relevant_past_data = df[df['timestamp'] <= yesterday_end].copy()
    grouped = df.groupby('product')
    relevant_past_grouped = relevant_past_data.groupby('product')
    for product in products:
        try:
            product_group = grouped.get_group(product)
            if product_group.empty: continue
            latest_data = product_group.iloc[-1]; current_price = latest_data['avg_price']
            if pd.isna(current_price): continue
            if product not in relevant_past_grouped.groups: continue
            past_product_df = relevant_past_grouped.get_group(product)
            past_window_df = past_product_df[(past_product_df['timestamp'] >= yesterday_start)]
            if past_window_df.empty: continue
            time_diff_series = (past_window_df['timestamp'] - yesterday).abs()
            if time_diff_series.empty: continue
            closest_index_label = time_diff_series.idxmin()
            closest_past_data = past_window_df.loc[closest_index_label]
            previous_price = closest_past_data['avg_price']
            if pd.isna(previous_price): continue
            if previous_price != 0:
                 change_pct = ((current_price - previous_price) / previous_price) * 100; trends[product] = change_pct
                 processed_count += 1
            else: trends[product] = None
        except KeyError: continue
        except Exception as e: logging.error(f"Error calculating trend for {product}: {e}", exc_info=True); continue
    logging.info(f"Finished trend calculation. Calculated trends for {processed_count} out of {len(products)} products.")
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in trends.items()}

def main():
    """Main build process."""
    logging.info("--- Starting Static Site Build ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

    # --- Parse Categories ---
    item_categories = parse_categories(CATEGORY_FILE_PATH)
    if not item_categories:
        logging.warning("Category data is empty or failed to load. Categories will be missing in output.")

    # --- Download and Load Data ---
    db_ok = download_file(DB_URL, DB_PATH)
    json_ok = download_file(JSON_URL, JSON_PATH)
    if not json_ok: logging.error("Failed to download milkyapi.json. Cannot proceed."); return
    if not db_ok: logging.warning("Failed to download market.db. Historical data may be missing.")

    historical_df = load_historical_data(days_to_load=HISTORICAL_DAYS)
    live_df, vendor_prices = load_live_data()

    if historical_df.empty and live_df.empty:
         logging.error("Both historical and live data are empty. Cannot build site.")
         combined_df = pd.DataFrame()
    else:
         logging.info("Concatenating historical and live data...")
         combined_df = pd.concat([historical_df, live_df], ignore_index=True)

    if not combined_df.empty:
        logging.info(f"Processing combined data ({len(combined_df)} records)...")
        combined_df.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True)
        combined_df.sort_values(by=['product', 'timestamp'], inplace=True)
        combined_df['buy'] = pd.to_numeric(combined_df['buy'], errors='coerce')
        combined_df['ask'] = pd.to_numeric(combined_df['ask'], errors='coerce')
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
        combined_df.dropna(subset=['timestamp'], inplace=True)
        logging.info(f"Final processed data size: {len(combined_df)} records.")
    else:
        logging.warning("Combined data is empty after processing.")

    all_products = sorted(list(combined_df['product'].unique())) if not combined_df.empty else []
    product_trends = calculate_trends(combined_df.copy(), all_products) # Use copy

    # --- Generate JSON Data Files ---
    logging.info("Generating JSON data files...")

    # 1. Market Summary (Includes category now)
    market_summary = []
    if not combined_df.empty:
        latest_data_map = combined_df.groupby('product').last()
        for product in all_products:
            if product in latest_data_map.index:
                latest = latest_data_map.loc[product]
                market_summary.append({
                    'name': product,
                    'category': item_categories.get(product, 'Unknown'), # Add category
                    'buy': latest['buy'] if pd.notna(latest['buy']) else None,
                    'ask': latest['ask'] if pd.notna(latest['ask']) else None,
                    'vendor': vendor_prices.get(product), # Already None if missing
                    'trend': product_trends.get(product) # Already None if missing/invalid
                })
    summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f, allow_nan=False)
        logging.info(f"Saved market summary to {summary_path}")
    except ValueError as ve: logging.error(f"ValueError saving market summary JSON: {ve}")
    except Exception as e: logging.error(f"Failed to save market summary JSON: {e}")


    # 2. Full Historical Data
    if not combined_df.empty:
        chart_history_data = combined_df.copy()
        chart_history_data['timestamp'] = chart_history_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        for col in ['buy', 'ask']:
            chart_history_data[col] = chart_history_data[col].replace([np.inf, -np.inf], np.nan)
            chart_history_data[col] = chart_history_data[col].astype(object)
            chart_history_data.loc[chart_history_data[col].isna(), col] = None
        chart_history_list = chart_history_data[['product', 'timestamp', 'buy', 'ask']].to_dict(orient='records')
        history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
        try:
            logging.info(f"Attempting to save market history JSON ({len(chart_history_list)} records)...")
            with open(history_path, 'w') as f:
                json.dump(chart_history_list, f, allow_nan=False)
            logging.info(f"Saved full market history to {history_path}")
        except ValueError as ve:
            logging.error(f"ValueError saving market history JSON: {ve}")
            problematic_sample = []
            for i, record in enumerate(chart_history_list):
                try: json.dumps(record, allow_nan=False)
                except ValueError: problematic_sample.append({'index': i, 'record': record});
                if len(problematic_sample) >= 5: break
            logging.error(f"Problematic records sample: {problematic_sample}")
        except Exception as e: logging.error(f"Failed to save market history JSON: {e}")
    else: logging.warning("Skipping market history JSON generation.")

    # --- Render HTML Template ---
    logging.info("Rendering HTML template...")
    try:
        # Get unique categories for the filter dropdown
        unique_categories = sorted(list(set(item_categories.values()))) if item_categories else []

        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('index.html')
        html_context = {
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categories': unique_categories # Pass categories to template
            }
        html_content = template.render(html_context)
        html_path = os.path.join(OUTPUT_DIR, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f: f.write(html_content)
        logging.info(f"Saved main HTML to {html_path}")
    except Exception as e: logging.error(f"Failed to render or save HTML template: {e}", exc_info=True)

    logging.info("--- Static Site Build Finished ---")

if __name__ == '__main__':
    main()

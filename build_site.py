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
from jinja2 import Environment, FileSystemLoader # For rendering HTML template

# --- Configuration --
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
TEMPLATE_DIR = "templates" # Directory for Jinja templates - Added for clarity

# History and Trend Settings
HISTORICAL_DAYS = 30 # How much history to process
TREND_WINDOW_HOURS = 12 # +/- hours around 24h ago for trend calc

# --- Category Parsing (Revised v3 - Based on new cata.txt) ---
# (parse_categories function remains the same as provided by user)
def parse_categories(filepath):
    """Parses the cata.txt file into a dictionary {item_name: category}."""
    categories = {}
    current_main_category = "Unknown"
    current_sub_category = None
    current_display_category = "Unknown" # Category name to assign to items

    # Define known main categories and subcategories from the new file structure
    known_main_categories = [
        "Currencies", "Loots", "Resources", "Consumables", "Books", "Keys",
        "Equipment", "Jewelry", "Trinket", "Tools"
    ]
    equipment_subcategories = [
        "Main Hand", "Off Hand", "Head", "Body", "Legs", "Hands", # Renamed from Gloves
        "Feet", "Back", "Pouch"
    ]
    # Note: Jewelry is now a main category according to the file structure
    tool_subcategories = [
        "Milking", "Foraging", "Woodcutting", "Cheesesmithing", "Crafting",
        "Tailoring", "Cooking", "Brewing", "Alchemy", "Enhancing"
    ]

    logging.info(f"Attempting to parse category file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and potential separators
                if not line or line == '•':
                    continue

                logging.debug(f"Processing Line {line_num}: '{line}'")

                # Check if it's a known main category
                if line in known_main_categories:
                    current_main_category = line
                    current_sub_category = None # Reset subcategory
                    current_display_category = current_main_category # Assign main category by default
                    logging.debug(f"  -> Matched Main Category: {current_main_category}")
                    continue # Move to next line

                # Check if it's an equipment subcategory
                if current_main_category == "Equipment" and line in equipment_subcategories:
                    current_sub_category = line
                    current_display_category = f"Equipment / {current_sub_category}"
                    logging.debug(f"  -> Matched Equipment Sub Category: {current_sub_category} -> Display: {current_display_category}")
                    continue # Move to next line

                # Check if it's a tool subcategory
                if current_main_category == "Tools" and line in tool_subcategories:
                    current_sub_category = line
                    current_display_category = f"Tools / {current_sub_category}"
                    logging.debug(f"  -> Matched Tool Sub Category: {current_sub_category} -> Display: {current_display_category}")
                    continue # Move to next line

                # Otherwise, assume it's a line containing item(s)
                # Split by comma, strip whitespace from each item
                item_names = [name.strip() for name in line.split(',') if name.strip()]

                if not item_names:
                    logging.warning(f"  -> Line {line_num} parsed as empty item list: '{line}'")
                    continue

                for item_name in item_names:
                    if item_name: # Ensure not empty after stripping
                        # Assign the current display category
                        # Handle edge case where an item might appear before any category header
                        if current_display_category == "Unknown" and current_main_category != "Unknown":
                             current_display_category = current_main_category

                        categories[item_name] = current_display_category
                        logging.debug(f"  -> Found Item: '{item_name}' -> '{current_display_category}'")
                    else:
                         logging.warning(f"  -> Found empty item name after split/strip on line {line_num}: '{line}'")


        logging.info(f"Parsed {len(categories)} items from category file.")
        # Log a sample for verification
        sample_items = list(categories.items())[:5] + list(categories.items())[-5:]
        logging.debug(f"Category parsing sample (first/last 5): {sample_items}")
        # Log specific items if needed for debugging
        logging.debug(f"Category for 'Necklace Of Efficiency': {categories.get('Necklace Of Efficiency')}")
        logging.debug(f"Category for 'Cheese Brush': {categories.get('Cheese Brush')}")
        logging.debug(f"Category for 'Rough Hide': {categories.get('Rough Hide')}")
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}. Categories will be missing.")
        return {}
    except Exception as e:
        logging.error(f"Error parsing category file {filepath}: {e}", exc_info=True)
        return {}


# --- Data Fetching ---
# (download_file function remains the same as provided by user)
def download_file(url, local_path):
    logging.info(f"Attempting to download {url} to {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        logging.info(f"Successfully downloaded {local_path}")
        return True
    except Exception as e: logging.error(f"Error downloading {url}: {e}"); return False

# --- Data Loading and Processing ---
# (load_historical_data function remains the same as provided by user)
def load_historical_data(days_to_load):
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

# (load_live_data function remains the same as provided by user)
def load_live_data():
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

# (calculate_trends function remains the same as provided by user)
def calculate_trends(df, products):
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
                    'category': item_categories.get(product, 'Unknown'),
                    'buy': latest['buy'] if pd.notna(latest['buy']) else None,
                    'ask': latest['ask'] if pd.notna(latest['ask']) else None,
                    'vendor': vendor_prices.get(product),
                    'trend': product_trends.get(product)
                })
            else:
                 logging.warning(f"Product '{product}' not found in latest_data_map despite being in unique list.")

    summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f, allow_nan=False)
        logging.info(f"Saved market summary to {summary_path}")
    except ValueError as ve: logging.error(f"ValueError saving market summary JSON: {ve}")
    except Exception as e: logging.error(f"Failed to save market summary JSON: {e}")


    # 2. Full Historical Data - MODIFIED TO OUTPUT NESTED OBJECT
    nested_history_dict = {}
    if not combined_df.empty:
        logging.info("Grouping historical data by product for nested JSON...")
        # Ensure timestamp is datetime for calculations
        if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
             combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
             combined_df.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed

        # Convert timestamp to UTC if it doesn't have timezone info
        # This ensures consistent epoch seconds regardless of server timezone
        if combined_df['timestamp'].dt.tz is None:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('UTC')
        else:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert('UTC')

        # Group by product
        grouped_history = combined_df.groupby('product')

        for product_name, group_df in grouped_history:
            buy_list = []
            ask_list = []

            # Process buy data for the product
            buy_data = group_df[['timestamp', 'buy']].dropna(subset=['buy'])
            if not buy_data.empty:
                 buy_list = [
                     {"timestamp": int(ts.timestamp()), "price": price}
                     for ts, price in zip(buy_data['timestamp'], buy_data['buy'])
                     if pd.notna(price) and np.isfinite(price) # Ensure price is valid number
                 ]

            # Process ask data for the product
            ask_data = group_df[['timestamp', 'ask']].dropna(subset=['ask'])
            if not ask_data.empty:
                 ask_list = [
                     {"timestamp": int(ts.timestamp()), "price": price}
                     for ts, price in zip(ask_data['timestamp'], ask_data['ask'])
                     if pd.notna(price) and np.isfinite(price) # Ensure price is valid number
                 ]

            # Only add product if it has at least one buy or ask point
            if buy_list or ask_list:
                 nested_history_dict[product_name] = {"buy": buy_list, "ask": ask_list}

    # Save the nested dictionary
    history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
    try:
        logging.info(f"Attempting to save nested market history JSON ({len(nested_history_dict)} products)...")
        with open(history_path, 'w') as f:
            json.dump(nested_history_dict, f, allow_nan=False) # allow_nan=False handles residual issues
        logging.info(f"Saved nested market history to {history_path}")
    except ValueError as ve:
        logging.error(f"ValueError saving market history JSON: {ve}")
        # Add more detailed logging for complex structures if needed
    except Exception as e:
        logging.error(f"Failed to save market history JSON: {e}")


    # --- Render HTML Template ---
    logging.info("Rendering HTML template...")
    try:
        # Get unique categories for the filter dropdown FROM THE PARSED DATA
        unique_categories = sorted(list(set(item_categories.values()))) if item_categories else []

        # Ensure TEMPLATE_DIR is defined correctly
        if not os.path.isdir(TEMPLATE_DIR):
             logging.error(f"Template directory '{TEMPLATE_DIR}' not found!")
             return # Stop if template dir is missing

        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR)) # Use TEMPLATE_DIR
        template = env.get_template('index.html')
        html_context = {
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categories': unique_categories # Pass parsed categories
            }
        html_content = template.render(html_context)
        html_path = os.path.join(OUTPUT_DIR, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f: f.write(html_content)
        logging.info(f"Saved main HTML to {html_path}")
    except Exception as e: logging.error(f"Failed to render or save HTML template: {e}", exc_info=True)

    logging.info("--- Static Site Build Finished ---")

if __name__ == '__main__':
    main()

import os
import sqlite3
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import math
import shutil # For copying static assets if needed
from jinja2 import Environment, FileSystemLoader # For rendering HTML template

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Source Data URLs and Paths
DB_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/market.db"
JSON_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/milkyapi.json"
DB_PATH = "market.db" # Downloaded file paths
JSON_PATH = "milkyapi.json"

# Output directory for static files
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data") # Subdir for data files

# History and Trend Settings
HISTORICAL_DAYS = 30 # How much history to process
TREND_WINDOW_HOURS = 12 # +/- hours around 24h ago for trend calc

# --- Data Fetching ---
def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    logging.info(f"Attempting to download {url} to {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=60) # Increased timeout slightly
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
    """Loads historical data from the SQLite database (wide format)."""
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.warning(f"{DB_PATH} not found.")
        return pd.DataFrame()
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp()
        ask_query = "SELECT * FROM ask WHERE time >= ?"
        ask_df_wide = pd.read_sql_query(ask_query, conn, params=(cutoff_timestamp,))
        bid_query = "SELECT * FROM bid WHERE time >= ?"
        bid_df_wide = pd.read_sql_query(bid_query, conn, params=(cutoff_timestamp,))
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
                    ask_df_long.drop(columns=['time'], inplace=True)
                    ask_df_long.dropna(subset=['timestamp', 'ask'], inplace=True)
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
                    bid_df_long.drop(columns=['time'], inplace=True)
                    bid_df_long.dropna(subset=['timestamp', 'buy'], inplace=True)
                    bid_df_long = bid_df_long[['product', 'buy', 'timestamp']]
                except Exception as melt_error: logging.error(f"Error melting 'bid' data: {melt_error}", exc_info=True)
            else: logging.warning("No item columns found in 'bid' table.")
        del bid_df_wide

        logging.info("Merging melted ask and bid data...")
        if not ask_df_long.empty and not bid_df_long.empty:
            merged_df = pd.merge(ask_df_long, bid_df_long, on=['product', 'timestamp'], how='outer')
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
    """Loads live data from JSON and extracts vendor prices."""
    logging.info(f"Loading live data from {JSON_PATH}")
    vendor_prices = {}
    live_records_df = pd.DataFrame()
    if not os.path.exists(JSON_PATH):
        logging.warning(f"{JSON_PATH} not found.")
        return live_records_df, vendor_prices
    try:
        with open(JSON_PATH, 'r') as f: data = json.load(f)
        if 'market' not in data or not isinstance(data['market'], dict):
             logging.error("Invalid JSON structure."); return live_records_df, vendor_prices

        market_data = data['market']; records = []
        current_time = datetime.now()
        for product_name, price_info in market_data.items():
            if isinstance(price_info, dict) and 'ask' in price_info and 'bid' in price_info:
                 ask_price = pd.NA if price_info['ask'] == -1 else price_info['ask']
                 buy_price = pd.NA if price_info['bid'] == -1 else price_info['bid']
                 vendor_price = price_info.get('vendor', pd.NA)
                 records.append({'product': product_name, 'buy': buy_price, 'ask': ask_price, 'timestamp': current_time})
                 # Store vendor price (handle -1/None)
                 if vendor_price == -1 or vendor_price is None: vendor_prices[product_name] = None
                 else:
                     try: vendor_prices[product_name] = int(vendor_price) # Assuming integer vendor price
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
    except Exception as e:
        logging.error(f"Error loading live data: {e}", exc_info=True)
        return pd.DataFrame(), {} # Return empty on error

def calculate_trends(df, products):
    """Calculates 24h % change for a list of products."""
    trends = {}
    if df.empty or not products: return trends
    now = datetime.now(); yesterday = now - timedelta(hours=24)
    yesterday_start = yesterday - timedelta(hours=TREND_WINDOW_HOURS); yesterday_end = yesterday + timedelta(hours=TREND_WINDOW_HOURS)
    logging.info(f"Calculating trends for {len(products)} products relative to {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: return trends
    except Exception as e: logging.error(f"Error converting timestamp for trend calc: {e}"); return trends
    df['avg_price'] = df[['buy', 'ask']].mean(axis=1)
    df['avg_price'] = df['avg_price'].fillna(df['buy']).fillna(df['ask'])
    relevant_past_data = df[df['timestamp'] <= yesterday_end].copy()
    grouped = df.groupby('product')
    relevant_past_grouped = relevant_past_data.groupby('product')
    processed_count = 0
    for product in products:
        try:
            product_group = grouped.get_group(product)
            if product_group.empty: continue
            latest_data = product_group.iloc[-1]
            current_price = latest_data['avg_price']
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
    # Convert NaN trends to None for JSON compatibility before returning
    return {k: (v if pd.notna(v) else None) for k, v in trends.items()}

def main():
    """Main build process."""
    logging.info("--- Starting Static Site Build ---")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

    # Download source files
    db_ok = download_file(DB_URL, DB_PATH)
    json_ok = download_file(JSON_URL, JSON_PATH)

    if not db_ok: logging.warning("Failed to download market.db. Historical data may be missing.")
    if not json_ok: logging.error("Failed to download milkyapi.json. Cannot proceed without live data."); return

    # Load data
    historical_df = load_historical_data(days_to_load=HISTORICAL_DAYS)
    live_df, vendor_prices = load_live_data()

    # Combine data
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

    # Calculate trends for all products
    all_products = sorted(list(combined_df['product'].unique())) if not combined_df.empty else []
    product_trends = calculate_trends(combined_df.copy(), all_products) # Use copy

    # --- Generate JSON Data Files ---
    logging.info("Generating JSON data files...")

    # 1. Market Summary (for list view, search, sort)
    market_summary = []
    if not combined_df.empty:
        # Get latest entry for each product
        latest_data_map = combined_df.groupby('product').last()
        for product in all_products: # Iterate in sorted order
            if product in latest_data_map.index:
                latest = latest_data_map.loc[product]
                market_summary.append({
                    'name': product,
                    # Convert NA to None for JSON
                    'buy': latest['buy'] if pd.notna(latest['buy']) else None,
                    'ask': latest['ask'] if pd.notna(latest['ask']) else None,
                    'vendor': vendor_prices.get(product), # Already None if missing/invalid
                    'trend': product_trends.get(product) # Already None if missing/invalid
                })

    summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
    try:
        with open(summary_path, 'w') as f:
            json.dump(market_summary, f) # Save as list of objects
        logging.info(f"Saved market summary to {summary_path}")
    except Exception as e:
        logging.error(f"Failed to save market summary JSON: {e}")

    # 2. Full Historical Data (for client-side charting)
    # Convert entire dataframe to list of records for JSON
    if not combined_df.empty:
        chart_history_data = combined_df.copy()
        # Convert NA to None and timestamp to string
        chart_history_data['buy'] = chart_history_data['buy'].where(pd.notna, None)
        chart_history_data['ask'] = chart_history_data['ask'].where(pd.notna, None)
        chart_history_data['timestamp'] = chart_history_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        chart_history_list = chart_history_data.to_dict(orient='records')

        history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
        try:
            with open(history_path, 'w') as f:
                json.dump(chart_history_list, f)
            logging.info(f"Saved full market history to {history_path}")
        except Exception as e:
            logging.error(f"Failed to save market history JSON: {e}")
    else:
        logging.warning("Skipping market history JSON generation as data is empty.")


    # --- Render HTML Template ---
    logging.info("Rendering HTML template...")
    try:
        # Assumes 'templates/index.html' exists relative to this script
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('index.html')
        # Pass only minimal necessary data, JS will load the rest
        html_context = {
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        html_content = template.render(html_context)
        html_path = os.path.join(OUTPUT_DIR, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Saved main HTML to {html_path}")
    except Exception as e:
        logging.error(f"Failed to render or save HTML template: {e}", exc_info=True)

    # --- Copy Static Assets (Optional) ---
    # If you have CSS files, etc., copy them to OUTPUT_DIR
    # Example: shutil.copyfile('style.css', os.path.join(OUTPUT_DIR, 'style.css'))

    logging.info("--- Static Site Build Finished ---")

if __name__ == '__main__':
    main()

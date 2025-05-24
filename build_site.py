import os
import sqlite3
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from collections import defaultdict # For grouping trends by category
from jinja2 import Environment, FileSystemLoader, TemplateNotFound # For rendering HTML template

# --- Configuration --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s') # Added lineno

# Source Data URLs and Paths
DB_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/market.db"
JSON_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/milkyapi.json"
DB_PATH = "market.db"
JSON_PATH = "milkyapi.json"
CATEGORY_FILE_PATH = "cata.txt"

# Output directory for static files
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
TEMPLATE_DIR = "templates"

# History and Trend Settings
HISTORICAL_DAYS = 365 * 5 # For raw market_history.json (line charts) / general trends
HISTORICAL_DAYS_FOR_OHLCV = 90 # Shorter period for OHLCV (e.g., 90 days)
VOLATILITY_DAYS = 7
CANDLESTICK_INTERVALS = ['1h', '4h', '1D', '7D']

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
            for line_num, line in enumerate(f, 1):
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
                if not item_names:
                    continue

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
def load_historical_data(days_to_load=0, for_ohlcv=False):
    load_period_desc = f"up to {days_to_load} days" if days_to_load > 0 else "ALL"
    purpose_desc = "for OHLCV" if for_ohlcv else "for general history"
    logging.info(f"Loading {load_period_desc} historical data {purpose_desc} from {DB_PATH}")

    if not os.path.exists(DB_PATH):
        logging.warning(f"{DB_PATH} not found ({purpose_desc}).")
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            if days_to_load > 0:
                cutoff_timestamp = (datetime.now(timezone.utc) - timedelta(days=days_to_load)).timestamp()
                ask_query = "SELECT * FROM ask WHERE time >= ?"
                bid_query = "SELECT * FROM bid WHERE time >= ?"
                ask_df_wide = pd.read_sql_query(ask_query, conn, params=(cutoff_timestamp,))
                bid_df_wide = pd.read_sql_query(bid_query, conn, params=(cutoff_timestamp,))
            else: # Load all data (or effectively all if HISTORICAL_DAYS is large)
                ask_query = "SELECT * FROM ask"
                bid_query = "SELECT * FROM bid"
                ask_df_wide = pd.read_sql_query(ask_query, conn)
                bid_df_wide = pd.read_sql_query(bid_query, conn)

        logging.info(f"DB Query ({purpose_desc}): Loaded {len(ask_df_wide)} wide ask, {len(bid_df_wide)} wide bid records.")
        if ask_df_wide.empty and bid_df_wide.empty:
            return pd.DataFrame()

        ask_df_long = pd.DataFrame()
        if not ask_df_wide.empty and 'time' in ask_df_wide.columns:
            item_columns_ask = [col for col in ask_df_wide.columns if col.lower() != 'time']
            if item_columns_ask:
                try:
                    ask_df_long = ask_df_wide.melt(id_vars=['time'], value_vars=item_columns_ask, var_name='product', value_name='ask')
                    ask_df_long['timestamp'] = pd.to_datetime(ask_df_long['time'], unit='s', errors='coerce').dt.tz_localize('UTC') # Localize to UTC immediately
                    ask_df_long['ask'] = pd.to_numeric(ask_df_long['ask'], errors='coerce').replace(-1, pd.NA)
                    ask_df_long.drop(columns=['time'], inplace=True)
                    ask_df_long.dropna(subset=['timestamp', 'product'], inplace=True)
                    ask_df_long = ask_df_long[['product', 'ask', 'timestamp']]
                except Exception as e: logging.error(f"Error processing 'ask' data ({purpose_desc}): {e}", exc_info=True)
            else: logging.warning(f"No item columns found in 'ask' table ({purpose_desc}).")
        del ask_df_wide

        bid_df_long = pd.DataFrame()
        if not bid_df_wide.empty and 'time' in bid_df_wide.columns:
            item_columns_bid = [col for col in bid_df_wide.columns if col.lower() != 'time']
            if item_columns_bid:
                try:
                    bid_df_long = bid_df_wide.melt(id_vars=['time'], value_vars=item_columns_bid, var_name='product', value_name='buy')
                    bid_df_long['timestamp'] = pd.to_datetime(bid_df_long['time'], unit='s', errors='coerce').dt.tz_localize('UTC') # Localize to UTC
                    bid_df_long['buy'] = pd.to_numeric(bid_df_long['buy'], errors='coerce').replace(-1, pd.NA)
                    bid_df_long.drop(columns=['time'], inplace=True)
                    bid_df_long.dropna(subset=['timestamp', 'product'], inplace=True)
                    bid_df_long = bid_df_long[['product', 'buy', 'timestamp']]
                except Exception as e: logging.error(f"Error processing 'bid' data ({purpose_desc}): {e}", exc_info=True)
            else: logging.warning(f"No item columns found in 'bid' table ({purpose_desc}).")
        del bid_df_wide

        if not ask_df_long.empty and not bid_df_long.empty:
            merged_df = pd.merge(ask_df_long, bid_df_long, on=['product', 'timestamp'], how='outer')
        elif not ask_df_long.empty:
            merged_df = ask_df_long.copy(); merged_df['buy'] = pd.NA
        elif not bid_df_long.empty:
            merged_df = bid_df_long.copy(); merged_df['ask'] = pd.NA
        else:
            return pd.DataFrame() # Both were empty
        del ask_df_long, bid_df_long

        if not merged_df.empty:
            # Timestamps should already be UTC from above
            merged_df.sort_values(by=['product', 'timestamp'], inplace=True)
            final_cols = ['product', 'buy', 'ask', 'timestamp'] # Ensure consistent column order
            for col in final_cols:
                if col not in merged_df.columns: merged_df[col] = pd.NA
            merged_df = merged_df[final_cols]
        logging.info(f"Finished historical data processing ({purpose_desc}). Records: {len(merged_df)}")
        return merged_df

    except sqlite3.Error as e:
        logging.error(f"Database error accessing {DB_PATH} ({purpose_desc}): {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error reading {DB_PATH} ({purpose_desc}): {e}", exc_info=True)
        return pd.DataFrame()

def load_live_data():
    """Loads live market data and vendor prices from the JSON file."""
    logging.info(f"Loading live data from {JSON_PATH}")
    vendor_prices = {}
    live_records = []
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
        current_time_utc = datetime.now(timezone.utc) # Use timezone-aware timestamp

        for product_name, price_info in market_data.items():
            if isinstance(price_info, dict) and 'ask' in price_info and 'bid' in price_info:
                ask_price = pd.NA if price_info['ask'] == -1 else price_info['ask']
                buy_price = pd.NA if price_info['bid'] == -1 else price_info['bid']
                vendor_price = price_info.get('vendor', pd.NA)

                live_records.append({
                    'product': product_name,
                    'buy': buy_price,
                    'ask': ask_price,
                    'timestamp': current_time_utc # Assign UTC timestamp
                })

                if vendor_price == -1 or vendor_price is None or pd.isna(vendor_price):
                    vendor_prices[product_name] = None
                else:
                    try:
                        vendor_prices[product_name] = int(vendor_price)
                    except (ValueError, TypeError):
                        vendor_prices[product_name] = None
            # else: logging.debug(f"Skipping invalid/incomplete market data item in JSON: '{product_name}'") # Too verbose for INFO

        live_records_df = pd.DataFrame(live_records)
        if not live_records_df.empty:
            live_records_df['buy'] = pd.to_numeric(live_records_df['buy'], errors='coerce')
            live_records_df['ask'] = pd.to_numeric(live_records_df['ask'], errors='coerce')
            # Timestamp is already datetime and UTC
            live_records_df = live_records_df[['product', 'buy', 'ask', 'timestamp']]

        logging.info(f"Loaded {len(live_records_df)} live records and {len(vendor_prices)} vendor prices.")
        return live_records_df, vendor_prices
    except Exception as e:
        logging.error(f"Unexpected error loading live data from {JSON_PATH}: {e}", exc_info=True)
        return pd.DataFrame(), {}

# --- Trend Calculation ---
def calculate_trends(df, products):
    trends = {}
    if df.empty or not products: return trends
    now = datetime.now(timezone.utc)
    twenty_four_hours_ago = now - timedelta(hours=24)
    seven_days_ago = now - timedelta(days=7)
    logging.info(f"Calculating trends for {len(products)} products...")

    # Ensure timestamp is UTC (should be already if processed correctly)
    if df['timestamp'].dt.tz != timezone.utc:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    grouped = df.groupby('product')
    processed_count = 0
    for product in products:
        try:
            product_group = grouped.get_group(product)
            if product_group.empty: continue
            latest_data = product_group.iloc[-1]
            current_price = latest_data['ask']
            if pd.isna(current_price): continue

            previous_price = pd.NA
            df_last_24h = product_group[product_group['timestamp'] >= twenty_four_hours_ago]
            valid_ask_24h = df_last_24h['ask'].dropna()
            if len(valid_ask_24h) > 1: previous_price = valid_ask_24h.mean()
            elif len(valid_ask_24h) == 1: previous_price = valid_ask_24h.iloc[0]
            else:
                df_last_7d = product_group[product_group['timestamp'] >= seven_days_ago]
                valid_ask_7d = df_last_7d['ask'].dropna()
                if len(valid_ask_7d) > 0: previous_price = valid_ask_7d.mean()

            if pd.notna(previous_price):
                if previous_price != 0:
                    trends[product] = ((current_price - previous_price) / previous_price) * 100
                else: trends[product] = float('inf') if current_price > 0 else 0.0
                processed_count +=1
        except KeyError: continue # Product not in df
        except Exception as e: logging.error(f"Trend error for {product}: {e}")
    logging.info(f"Finished trend calculation. Trends for {processed_count} products.")
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in trends.items()}


# --- Volatility & CV Calculation ---
def calculate_volatility_and_cv(df, products, days=7):
    results = {}
    if df.empty or not products: return results
    logging.info(f"Calculating {days}-day volatility & CV for {len(products)} products...")
    now = datetime.now(timezone.utc)
    time_cutoff = now - timedelta(days=days)

    if df['timestamp'].dt.tz != timezone.utc:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    df_period = df[df['timestamp'] >= time_cutoff]
    if df_period.empty: return results
    
    grouped = df_period.groupby('product')['ask'] # Group by product and select 'ask' series
    processed_count = 0
    for product_name, ask_series in grouped:
        if product_name not in products: continue # Only for filtered products
        valid_ask_period = ask_series.dropna()
        if len(valid_ask_period) >= 2:
            std_dev = valid_ask_period.std(ddof=1)
            mean_price = valid_ask_period.mean()
            cv_percent = (std_dev / mean_price * 100) if mean_price and mean_price != 0 else None
            results[product_name] = {
                'std': std_dev if pd.notna(std_dev) and np.isfinite(std_dev) else None,
                'cv': cv_percent if pd.notna(cv_percent) and np.isfinite(cv_percent) else None
            }
            processed_count +=1
    logging.info(f"Finished volatility/CV. Stats for {processed_count} products.")
    return results

# --- Market Index Calculation ---
def calculate_market_indices(product_trends, item_categories):
    if not product_trends or not item_categories: return {}
    logging.info("Calculating market indices by category...")
    category_trends = defaultdict(list)
    for product, trend_val in product_trends.items():
        if trend_val is not None: # Trend must be valid
            category = item_categories.get(product)
            if category and category != "Unknown":
                category_trends[category].append(trend_val)
    
    market_indices = {
        cat: (sum(trends) / len(trends)) if trends else None
        for cat, trends in category_trends.items()
    }
    logging.info(f"Calculated market indices for {len(market_indices)} categories.")
    return {k: (v if pd.notna(v) and np.isfinite(v) else None) for k, v in market_indices.items()}


# --- OPTIMIZED OHLCV Calculation ---
def calculate_ohlcv_data_optimized(df_source, products, intervals):
    ohlcv_results = defaultdict(lambda: defaultdict(list))
    if df_source.empty or not products or not intervals:
        logging.warning("OHLCV: Source DataFrame or products/intervals list is empty.")
        return ohlcv_results

    logging.info(f"OHLCV: Starting calculation for {len(products)} products, intervals: {intervals}.")
    df = df_source.copy() # Work on a copy

    # Ensure timestamp is datetime64[ns, UTC] and 'ask' is numeric
    if not pd.api.types.is_datetime64_ns_dtype(df['timestamp']) or df['timestamp'].dt.tz != timezone.utc:
        logging.debug("OHLCV: Timestamp column re-processing for UTC.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_convert('UTC')
    df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
    df.dropna(subset=['timestamp', 'product', 'ask'], inplace=True)

    if df.empty:
        logging.warning("OHLCV: DataFrame empty after initial cleaning.")
        return ohlcv_results

    df_filtered = df[df['product'].isin(products)].set_index('timestamp')
    if df_filtered.empty:
        logging.warning("OHLCV: No data for specified products after filtering.")
        return ohlcv_results
    
    logging.info(f"OHLCV: Filtered data size for OHLCV: {len(df_filtered)} records.")
    grouped_by_product = df_filtered.groupby('product')['ask']
    
    product_count = 0
    total_products_with_data = len(grouped_by_product)

    for product_name, ask_series in grouped_by_product:
        product_count += 1
        if product_count % 50 == 0 or product_count == 1 or product_count == total_products_with_data:
             logging.info(f"OHLCV: Processing product {product_count}/{total_products_with_data}: {product_name} (Data points: {len(ask_series)})")

        if ask_series.empty: continue

        for interval_str in intervals:
            try:
                resampler = ask_series.resample(interval_str)
                ohlc_df = resampler.ohlc()
                ohlc_df.dropna(how='all', inplace=True)
                if ohlc_df.empty: continue

                volume_series = resampler.count().reindex(ohlc_df.index).fillna(0).astype(int)
                
                ohlc_df['t'] = (ohlc_df.index.astype(np.int64) // 10**9)
                ohlc_df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c'}, inplace=True)
                ohlc_df['v'] = volume_series
                
                cols_for_dict = ['t', 'o', 'h', 'l', 'c', 'v']
                # Ensure all columns exist, add if missing (e.g., if ohlc returns empty for a perfect interval match)
                for col in cols_for_dict:
                    if col not in ohlc_df.columns: ohlc_df[col] = np.nan if col != 'v' else 0


                final_candles_df = ohlc_df[cols_for_dict].copy() # Use .copy() before modifying for np.where

                for col in ['o', 'h', 'l', 'c']:
                    # Using .loc to avoid SettingWithCopyWarning if final_candles_df was a slice
                    final_candles_df.loc[:, col] = np.where(np.isfinite(final_candles_df[col]), final_candles_df[col], None)
                
                # Volume should be fine, but ensure it's int, handle potential NaNs from merge if any
                final_candles_df.loc[:, 'v'] = final_candles_df['v'].fillna(0).astype(int)


                records_list = final_candles_df.to_dict('records')
                if records_list:
                    ohlcv_results[product_name][interval_str] = records_list
            except Exception as e:
                logging.error(f"OHLCV: Error for {product_name} interval {interval_str}: {e}", exc_info=True)
    
    logging.info(f"OHLCV: Finished calculation. Populated OHLCV for {len(ohlcv_results)} products.")
    return ohlcv_results


# --- Main Build Process ---
def main():
    logging.info("--- Starting Static Site Build ---")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")

        item_categories = parse_categories(CATEGORY_FILE_PATH)
        if not item_categories:
            logging.warning("Category data is empty. Categories will be missing.")

        logging.info("--- Downloading Data ---")
        db_ok = download_file(DB_URL, DB_PATH)
        json_ok = download_file(JSON_URL, JSON_PATH)

        if not json_ok:
            logging.error("Failed to download milkyapi.json. Critical live data missing. Cannot proceed.")
            return
        if not db_ok:
            logging.warning("Failed to download market.db. Historical data might be incomplete.")

        logging.info("--- Loading Data ---")
        # Load full history for general trends, market_history.json
        historical_df_full = load_historical_data(days_to_load=HISTORICAL_DAYS, for_ohlcv=False)
        # Load shorter history specifically for OHLCV calculations
        historical_df_ohlcv_source = load_historical_data(days_to_load=HISTORICAL_DAYS_FOR_OHLCV, for_ohlcv=True)
        
        live_df, vendor_prices = load_live_data()

        # --- Combine data for general calculations (trends, volatility, summary) ---
        # This uses the FULL historical data + live data
        if historical_df_full.empty and live_df.empty:
             logging.warning("Both FULL historical and live data are empty. Most calculations will be skipped.")
             combined_df_full = pd.DataFrame()
        elif live_df.empty:
            logging.info("Live data is empty, using only FULL historical data for general calculations.")
            combined_df_full = historical_df_full.copy()
        elif historical_df_full.empty:
            logging.warning("FULL Historical data is empty, using only live data for general calculations.")
            combined_df_full = live_df.copy()
        else:
            logging.info("Concatenating FULL historical and live data for general calculations...")
            combined_df_full = pd.concat([historical_df_full, live_df], ignore_index=True)
        
        # Process combined_df_full
        if not combined_df_full.empty:
            logging.info(f"Processing combined_df_full ({len(combined_df_full)} records)...")
            # Standardize: Ensure 'timestamp' is datetime and UTC, sort, drop duplicates
            combined_df_full['timestamp'] = pd.to_datetime(combined_df_full['timestamp'], errors='coerce')
            combined_df_full.dropna(subset=['timestamp'], inplace=True)
            if not combined_df_full.empty: # Check again after dropna
                if combined_df_full['timestamp'].dt.tz is None:
                    combined_df_full['timestamp'] = combined_df_full['timestamp'].dt.tz_localize('UTC')
                else:
                    combined_df_full['timestamp'] = combined_df_full['timestamp'].dt.tz_convert('UTC')
                combined_df_full.sort_values(by=['product', 'timestamp'], inplace=True)
                combined_df_full.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True)
                for col_type in ['buy', 'ask']:
                    combined_df_full[col_type] = pd.to_numeric(combined_df_full[col_type], errors='coerce')
            logging.info(f"Final processed combined_df_full size: {len(combined_df_full)} records.")


        # --- Prepare data for OHLCV (shorter history + live data) ---
        if historical_df_ohlcv_source.empty and live_df.empty:
            logging.warning("Both OHLCV historical source and live data are empty. OHLCV will be empty.")
            combined_df_ohlcv = pd.DataFrame()
        elif live_df.empty:
            logging.info("Live data is empty, using only OHLCV historical source for OHLCV.")
            combined_df_ohlcv = historical_df_ohlcv_source.copy()
        elif historical_df_ohlcv_source.empty:
            logging.warning("OHLCV historical source is empty, using only live data for OHLCV.")
            combined_df_ohlcv = live_df.copy()
        else:
            logging.info("Concatenating OHLCV historical source and live data...")
            combined_df_ohlcv = pd.concat([historical_df_ohlcv_source, live_df], ignore_index=True)

        # Process combined_df_ohlcv
        if not combined_df_ohlcv.empty:
            logging.info(f"Processing combined_df_ohlcv ({len(combined_df_ohlcv)} records)...")
            combined_df_ohlcv['timestamp'] = pd.to_datetime(combined_df_ohlcv['timestamp'], errors='coerce')
            combined_df_ohlcv.dropna(subset=['timestamp'], inplace=True)
            if not combined_df_ohlcv.empty:
                if combined_df_ohlcv['timestamp'].dt.tz is None:
                    combined_df_ohlcv['timestamp'] = combined_df_ohlcv['timestamp'].dt.tz_localize('UTC')
                else:
                    combined_df_ohlcv['timestamp'] = combined_df_ohlcv['timestamp'].dt.tz_convert('UTC')
                combined_df_ohlcv.sort_values(by=['product', 'timestamp'], inplace=True)
                combined_df_ohlcv.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True)
                for col_type in ['buy', 'ask']:
                     combined_df_ohlcv[col_type] = pd.to_numeric(combined_df_ohlcv[col_type], errors='coerce')
            logging.info(f"Final processed combined_df_ohlcv size: {len(combined_df_ohlcv)} records.")


        # --- Filter Products (based on vendor price, using combined_df_full for most comprehensive product list) ---
        all_products_initial = sorted(list(combined_df_full['product'].unique())) if not combined_df_full.empty else []
        if not all_products_initial:
             logging.warning("No products found in combined_df_full to process. Most outputs will be empty.")
             filtered_products = []
        else:
            logging.info(f"Applying vendor price filter to {len(all_products_initial)} initial products...")
            filtered_products = [
                p for p in all_products_initial
                if p == "Bag Of 10 Cowbells" or ((vp := vendor_prices.get(p)) is not None and vp > 0)
            ] # This logic is from your original script
            logging.info(f"Filtered down to {len(filtered_products)} products based on vendor price.")

        # --- Calculations ---
        # Trends, Volatility, Summary use combined_df_full
        product_trends = calculate_trends(combined_df_full.copy() if not combined_df_full.empty else pd.DataFrame(), filtered_products)
        product_volatility_stats = calculate_volatility_and_cv(combined_df_full.copy() if not combined_df_full.empty else pd.DataFrame(), filtered_products, days=VOLATILITY_DAYS)
        market_indices = calculate_market_indices(product_trends, item_categories)

        # OHLCV Data uses combined_df_ohlcv
        product_ohlcv_data = {}
        if not combined_df_ohlcv.empty and filtered_products:
            logging.info("Calculating OPTIMIZED OHLCV data...")
            product_ohlcv_data = calculate_ohlcv_data_optimized(combined_df_ohlcv, filtered_products, CANDLESTICK_INTERVALS)
        else:
            logging.warning("Skipping OHLCV calculation: OHLCV source data or filtered products list is empty.")

        # --- Generate JSON Data Files ---
        logging.info("--- Generating JSON Data Files ---")

        # 1. Market Summary (uses latest from combined_df_full)
        market_summary = []
        if not combined_df_full.empty and filtered_products:
            try:
                # Get latest data for each product from the FULL dataset
                latest_data_map = combined_df_full.groupby('product').last()
                for product in filtered_products:
                    if product in latest_data_map.index:
                        latest = latest_data_map.loc[product]
                        vol_stats = product_volatility_stats.get(product, {'std': None, 'cv': None})
                        market_summary.append({
                            'name': product,
                            'category': item_categories.get(product, 'Unknown'),
                            'buy': latest['buy'] if pd.notna(latest['buy']) else None,
                            'ask': latest['ask'] if pd.notna(latest['ask']) else None,
                            'vendor': vendor_prices.get(product), # From live_data
                            'trend': product_trends.get(product),
                            f'volatility_{VOLATILITY_DAYS}d': vol_stats['std'],
                            f'volatility_norm_{VOLATILITY_DAYS}d': vol_stats['cv']
                        })
            except Exception as e:
                logging.error(f"Error creating market summary list: {e}", exc_info=True)
        
        summary_path = os.path.join(OUTPUT_DATA_DIR, 'market_summary.json')
        try:
            with open(summary_path, 'w') as f: json.dump(market_summary, f, allow_nan=False)
            logging.info(f"Saved market summary ({len(market_summary)} items) to {summary_path}")
        except Exception as e: logging.error(f"Error saving market summary JSON: {e}", exc_info=True)

        # 2. Full Historical Data (market_history.json, uses combined_df_full)
        nested_history_dict = {}
        if not combined_df_full.empty:
            try:
                grouped_history = combined_df_full.groupby('product')
                for product_name, group_df in grouped_history:
                    if product_name not in filtered_products: continue # Only for filtered products
                    
                    buy_list = [{"timestamp": int(ts.timestamp()), "price": p if pd.notna(p) and np.isfinite(p) else None}
                                for ts, p in zip(group_df['timestamp'], group_df['buy'])]
                    ask_list = [{"timestamp": int(ts.timestamp()), "price": p if pd.notna(p) and np.isfinite(p) else None}
                                for ts, p in zip(group_df['timestamp'], group_df['ask'])]
                    if buy_list or ask_list: # Only add if there's some data
                         nested_history_dict[product_name] = {"buy": buy_list, "ask": ask_list}
            except Exception as e:
                logging.error(f"Error processing data for market_history.json: {e}", exc_info=True)

        history_path = os.path.join(OUTPUT_DATA_DIR, 'market_history.json')
        try:
            with open(history_path, 'w') as f: json.dump(nested_history_dict, f, allow_nan=False)
            logging.info(f"Saved market history ({len(nested_history_dict)} products) to {history_path}")
        except Exception as e: logging.error(f"Error saving market history JSON: {e}", exc_info=True)

        # 3. Market Indices JSON (already calculated)
        indices_path = os.path.join(OUTPUT_DATA_DIR, 'market_indices.json')
        try:
            with open(indices_path, 'w') as f: json.dump(market_indices, f, allow_nan=False)
            logging.info(f"Saved market indices ({len(market_indices)} categories) to {indices_path}")
        except Exception as e: logging.error(f"Error saving market indices JSON: {e}", exc_info=True)

        # 4. OHLCV Data JSON (already calculated from product_ohlcv_data)
        ohlcv_path = os.path.join(OUTPUT_DATA_DIR, 'market_ohlcv.json')
        try:
            with open(ohlcv_path, 'w') as f: json.dump(product_ohlcv_data, f, allow_nan=False)
            logging.info(f"Saved OHLCV data ({len(product_ohlcv_data)} products) to {ohlcv_path}")
        except Exception as e: logging.error(f"Error saving OHLCV data JSON: {e}", exc_info=True)


        # --- Render HTML Template ---
        logging.info("--- Rendering HTML Template ---")
        try:
            unique_categories = sorted(list(set(cat for cat in item_categories.values() if cat != "Unknown")))
            if not os.path.isdir(TEMPLATE_DIR):
                 logging.error(f"Template directory '{TEMPLATE_DIR}' not found!")
            else:
                env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
                template = env.get_template('index.html') # Assumes index.html is your main template
                html_context = {
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'), # Be explicit about UTC
                    'categories': unique_categories,
                    'data_freshness_note': "Note: Market data is updated periodically and may be up to 12 hours old."
                }
                html_content = template.render(html_context)
                html_path = os.path.join(OUTPUT_DIR, 'index.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logging.info(f"Saved main HTML to {html_path}")
        except TemplateNotFound as e:
             logging.error(f"Jinja2 template not found: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error rendering or saving HTML template: {e}", exc_info=True)

    except Exception as e:
         logging.critical(f"A critical error occurred during the main build process: {e}", exc_info=True)
    finally:
        logging.info("--- Static Site Build Finished ---")

if __name__ == '__main__':
    main()

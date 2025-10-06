import os
import sqlite3
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import shutil
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Source Data URLs and Paths
DB_URL = "https://raw.githubusercontent.com/holychikenz/MWIApi/main/market.db"
JSON_URL = "https://www.milkywayidle.com/game_data/marketplace.json"
DB_PATH = "market.db"
JSON_PATH = "marketplace.json"
CATEGORY_FILE_PATH = "cata.txt"

# Output directory for static files
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
TEMPLATE_DIR = "templates"

# History and Trend Settings
HISTORICAL_DAYS = 30
VOLATILITY_DAYS = 7

# --- Utility Function ---
def get_item_name_from_hrid(product_hrid):
    cleaned_name = product_hrid.split('/')[-1]
    human_readable_name = cleaned_name.replace('_', ' ').title()
    category_key = human_readable_name.lower()
    return human_readable_name, category_key

# --- Category Parsing ---
def parse_categories(filepath):
    categories = {}
    current_main_category = "Unknown"
    current_display_category = "Unknown"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Simple category detection: if a line has no comma, it's a category title
                if ',' not in line:
                    current_main_category = line
                    # This logic is simplified; cata.txt structure is assumed to be Category then items
                    current_display_category = current_main_category
                else:
                    item_names = [name.strip() for name in line.split(',') if name.strip()]
                    for item_name in item_names:
                        categories[item_name.lower()] = current_display_category
        logging.info(f"Parsed {len(categories)} items from category file.")
        return categories
    except FileNotFoundError:
        logging.error(f"Category file not found at {filepath}.")
        return {}
    except Exception as e:
        logging.error(f"Error parsing category file {filepath}: {e}", exc_info=True)
        return {}


# --- Data Fetching ---
def download_file(url, local_path):
    logging.info(f"Downloading {url} to {local_path}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        # Check for small file size as a proxy for download error
        if len(response.content) < 1024:
             logging.warning(f"Downloaded file {local_path} is very small ({len(response.content)} bytes). It might be an error page.")
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Successfully downloaded {local_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading {url}: {e}")
        return False
    return False

# --- Data Loading and Processing ---
def load_historical_data(days_to_load):
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.warning(f"{DB_PATH} not found.")
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp()
            query = "SELECT * FROM ask WHERE time >= ? UNION ALL SELECT * FROM bid WHERE time >= ?"
            df_wide = pd.read_sql_query(query, conn, params=(cutoff_timestamp, cutoff_timestamp))
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}", exc_info=True)
        return pd.DataFrame()
    
    if df_wide.empty:
        return pd.DataFrame()

    logging.info("Melting wide-format database data...")
    df_long = df_wide.melt(id_vars=['time'], var_name='product', value_name='price')
    df_long['timestamp'] = pd.to_datetime(df_long['time'], unit='s', utc=True)
    df_long['price'] = pd.to_numeric(df_long['price'], errors='coerce').replace(-1, pd.NA)

    # Heuristic to separate ask and bid based on typical price differences
    # This is an assumption because the unified query loses the source table info
    df_long['type'] = np.where(df_long.groupby('product')['price'].transform('mean') > 0, 'ask', 'bid')
    
    df_pivoted = df_long.pivot_table(index=['timestamp', 'product'], columns='type', values='price').reset_index()
    df_pivoted.rename(columns={'ask': 'ask', 'bid': 'buy'}, inplace=True)
    
    return df_pivoted[['product', 'buy', 'ask', 'timestamp']]

def load_live_data():
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
        
        vendor_prices = data.get('vendorPrice', {})
        market_data = data.get('marketData', {})
        current_time = datetime.now(timezone.utc)

        for product_path, tiers in market_data.items():
            for tier_str, prices in tiers.items():
                record = {
                    'product': product_path,
                    'buy': prices.get('b', -1),
                    'ask': prices.get('a', -1),
                    'timestamp': current_time,
                    'tier': tier_str
                }
                record['buy'] = pd.NA if record['buy'] == -1 else record['buy']
                record['ask'] = pd.NA if record['ask'] == -1 else record['ask']
                
                all_tier_records.append(record)
                if tier_str == "0":
                    base_records.append(record)

        base_df = pd.DataFrame(base_records)
        all_tiers_df = pd.DataFrame(all_tier_records)
        
        logging.info(f"Loaded {len(base_df)} base (Tier 0) live records.")
        return base_df, all_tiers_df, vendor_prices
    except Exception as e:
        logging.error(f"Error loading live data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), {}

def calculate_trends(df):
    trends = {}
    now = datetime.now(timezone.utc)
    twenty_four_hours_ago = now - timedelta(hours=24)
    
    if df.empty: return trends
    
    logging.info("Calculating 24h trends...")
    
    # Ensure timestamp is timezone-aware
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    for product, group in df.groupby('product'):
        latest = group.iloc[-1]
        current_price = latest['ask']
        if pd.isna(current_price): continue

        past_group = group[group['timestamp'] < twenty_four_hours_ago]
        if not past_group.empty:
            past_price = past_group['ask'].dropna().mean()
            if pd.notna(past_price) and past_price > 0:
                trends[product] = ((current_price - past_price) / past_price) * 100
    
    return {k: v for k, v in trends.items() if pd.notna(v) and np.isfinite(v)}

def calculate_volatility_and_cv(df, days=7):
    results = {}
    now = datetime.now(timezone.utc)
    time_cutoff = now - timedelta(days=days)

    if df.empty: return results
    
    logging.info(f"Calculating {days}-day volatility...")
    
    df_period = df[df['timestamp'] >= time_cutoff]

    for product, group in df_period.groupby('product'):
        valid_ask = group['ask'].dropna()
        if len(valid_ask) >= 2:
            std_dev = valid_ask.std()
            mean_price = valid_ask.mean()
            if mean_price > 0:
                results[product] = {
                    'std': std_dev,
                    'cv': (std_dev / mean_price) * 100
                }
    return results

def calculate_market_indices(product_trends, item_categories):
    if not product_trends or not item_categories:
        return {}
    
    logging.info("Calculating market indices...")
    category_trends = defaultdict(list)
    for product, trend in product_trends.items():
        _, category_key = get_item_name_from_hrid(product)
        category = item_categories.get(category_key, "Unknown")
        if category != "Unknown":
            category_trends[category].append(trend)
            
    return {cat: np.mean(trends) for cat, trends in category_trends.items() if trends}


def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    item_categories = parse_categories(CATEGORY_FILE_PATH)
    
    db_ok = download_file(DB_URL, DB_PATH)
    json_ok = download_file(JSON_URL, JSON_PATH)
    
    if not json_ok:
        logging.critical("Failed to download critical live data. Aborting.")
        return

    base_live_df, all_tiers_live_df, vendor_prices = load_live_data()
    historical_df = load_historical_data(HISTORICAL_DAYS) if db_ok else pd.DataFrame()

    combined_df = pd.concat([historical_df, base_live_df], ignore_index=True)
    combined_df.sort_values(by=['product', 'timestamp'], inplace=True)
    combined_df.drop_duplicates(subset=['product', 'timestamp'], keep='last', inplace=True)

    product_trends = calculate_trends(combined_df.copy())
    product_volatility = calculate_volatility_and_cv(combined_df.copy(), VOLATILITY_DAYS)
    market_indices = calculate_market_indices(product_trends, item_categories)

    market_summary = []
    latest_data = combined_df.groupby('product').last()

    for product_hrid, latest in latest_data.iterrows():
        human_name, category_key = get_item_name_from_hrid(product_hrid)
        vol_stats = product_volatility.get(product_hrid, {})
        
        market_summary.append({
            'name': human_name,
            'category': item_categories.get(category_key, 'Unknown'),
            'buy': latest['buy'] if pd.notna(latest['buy']) else None,
            'ask': latest['ask'] if pd.notna(latest['ask']) else None,
            'vendor': vendor_prices.get(product_hrid),
            'trend': product_trends.get(product_hrid),
            'volatility_norm_7d': vol_stats.get('cv')
        })

    # Generate JSON outputs
    def write_json(data, filename):
        path = os.path.join(OUTPUT_DATA_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, allow_nan=False, default=lambda x: None)
        logging.info(f"Saved {filename}")

    write_json(market_summary, 'market_summary.json')
    write_json(market_indices, 'market_indices.json')
    
    enhanced_data = defaultdict(lambda: {'tiers': {}})
    for _, row in all_tiers_live_df.iterrows():
        human_name, _ = get_item_name_from_hrid(row['product'])
        enhanced_data[human_name]['tiers'][row['tier']] = {'ask': row['ask'], 'buy': row['buy']}
    write_json(dict(enhanced_data), 'market_enhanced.json')
    
    history_data = {}
    for name, group in combined_df.groupby('product'):
        human_name, _ = get_item_name_from_hrid(name)
        history_points = group[['timestamp', 'ask', 'buy']].dropna(subset=['ask', 'buy'], how='all').values.tolist()
        history_data[human_name] = [[int(ts.timestamp() * 1000), ask, buy] for ts, ask, buy in history_points]
    write_json(history_data, 'market_history.json')

    shutil.copy(os.path.join(TEMPLATE_DIR, "index.html"), os.path.join(OUTPUT_DIR, "index.html"))
    logging.info("--- Build Finished Successfully ---")

if __name__ == '__main__':
    main()

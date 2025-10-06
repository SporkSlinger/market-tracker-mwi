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
    cleaned_name = product_hrid.split('/')[-1] if product_hrid.startswith('/items/') else product_hrid
    human_readable_name = cleaned_name.replace('_', ' ').title()
    category_key = human_readable_name.lower()
    return human_readable_name, category_key

# --- Category Parsing ---
def parse_categories(filepath):
    categories = {}
    current_main_category = "Unknown"
    current_sub_category = None

    main_categories = ["Currencies", "Loots", "Resources", "Consumables", "Books", "Keys", "Equipment", "Jewelry", "Trinket", "Tools"]
    equipment_subcategories = ["Main Hand", "Off Hand", "Head", "Body", "Legs", "Hands", "Feet", "Back", "Pouch", "Two Hand"]
    tools_subcategories = ["Milking", "Foraging", "Woodcutting", "Cheesesmithing", "Crafting", "Tailoring", "Cooking", "Brewing", "Alchemy", "Enhancing"]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.replace(u'\xa0', u' ').strip():
                    continue

                if line in main_categories:
                    current_main_category = line
                    current_sub_category = None
                elif current_main_category == "Equipment" and line in equipment_subcategories:
                    current_sub_category = line
                elif current_main_category == "Tools" and line in tools_subcategories:
                    current_sub_category = line
                else:
                    display_category = current_main_category
                    if current_sub_category:
                        display_category = f"{current_main_category} / {current_sub_category}"
                    
                    item_names = [name.strip() for name in line.split(',') if name.strip()]
                    for item_name in item_names:
                        categories[item_name.lower()] = display_category
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
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Successfully downloaded {local_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error downloading {url}: {e}")
        return False

# --- Data Loading ---
def load_historical_data(days_to_load):
    logging.info(f"Loading historical data for {days_to_load} days from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.warning(f"{DB_PATH} not found.")
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cutoff_timestamp = (datetime.now() - timedelta(days=days_to_load)).timestamp()
            ask_df_wide = pd.read_sql_query("SELECT * FROM ask WHERE time >= ?", conn, params=(cutoff_timestamp,))
            bid_df_wide = pd.read_sql_query("SELECT * FROM bid WHERE time >= ?", conn, params=(cutoff_timestamp,))
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}", exc_info=True)
        return pd.DataFrame()

    def melt_and_process(df, value_name):
        if df.empty or 'time' not in df.columns: return pd.DataFrame()
        id_vars, value_vars = ['time'], [col for col in df.columns if col != 'time']
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='product', value_name=value_name)
        df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce').replace(-1, pd.NA)
        df_long.dropna(subset=[value_name], inplace=True)
        df_long['timestamp'] = pd.to_datetime(df_long['time'], unit='s', utc=True)
        return df_long[['product', 'timestamp', value_name]]

    ask_df = melt_and_process(ask_df_wide, 'ask')
    buy_df = melt_and_process(bid_df_wide, 'buy')

    if ask_df.empty and buy_df.empty: return pd.DataFrame()
    if ask_df.empty:
        buy_df['ask'] = pd.NA
        return buy_df
    if buy_df.empty:
        ask_df['buy'] = pd.NA
        return ask_df

    return pd.merge(ask_df, buy_df, on=['product', 'timestamp'], how='outer')

def load_live_data():
    logging.info(f"Loading live data from {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        logging.warning(f"{JSON_PATH} not found.")
        return pd.DataFrame(), pd.DataFrame(), {}

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vendor_prices = data.get('vendorPrice', {})
        market_data = data.get('marketData', {})
        current_time = datetime.now(timezone.utc)
        all_records = []

        for product_path, tiers in market_data.items():
            for tier_str, prices in tiers.items():
                all_records.append({
                    'product': product_path,
                    'buy': prices.get('b', -1),
                    'ask': prices.get('a', -1),
                    'timestamp': current_time,
                    'tier': tier_str
                })
        
        all_tiers_df = pd.DataFrame(all_records)
        all_tiers_df['buy'] = all_tiers_df['buy'].replace(-1, pd.NA)
        all_tiers_df['ask'] = all_tiers_df['ask'].replace(-1, pd.NA)

        base_df = all_tiers_df[all_tiers_df['tier'] == "0"].copy()
        logging.info(f"Loaded {len(base_df)} base (Tier 0) live records.")
        return base_df, all_tiers_df, vendor_prices
    except Exception as e:
        logging.error(f"Error loading live data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), {}

# --- Single-Pass Data Processing ---
def process_data_in_one_pass(df, item_categories, vendor_prices, vol_days=7):
    # FIX: This function is now more efficient and builds the market summary directly.
    logging.info("Processing all data in a single pass...")
    market_summary = []
    history = defaultdict(list)
    
    now = datetime.now(timezone.utc)
    trend_cutoff = now - timedelta(hours=24)
    vol_cutoff = now - timedelta(days=vol_days)

    for product, group in df.groupby('product'):
        group = group.sort_values(by='timestamp')
        
        # 1. Get Latest Data
        latest_row = group.iloc[-1]
        current_price = latest_row['ask']
        
        trend = None
        volatility = None

        # 2. Resilient Trend Calculation
        if pd.notna(current_price) and len(group) > 1:
            past_24h_prices = group[(group['timestamp'] >= trend_cutoff) & (group['timestamp'] < latest_row['timestamp'])]['ask'].dropna()
            
            previous_price = None
            if not past_24h_prices.empty:
                previous_price = past_24h_prices.mean()
            else:
                previous_prices = group[group['timestamp'] < latest_row['timestamp']]['ask'].dropna()
                if not previous_prices.empty:
                    previous_price = previous_prices.iloc[-1]

            if previous_price is not None and previous_price > 0:
                trend = ((current_price - previous_price) / previous_price) * 100

        # 3. Volatility Calculation
        vol_group = group[group['timestamp'] >= vol_cutoff]
        valid_ask = vol_group['ask'].dropna()
        if len(valid_ask) >= 2:
            std_dev = valid_ask.std()
            mean_price = valid_ask.mean()
            if mean_price > 0:
                volatility = (std_dev / mean_price) * 100
        
        # 4. Assemble Market Summary entry
        human_name, category_key = get_item_name_from_hrid(product)
        market_summary.append({
            'name': human_name,
            'category': item_categories.get(category_key, 'Unknown'),
            'buy': latest_row['buy'] if pd.notna(latest_row['buy']) else None,
            'ask': latest_row['ask'] if pd.notna(latest_row['ask']) else None,
            'vendor': vendor_prices.get(product),
            'trend': trend,
            'volatility_norm_7d': volatility
        })
        
        # 5. History Generation
        group['js_timestamp'] = group['timestamp'].astype(np.int64) // 10**6
        history_points = group[['js_timestamp', 'ask', 'buy']].dropna(how='all', subset=['ask', 'buy']).values.tolist()
        if history_points:
            history[human_name] = history_points

    return market_summary, history

def calculate_market_indices(market_summary):
    if not market_summary: return {}
    logging.info("Calculating market indices...")
    category_trends = defaultdict(list)
    for item in market_summary:
        trend = item.get('trend')
        category = item.get('category', 'Unknown')
        if category != "Unknown" and trend is not None and np.isfinite(trend):
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

    # FIX: Use the optimized single-pass function and get results directly.
    market_summary, history_data = process_data_in_one_pass(combined_df.copy(), item_categories, vendor_prices, VOLATILITY_DAYS)
    
    # Market indices can now be calculated from the final summary.
    market_indices = calculate_market_indices(market_summary)

    def write_json(data, filename):
        path = os.path.join(OUTPUT_DATA_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, default=lambda x: None if pd.isna(x) else x)
        logging.info(f"Saved {filename}")

    write_json(market_summary, 'market_summary.json')
    write_json(market_indices, 'market_indices.json')
    write_json(history_data, 'market_history.json')
    
    enhanced_data = defaultdict(lambda: {'tiers': {}})
    for _, row in all_tiers_live_df.iterrows():
        human_name, _ = get_item_name_from_hrid(row['product'])
        enhanced_data[human_name]['tiers'][row['tier']] = {'ask': row['ask'], 'buy': row['buy']}
    write_json(dict(enhanced_data), 'market_enhanced.json')

    shutil.copy(os.path.join(TEMPLATE_DIR, "index.html"), os.path.join(OUTPUT_DIR, "index.html"))
    logging.info("--- Build Finished Successfully ---")

if __name__ == '__main__':
    main()

# market-tracker-mwi
Market Tracker for MWI
# MWI Market Tracker (Static Site Generator)

This project automatically fetches market data for Milky Way Idle from public sources, processes it, calculates trends, and generates a static website displaying market information with charts.

## Features

* Displays current Buy/Ask/Vendor prices
* Shows item categories (Loots, Resources, Equipment, etc.).
* Calculates and displays 24-hour price trends (updated daily).
* Provides interactive price history charts (using Chart.js) with selectable time ranges (24h, 48h, 7d, 30d).
* Client-side search, sorting (Name, Category, Price, Trend), and pagination.
* Static site hosted on GitHub Pages, updated automatically via GitHub Actions.


## Viewing the Site

The live site can be viewed at: https://sporkslinger.github.io/market-tracker-mwi

## Local Development / Building (Optional)

1.  Clone the repository.
2.  Ensure Python 3.9+ is installed.
3.  Create and activate a virtual environment (recommended).
4.  Install dependencies: `pip install -r requirements.txt`
5.  Run the build script: `python build_site.py`
6.  Serve the generated `output/` directory locally: `cd output && python -m http.server`
7.  Open `http://localhost:8000` in your browser.

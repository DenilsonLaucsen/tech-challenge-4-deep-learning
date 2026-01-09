import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download(ticker="AAPL", start="2018-01-01", period=None):
    tk = yf.Ticker(ticker)
    if period:
        df = tk.history(period=period)
    else:
        if start:
            df = tk.history(start=start)
        else:
            df = tk.history(period="max")
    df.to_csv(DATA_DIR / f"{ticker}.csv")
    print(f"Dados salvos em {DATA_DIR}/{ticker}.csv")


if __name__ == "__main__":
    download()

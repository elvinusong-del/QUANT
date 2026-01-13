# data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd


UTC_TZ = "UTC"

@dataclass(frozen=True)
class DataPaths:
    root: Path
    agg_trades_dir: str = "aggTrades"
    book_depth_dir: str = "bookDepth"
    book_ticker_dir: str = "bookTicker"  # ✅ Ticker 폴더
    funding_dir: str = "funding_rate_daily"
    klines_1m_dir: str = "klines_1m"

    def agg_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.agg_trades_dir / symbol / f"{ymd}.parquet"

    def book_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.book_depth_dir / symbol / f"{ymd}.parquet"
    
    def ticker_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.book_ticker_dir / symbol / f"{ymd}.parquet"

    def klines_1m_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.klines_1m_dir / symbol / f"{symbol}_{ymd}_1m.parquet"


def _ms_to_utc_datetime(ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ms, unit="ms", utc=True, errors='coerce')


def _coerce_float64(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")


class DataLoader:
    def __init__(self, data_root: str | Path):
        self.paths = DataPaths(root=Path(data_root))

    def load_aggtrades_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.agg_day(symbol, ymd)
        if not fp.exists(): return pd.DataFrame()
        df = pd.read_parquet(fp).copy()
        
        if "transact_time" in df.columns:
             df["ts"] = _ms_to_utc_datetime(df["transact_time"])
        
        _coerce_float64(df, ["price", "quantity"])
        df["symbol"] = symbol
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def load_book_ticker_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        """ ✅ Ticker 로드 함수 """
        fp = self.paths.ticker_day(symbol, ymd)
        if not fp.exists(): return pd.DataFrame()
        df = pd.read_parquet(fp).copy()

        # 컬럼 매핑 (파케이 파일 구조에 맞게)
        rename_map = {
            "transaction_time": "ts",
            "event_time": "ts",
            "best_bid_price": "bid_p",
            "best_bid_qty": "bid_q",
            "best_ask_price": "ask_p",
            "best_ask_qty": "ask_q"
        }
        df = df.rename(columns=rename_map)

        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

        _coerce_float64(df, ["bid_p", "bid_q", "ask_p", "ask_q"])
        df["symbol"] = symbol
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def load_klines_1m_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.klines_1m_day(symbol, ymd)
        if not fp.exists(): return pd.DataFrame()
        df = pd.read_parquet(fp).copy()
        df["open_ts"] = _ms_to_utc_datetime(df["open_time_ms"])
        _coerce_float64(df, ["open", "high", "low", "close", "volume"])
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")
        df["symbol"] = symbol
        df = df.sort_values("open_ts").reset_index(drop=True)
        return df
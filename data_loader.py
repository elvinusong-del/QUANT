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
    book_ticker_dir: str = "bookTicker"
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
        """ ✅ Ticker 로드 함수 (수정됨: 중복 컬럼 방지 및 파싱 최적화) """
        fp = self.paths.ticker_day(symbol, ymd)
        if not fp.exists(): return pd.DataFrame()
        
        try:
            df = pd.read_parquet(fp).copy()

            # 컬럼 매핑 (중복 방지 로직 적용)
            rename_map = {
                "best_bid_price": "bid_p",
                "best_bid_qty": "bid_q",
                "best_ask_price": "ask_p",
                "best_ask_qty": "ask_q"
            }
            
            # transaction_time과 event_time 중 하나만 ts로 매핑하고 나머지 삭제
            if "transaction_time" in df.columns:
                rename_map["transaction_time"] = "ts"
                if "event_time" in df.columns:
                    df = df.drop(columns=["event_time"])
            elif "event_time" in df.columns:
                rename_map["event_time"] = "ts"

            df = df.rename(columns=rename_map)

            if "ts" in df.columns:
                # 숫자형이면 unit='ms' 적용하여 고속 변환, 아니면 자동 파싱
                if pd.api.types.is_numeric_dtype(df["ts"]):
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                else:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

            _coerce_float64(df, ["bid_p", "bid_q", "ask_p", "ask_q"])
            df["symbol"] = symbol
            df = df.sort_values("ts").reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load bookTicker: {fp} ({e})")
            return pd.DataFrame()

    def load_klines_1m_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.klines_1m_day(symbol, ymd)
        if not fp.exists(): return pd.DataFrame()
        
        try:
            df = pd.read_parquet(fp).copy()
            
            # [수정] open_ts 생성 로직 개선 (open_time_kst 우선 사용)
            if "open_time_kst" in df.columns:
                 # 이미 datetime일 경우 변환, 문자열일 경우 파싱
                if pd.api.types.is_datetime64_any_dtype(df["open_time_kst"]):
                    df["open_ts"] = df["open_time_kst"].dt.tz_convert("UTC")
                else:
                    df["open_ts"] = pd.to_datetime(df["open_time_kst"], utc=True)
            elif "open_time" in df.columns:
                if pd.api.types.is_numeric_dtype(df["open_time"]):
                    df["open_ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                else:
                    df["open_ts"] = pd.to_datetime(df["open_time"], utc=True)
            elif "open_time_ms" in df.columns:
                 df["open_ts"] = _ms_to_utc_datetime(df["open_time_ms"])
            
            _coerce_float64(df, ["open", "high", "low", "close", "volume"])
            
            if "trades" in df.columns:
                df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")
            
            df["symbol"] = symbol
            if "open_ts" in df.columns:
                df = df.sort_values("open_ts").reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load klines: {fp} ({e})")
            return pd.DataFrame()
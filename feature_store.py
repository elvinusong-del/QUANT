# feature_store.py
from __future__ import annotations

from typing import Dict, Optional, Deque
from collections import deque
import pandas as pd

# ✅ 패키지명(hft_backtest_engine.) 삭제. 같은 폴더 features.py 참조
from features import (
    VPINCalculator,
    OFICalculator,
    compute_trade_count_spike,
    compute_qr,
)

class FeatureStore:
    def __init__(
        self,
        symbol: str,
        signal_interval_seconds: int = 5 * 60,
        vpin_bucket_volume: float = 1e6,
        vpin_history: int = 100,
        tc_window: int = 60,
        ofi_window_minutes: int = 5,
        ofi_z_window: int = 30,
        keep_book_minutes: int = 10,
        keep_agg_debug_rows: int = 0,
    ):
        self.symbol = symbol
        self.signal_interval_seconds = signal_interval_seconds

        # Calculators
        self.vpin_calc = VPINCalculator(bucket_volume=vpin_bucket_volume, history=vpin_history)
        self.ofi_calc = OFICalculator(window_minutes=ofi_window_minutes, z_window=ofi_z_window)

        self.tc_window = tc_window
        self.ofi_window_minutes = ofi_window_minutes
        self.keep_book_minutes = max(keep_book_minutes, ofi_window_minutes + 1)

        # Buffers
        self._kline_rows: Deque[dict] = deque(maxlen=self.tc_window + 2)
        # ✅ Ticker Data 저장소
        self._book_snaps: Deque[pd.DataFrame] = deque(maxlen=100_000)
        self._last_book_ts: Optional[pd.Timestamp] = None
        self._last_ticker_row: Optional[pd.Series] = None

        self.keep_agg_debug_rows = int(keep_agg_debug_rows)
        self._agg_debug: Deque[dict] = deque(maxlen=1) 
        
        self.last_feature_ts: Optional[pd.Timestamp] = None
        self.cached_features: Optional[Dict[str, float]] = None

    def update_trade(self, trade_row) -> None:
        self.vpin_calc.update_trade(trade_row)

    def update_kline(self, kline_df: pd.DataFrame) -> None:
        if kline_df is None or kline_df.empty: return
        row = kline_df.iloc[-1]
        self._kline_rows.append({
            "open_ts": row["open_ts"],
            "trades": float(row["trades"]),
        })

    def update_book(self, ticker_df: pd.DataFrame) -> None:
        """ ✅ Ticker DF 업데이트 """
        if ticker_df is None or ticker_df.empty: return
        
        current_ts = ticker_df["ts"].iloc[-1]
        self._last_ticker_row = ticker_df.iloc[-1]
        self._book_snaps.append(ticker_df)
        self._last_book_ts = current_ts

        cutoff = current_ts - pd.Timedelta(minutes=self.keep_book_minutes)
        while self._book_snaps and self._book_snaps[0]["ts"].iloc[0] < cutoff:
            self._book_snaps.popleft()

    def should_compute(self, ts: pd.Timestamp) -> bool:
        if self.last_feature_ts is None: return True
        return (ts - self.last_feature_ts).total_seconds() >= self.signal_interval_seconds

    def compute_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        # 1) VPIN
        vpin = self.vpin_calc.get_value()
        
        # 2) TC
        if len(self._kline_rows) < self.tc_window + 1:
            tc = {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}
        else:
            kline_df = pd.DataFrame(list(self._kline_rows)).sort_values("open_ts")
            tc = compute_trade_count_spike(kline_df, window=self.tc_window)

        # 3) OFI (Ticker)
        if not self._book_snaps:
            ofi = {"ofi_raw": 0.0, "z_ofi": 0.0}
        else:
            ticker_full = pd.concat(list(self._book_snaps), ignore_index=True).sort_values("ts")
            start = ts - pd.Timedelta(minutes=self.ofi_window_minutes)
            ticker_slice = ticker_full[(ticker_full["ts"] >= start) & (ticker_full["ts"] <= ts)]
            ofi = self.ofi_calc.update(ticker_slice)

        # 4) QR (Ticker)
        if self._last_ticker_row is None:
            qr = 0.0
        else:
            qr = compute_qr(self._last_ticker_row)

        features = {
            "ts": ts,
            "vpin_raw": float(vpin.get("vpin_raw", 0.0)),
            "vpin_cdf": float(vpin.get("vpin_cdf", 0.0)),
            "tc": float(tc.get("tc", 0.0)),
            "z_tc": float(tc.get("z_tc", 0.0)),
            "n_cdf": float(tc.get("n_cdf", 0.5)),
            "ofi_raw": float(ofi.get("ofi_raw", 0.0)),
            "z_ofi": float(ofi.get("z_ofi", 0.0)),
            "qr": float(qr),
        }
        self.last_feature_ts = ts
        self.cached_features = features
        return features

    def get_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        if self.cached_features is None or self.should_compute(ts):
            return self.compute_features(ts)
        return self.cached_features
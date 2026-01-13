# features.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# ======================================================
# A. VPIN (Stateful)
# ======================================================
class VPINCalculator:
    def __init__(self, bucket_volume: float, history: int = 100):
        self.bucket_volume = float(bucket_volume)
        self.history = int(history)
        self.acc_vol = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.bucket_imbalances: Deque[float] = deque(maxlen=self.history)

    def update_trade(self, trade) -> None:
        q = float(trade.quantity)
        self.acc_vol += q
        if bool(trade.is_buyer_maker):
            self.sell_vol += q
        else:
            self.buy_vol += q

        if self.acc_vol >= self.bucket_volume:
            denom = max(self.buy_vol + self.sell_vol, 1e-12)
            imb = abs(self.buy_vol - self.sell_vol) / denom
            self.bucket_imbalances.append(float(imb))
            self.acc_vol = 0.0
            self.buy_vol = 0.0
            self.sell_vol = 0.0

    def get_value(self) -> dict:
        if not self.bucket_imbalances:
            return {"vpin_raw": np.nan, "vpin_cdf": 0.0}
        vpin_raw = float(self.bucket_imbalances[-1])
        hist = np.asarray(self.bucket_imbalances, dtype="float64")
        vpin_cdf = float(np.mean(hist <= vpin_raw))
        return {"vpin_raw": vpin_raw, "vpin_cdf": vpin_cdf}


# ======================================================
# B. Trade Count Spike (Stateless)
# ======================================================
def compute_trade_count_spike(klines_1m: pd.DataFrame, window: int = 60) -> dict:
    if klines_1m is None or klines_1m.empty:
        return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}
    if "trades" not in klines_1m.columns:
        return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}

    tc_series = pd.to_numeric(klines_1m["trades"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    if len(tc_series) == 0:
         return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}

    tc_cur = float(tc_series[-1])
    if len(tc_series) < window + 1:
        return {"tc": tc_cur, "z_tc": 0.0, "n_cdf": 0.5}

    hist = tc_series[-window - 1:-1]
    mu = float(hist.mean())
    sigma = float(hist.std(ddof=0))
    z = 0.0 if sigma == 0 else (tc_cur - mu) / sigma
    n_cdf = float(norm.cdf(z))
    return {"tc": tc_cur, "z_tc": float(z), "n_cdf": n_cdf}


# ======================================================
# C. OFI (Stateful, Ticker Version)
# ======================================================
@dataclass
class _BBOState:
    bid_p: float
    bid_q: float
    ask_p: float
    ask_q: float

class OFICalculator:
    def __init__(self, window_minutes: int = 5, z_window: int = 30):
        self.window_minutes = int(window_minutes)
        self.z_window = int(z_window)
        self.prev_state: Optional[_BBOState] = None
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self.impact_window: Deque[Tuple[pd.Timestamp, float]] = deque()
        self.ofi_history: Deque[float] = deque(maxlen=self.z_window)

    def update(self, ticker_df: pd.DataFrame) -> dict:
        """
        ticker_df: [ts, bid_p, bid_q, ask_p, ask_q]
        """
        if ticker_df is None or ticker_df.empty:
            return {"ofi_raw": 0.0, "z_ofi": 0.0}

        # 중복 방지
        if self.last_processed_ts is not None:
            df = ticker_df[ticker_df["ts"] > self.last_processed_ts].copy()
        else:
            df = ticker_df.copy()

        if df.empty:
            ofi_raw = float(sum(v for _, v in self.impact_window))
            z_ofi = self._zscore(ofi_raw)
            return {"ofi_raw": ofi_raw, "z_ofi": z_ofi}

        for row in df.itertuples(index=False):
            ts = row.ts
            bp, bq = float(row.bid_p), float(row.bid_q)
            ap, aq = float(row.ask_p), float(row.ask_q)

            if self.prev_state is None:
                self.prev_state = _BBOState(bp, bq, ap, aq)
                self.last_processed_ts = ts
                continue

            # Bid Impact
            e_b = 0.0
            if bp > self.prev_state.bid_p:
                e_b = bq
            elif bp < self.prev_state.bid_p:
                e_b = -self.prev_state.bid_q
            else:
                e_b = bq - self.prev_state.bid_q

            # Ask Impact
            e_a = 0.0
            if ap > self.prev_state.ask_p:
                e_a = -self.prev_state.ask_q
            elif ap < self.prev_state.ask_p:
                e_a = aq
            else:
                e_a = aq - self.prev_state.ask_q

            ofi_val = e_b - e_a
            self.impact_window.append((ts, ofi_val))
            self.prev_state = _BBOState(bp, bq, ap, aq)
            self.last_processed_ts = ts

        # Window Pruning
        if self.impact_window:
            cutoff = self.last_processed_ts - pd.Timedelta(minutes=self.window_minutes)
            while self.impact_window and self.impact_window[0][0] < cutoff:
                self.impact_window.popleft()

        ofi_raw = float(sum(v for _, v in self.impact_window))
        self.ofi_history.append(ofi_raw)
        z_ofi = self._zscore(ofi_raw)

        return {"ofi_raw": ofi_raw, "z_ofi": z_ofi}

    def _zscore(self, ofi_raw: float) -> float:
        hist = np.asarray(self.ofi_history, dtype="float64")
        if len(hist) < 2:
            return 0.0
        mu = float(hist.mean())
        sigma = float(hist.std(ddof=0))
        return 0.0 if sigma == 0 else float((ofi_raw - mu) / sigma)


# ======================================================
# D. QR (Stateless, Ticker Version)
# ======================================================
def compute_qr(ticker_row: pd.Series | pd.DataFrame) -> float:
    if ticker_row is None:
        return 0.0
    
    if isinstance(ticker_row, pd.DataFrame):
        if ticker_row.empty: return 0.0
        row = ticker_row.iloc[-1]
    else:
        row = ticker_row

    try:
        bq = float(row.get("bid_q", 0.0))
        aq = float(row.get("ask_q", 0.0))
    except:
        return 0.0

    denom = bq + aq
    if denom == 0:
        return 0.0
    return (bq - aq) / denom
# execution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Order:
    order_id: int
    symbol: str
    side: int                 # +1 buy, -1 sell
    size: float               # notional (cash 단위)
    order_type: str           # "market" | "limit"
    price: Optional[float]
    created_ts: pd.Timestamp
    expire_ts: Optional[pd.Timestamp] = None
    status: str = "PENDING"


@dataclass
class FillResult:
    filled: bool
    fill_price: Optional[float]
    fee_bp: float = 0.0
    fee_amt: float = 0.0


class ExecutionEngine:
    def __init__(
        self,
        latency_ms: int = 5,
        slippage_bp: float = 1.0,
        fee_bp_market: float = 5.0,
        fee_bp_limit: float = 2.0,
    ):
        self.latency_ms = int(latency_ms)
        self.slippage_bp = float(slippage_bp)
        self.fee_bp_market = float(fee_bp_market)
        self.fee_bp_limit = float(fee_bp_limit)

    def order_active_ts(self, order: Order) -> pd.Timestamp:
        return order.created_ts + pd.Timedelta(milliseconds=self.latency_ms)

    def try_fill(self, order: Order, trade) -> FillResult:
        trade_price = float(trade.price)

        if order.order_type == "market":
            fill_price = self._apply_slippage(trade_price, order.side)
            fee_bp = self.fee_bp_market

        elif order.order_type == "limit":
            if order.price is None:
                return FillResult(False, None)

            limit_px = float(order.price)

            # Buy Limit: 시장가가 지정가보다 내려와야 체결
            if order.side == 1 and trade_price > limit_px:
                return FillResult(False, None)

            # Sell Limit: 시장가가 지정가보다 올라와야 체결
            if order.side == -1 and trade_price < limit_px:
                return FillResult(False, None)

            fill_price = trade_price
            fee_bp = self.fee_bp_limit

        else:
            raise ValueError(f"Unknown order_type: {order.order_type}")

        fee_amt = float(order.size) * (fee_bp * 1e-4)

        return FillResult(
            filled=True,
            fill_price=float(fill_price),
            fee_bp=float(fee_bp),
            fee_amt=float(fee_amt),
        )

    def _apply_slippage(self, price: float, side: int) -> float:
        slip = float(price) * self.slippage_bp * 1e-4
        return float(price) + float(side) * slip

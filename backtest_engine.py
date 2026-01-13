# backtest_engine.py
from __future__ import annotations

from typing import Dict, Optional
import itertools
import pandas as pd

# ✅ 패키지명 제거하고 직접 import
from data_loader import DataLoader
from strategy_base import Strategy, StrategyState, Position, Order as StrategyOrder
from execution import ExecutionEngine, Order as ExecOrder, FillResult
from feature_store import FeatureStore

COMPUTE_DELAY_MS = 10

class BacktestEngine:
    def __init__(
        self,
        loader: DataLoader,
        strategy: Strategy,
        execution: ExecutionEngine,
        feature_store: FeatureStore,
        initial_capital: float = 100.0,
        verbose: bool = True,
    ):
        self.loader = loader
        self.strategy = strategy
        self.execution = execution
        self.feature_store = feature_store
        self.verbose = verbose

        self.state = StrategyState(
            cash=initial_capital,
            positions={},
            current_ts=None,
        )
        self.active_orders: Dict[int, ExecOrder] = {}
        self.order_id_gen = itertools.count(1)
        self.fills = []
        self.closed_trades = []

        self._last_pushed_book_ts: Optional[pd.Timestamp] = None
        self._last_seen_book_ts: Optional[pd.Timestamp] = None

    def run_day(self, symbol: str, ymd: str):
        df = self.loader.load_aggtrades_day(symbol, ymd)
        if df is None or df.empty:
            if self.verbose: print(f"[WARN] no aggTrades: {symbol} {ymd}")
            return

        # ✅ Ticker 로드 (feature_store와 연동 확인됨)
        try:
            book_df = self.loader.load_book_ticker_day(symbol, ymd)
        except Exception as e:
            if self.verbose: print(f"[WARN] no bookTicker: {symbol} {ymd} ({e})")
            book_df = pd.DataFrame()

        try:
            klines = self.loader.load_klines_1m_day(symbol, ymd)
        except Exception as e:
            if self.verbose: print(f"[WARN] no klines_1m: {symbol} {ymd} ({e})")
            klines = pd.DataFrame()

        trades = df.sort_values("ts").reset_index(drop=True)

        # Book Ticker Indexing
        book_ts_list = []
        book_groups = None
        if not book_df.empty:
            book_df = book_df.sort_values("ts").reset_index(drop=True)
            book_groups = book_df.groupby("ts", sort=True).indices
            book_ts_list = sorted(book_groups.keys())
        book_ptr = 0

        # Kline Indexing
        kline_map = None
        if not klines.empty:
            klines = klines.sort_values("open_ts").reset_index(drop=True)
            kline_map = klines.set_index("open_ts", drop=False)
        last_min = None

        for trade in trades.itertuples(index=False):
            ts = trade.ts
            self.state.current_ts = ts

            self.feature_store.update_trade(trade)

            if kline_map is not None:
                cur_min = ts.floor("1min")
                if last_min is None:
                    last_min = cur_min
                elif cur_min > last_min:
                    last_completed_min = last_min
                    last_min = cur_min
                    if last_completed_min in kline_map.index:
                        self.feature_store.update_kline(kline_map.loc[[last_completed_min]])

            # Ticker Push
            if book_groups is not None and book_ts_list:
                while book_ptr < len(book_ts_list) and book_ts_list[book_ptr] <= ts:
                    self._last_seen_book_ts = book_ts_list[book_ptr]
                    book_ptr += 1

                last_seen = self._last_seen_book_ts
                if last_seen is not None and self._last_pushed_book_ts != last_seen:
                    idxs = book_groups[last_seen]
                    snapshot = book_df.loc[idxs]
                    self.feature_store.update_book(snapshot)
                    self._last_pushed_book_ts = last_seen

            self._process_active_orders(trade)

            exit_orders = self.strategy.on_tick(trade=trade, state=self.state)
            for o in exit_orders:
                self._submit_order(o, ts)

            entry_orders = self.strategy.on_signal(trade=trade, state=self.state)
            for o in entry_orders:
                self._submit_order(o, ts)

    def _submit_order(self, proto_order: StrategyOrder, ts: pd.Timestamp):
        oid = next(self.order_id_gen)
        created_ts = ts + pd.Timedelta(milliseconds=COMPUTE_DELAY_MS)
        expire_ts = None
        if proto_order.order_type == "limit":
            expire_ts = created_ts + pd.Timedelta(seconds=5)

        order = ExecOrder(
            order_id=oid,
            symbol=proto_order.symbol,
            side=proto_order.side,
            size=float(proto_order.size),
            order_type=proto_order.order_type,
            price=None if proto_order.price is None else float(proto_order.price),
            created_ts=created_ts,
            expire_ts=expire_ts,
            status="PENDING",
        )
        self.active_orders[oid] = order

    def _process_active_orders(self, trade):
        to_remove = []
        for oid, order in list(self.active_orders.items()):
            if trade.ts < self.execution.order_active_ts(order):
                continue

            if order.expire_ts and trade.ts >= order.expire_ts:
                order.status = "CANCELED"
                to_remove.append(oid)
                market_oid = next(self.order_id_gen)
                market_order = ExecOrder(
                    order_id=market_oid,
                    symbol=order.symbol,
                    side=order.side,
                    size=order.size,
                    order_type="market",
                    price=None,
                    created_ts=trade.ts,
                    expire_ts=None,
                    status="PENDING",
                )
                self.active_orders[market_oid] = market_order
                continue

            fill: FillResult = self.execution.try_fill(order, trade)
            if fill.filled:
                self._apply_fill(order, fill, trade.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    def _apply_fill(self, order: ExecOrder, fill: FillResult, ts: pd.Timestamp):
        symbol = order.symbol
        price = float(fill.fill_price)
        fee_bp = float(getattr(fill, "fee_bp", 0.0))
        fee_amt = float(getattr(fill, "fee_amt", 0.0))

        if symbol not in self.state.positions:
            pos = Position(symbol=symbol, size=order.size, side=order.side, entry_price=price, entry_ts=ts)
            setattr(pos, "entry_fee_amt", fee_amt)
            self.state.positions[symbol] = pos
            self.state.cash -= (order.size + fee_amt)
        else:
            pos = self.state.positions.pop(symbol)
            entry_fee_amt = float(getattr(pos, "entry_fee_amt", 0.0))
            gross_pnl = (pos.side * (price - pos.entry_price) / pos.entry_price * pos.size)
            self.state.cash += (pos.size + gross_pnl - fee_amt)
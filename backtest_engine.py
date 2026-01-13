# backtest_engine.py
from __future__ import annotations

from typing import Dict, Optional
import itertools
import pandas as pd

# ✅ 패키지명 제거하고 직접 import (Quant 버전 스타일 유지)
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
        
        # ✅ 복원: 거래 기록을 저장할 리스트 초기화
        self.fills = []
        self.closed_trades = []

        self._last_pushed_book_ts: Optional[pd.Timestamp] = None
        self._last_seen_book_ts: Optional[pd.Timestamp] = None

    def run_day(self, symbol: str, ymd: str):
        df = self.loader.load_aggtrades_day(symbol, ymd)
        if df is None or df.empty:
            if self.verbose: print(f"[WARN] no aggTrades: {symbol} {ymd}")
            return

        # ✅ 유지: Quant 버전의 핵심인 'Book Ticker' 데이터 로딩 사용
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
            # Ticker 데이터 정렬 (Quant 버전 로직)
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

        # ✅ 복원: 주문 제출 로그
        if self.verbose:
            print(
                f"[ORDER] {created_ts} {order.symbol} "
                f"{order.order_type.upper()} id={oid} (compute+{COMPUTE_DELAY_MS}ms)"
            )

    def _process_active_orders(self, trade):
        to_remove = []
        for oid, order in list(self.active_orders.items()):
            if trade.ts < self.execution.order_active_ts(order):
                continue

            if order.expire_ts and trade.ts >= order.expire_ts:
                order.status = "CANCELED"
                to_remove.append(oid)
                
                # ✅ 복원: 취소 로그
                if self.verbose:
                    print(f"[CANCEL] {trade.ts} {order.symbol} LIMIT id={oid}")

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
                
                # ✅ 복원: 강제 시장가 전환 로그
                if self.verbose:
                    print(f"[FORCE MARKET] {trade.ts} {order.symbol}")

                continue

            fill: FillResult = self.execution.try_fill(order, trade)
            if fill.filled:
                self._apply_fill(order, fill, trade.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    # ✅ 복원: FR 버전의 _apply_fill 로직 전체 (기록 및 PnL 계산 포함)
    def _apply_fill(self, order: ExecOrder, fill: FillResult, ts: pd.Timestamp):
        symbol = order.symbol
        price = float(fill.fill_price)

        # ExecutionEngine이 아직 fee 필드를 안 넣어줘도 죽지 않게 방어
        fee_bp = float(getattr(fill, "fee_bp", 0.0))
        fee_amt = float(getattr(fill, "fee_amt", 0.0))

        # =========================
        # 진입 (Entry)
        # =========================
        if symbol not in self.state.positions:
            pos = Position(
                symbol=symbol,
                size=order.size,
                side=order.side,
                entry_price=price,
                entry_ts=ts,
            )
            # entry fee 저장(나중에 exit에서 net_pnl 계산에 포함)
            setattr(pos, "entry_fee_amt", fee_amt)

            self.state.positions[symbol] = pos
            self.state.cash -= (order.size + fee_amt)

            # [복원] 진입 기록 저장
            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "ENTER",
                "side": order.side,
                "price": price,
                "size": order.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "fee_bp": fee_bp,
                "fee_amt": fee_amt,
                "gross_pnl": 0.0,
                "net_pnl": -fee_amt,
            })

            if self.verbose:
                print(f"[FILL ENTER] {ts} {symbol} price={price:.4f} fee_bp={fee_bp:.2f}")

        # =========================
        # 청산 (Exit)
        # =========================
        else:
            pos = self.state.positions.pop(symbol)

            entry_fee_amt = float(getattr(pos, "entry_fee_amt", 0.0))

            gross_pnl = (
                pos.side * (price - pos.entry_price)
                / pos.entry_price * pos.size
            )
            total_fee_amt = entry_fee_amt + fee_amt
            net_pnl = gross_pnl - total_fee_amt

            # cash update:
            # entry 때 이미 (size + entry_fee) 빠졌고,
            # exit 때는 (size + gross_pnl - exit_fee) 더해주면 최종이 net 반영됨
            self.state.cash += (pos.size + gross_pnl - fee_amt)

            # [복원] 청산 기록 저장
            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "EXIT",
                "side": -pos.side,
                "price": price,
                "size": pos.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "fee_bp": fee_bp,
                "fee_amt": fee_amt,
                "entry_fee_amt": entry_fee_amt,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
            })

            # [복원] 트레이드 요약 저장
            self.closed_trades.append({
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "size": pos.size,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "entry_fee_amt": entry_fee_amt,
                "exit_fee_amt": fee_amt,
                "total_fee_amt": total_fee_amt,
                "holding_sec": (ts - pos.entry_ts).total_seconds(),
                "win": net_pnl > 0,
            })

            if self.verbose:
                print(f"[FILL EXIT ] {ts} {symbol} gross={gross_pnl:.4f} net={net_pnl:.4f} fee_bp={fee_bp:.2f}")
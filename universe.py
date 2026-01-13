# hft_backtest_engine/universe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class UniverseConfig:
    lookback_days: int = 7      # ✅ 7일 기준 (원하면 바꾸면 됨) #7일 rolling window 중간값 상위 2개
    top_n: int = 2              # ✅ 상위 2개
    volume_col: str = "volume"  # 거래량 컬럼명 (klines_1m 기준)


class UniverseBuilder:
    """
    klines_1m 기반 rolling universe 생성기
    - 기준: 심볼별 '일별 거래량 합'의 lookback_days 기간 median
    - 출력: 날짜(00:00 UTC) -> [symbol1, symbol2, ...]
    """

    def __init__(self, data_loader, config: UniverseConfig):
        self.loader = data_loader
        self.config = config

    def build(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str],
    ) -> Dict[pd.Timestamp, List[str]]:
        """
        start_date, end_date: "YYYY-MM-DD"
        symbols: 유니버스 후보 심볼 리스트
        """

        daily_volumes = []

        for symbol in symbols:
            vols = self._load_daily_volume(symbol, start_date, end_date)
            if vols is not None and not vols.empty:
                daily_volumes.append(vols)

        if not daily_volumes:
            raise ValueError("No daily volume data loaded.")

        daily_df = pd.concat(daily_volumes, axis=0, ignore_index=True)

        # date는 UTC tz-aware Timestamp (00:00)
        daily_df["date"] = pd.to_datetime(daily_df["date"], utc=True)
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        universe_map: Dict[pd.Timestamp, List[str]] = {}

        all_dates = sorted(daily_df["date"].unique())

        for current_date in all_dates:
            window_start = current_date - pd.Timedelta(days=self.config.lookback_days)

            window_df = daily_df[
                (daily_df["date"] > window_start) &
                (daily_df["date"] <= current_date)
            ]

            # lookback_days가 다 차지 않으면 skip
            if window_df["date"].nunique() < self.config.lookback_days:
                continue

            med = (
                window_df.groupby("symbol")["daily_volume"]
                .median()
                .sort_values(ascending=False)
            )

            selected = med.head(self.config.top_n).index.tolist()

            # universe 적용 시점: current_date(00:00 UTC)
            universe_map[pd.Timestamp(current_date, tz="UTC")] = selected

        return universe_map

    def _load_daily_volume(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        klines_1m parquet들을 읽어서
        symbol별 date(UTC, 00:00) 단위 거래량 합계를 만든다.

        - 깨진 parquet은 스킵
        - volume_col 없으면 스킵
        """

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        days = self.loader.list_available_days(symbol, "klines_1m")
        rows = []

        for day in days:
            day_ts = pd.Timestamp(day, tz="UTC")
            if day_ts < start_ts or day_ts > end_ts:
                continue

            try:
                df = self.loader.load_klines_1m_day(symbol, day)
            except Exception as e:
                print(f"[WARN] skip klines_1m (read fail) {symbol} {day}: {e}")
                continue

            if df is None or df.empty:
                print(f"[WARN] skip klines_1m (empty) {symbol} {day}")
                continue

            if self.config.volume_col not in df.columns:
                print(f"[WARN] skip klines_1m (no {self.config.volume_col}) {symbol} {day}")
                continue

            if "open_ts" not in df.columns:
                print(f"[WARN] skip klines_1m (no open_ts) {symbol} {day}")
                continue

            # open_ts 기준 날짜 (UTC, tz-aware) -> 00:00
            try:
                date = df["open_ts"].dt.floor("D").iloc[0]
            except Exception as e:
                print(f"[WARN] skip klines_1m (bad open_ts) {symbol} {day}: {e}")
                continue

            # 거래량 합
            daily_volume = pd.to_numeric(df[self.config.volume_col], errors="coerce").sum()

            if pd.isna(daily_volume) or daily_volume < 0:
                continue

            rows.append({
                "date": date,
                "symbol": symbol,
                "daily_volume": float(daily_volume),
            })

        if not rows:
            return None

        out = pd.DataFrame(rows)

        # 같은 date/symbol 중복 방어
        out = (
            out.groupby(["date", "symbol"], as_index=False)["daily_volume"]
            .sum()
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )

        return out

    def build_from_daily_volume_parquet(
        self,
        daily_volume_path: str,
        start_date: str,
        end_date: str,
    ) -> Dict[pd.Timestamp, List[str]]:
        """
        daily_volume parquet 기반 유니버스 생성
        parquet columns: date(UTC), symbol, daily_volume
        """
        df = pd.read_parquet(daily_volume_path)

        df["date"] = pd.to_datetime(df["date"], utc=True)

        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
        df = df.dropna(subset=["daily_volume"])
        df = df[df["daily_volume"] >= 0]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("daily_volume parquet has no rows in the requested date range.")

        universe_map: Dict[pd.Timestamp, List[str]] = {}
        all_dates = sorted(df["date"].unique())

        for current_date in all_dates:
            window_start = current_date - pd.Timedelta(days=self.config.lookback_days)

            window_df = df[(df["date"] > window_start) & (df["date"] <= current_date)]

            if window_df["date"].nunique() < self.config.lookback_days:
                continue

            med = (
                window_df.groupby("symbol")["daily_volume"]
                .median()
                .sort_values(ascending=False)
            )

            selected = med.head(self.config.top_n).index.tolist()
            universe_map[current_date] = selected

        return universe_map
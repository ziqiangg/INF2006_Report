"""
WFO28.py

Updated Walk-Forward Optimisation (WFO) module aligned to EnhancedStrategy v5.4.

This file is written in notebook-cell style but packaged as a standalone module.
It expects these symbols to be available in runtime context:
  - EnhancedStrategy
  - TradingSimulation
  - Portfolio
  - STARTING_CASH
"""

from __future__ import annotations

import bisect
import itertools
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class WFOEnhancedStrategy(EnhancedStrategy):
	"""
	EnhancedStrategy v5.4 alias used exclusively by WalkForwardOptimiser.

	No trading logic is added or changed here. This subclass exists to:
	  1. Give WFO instances a distinct type from the main evaluation
		 instance and avoid accidental mutation of the primary object.
	  2. Let WalkForwardOptimiser instantiate strategy runs via a
		 self-documenting class name.
	  3. Provide a clean anchor for future WFO-specific overrides.

	All v5.4 class attributes and logic (including C10/C11/C12) are
	inherited unchanged. Per-run parameter overrides are applied as
	instance attributes inside _make_run_strategy().
	"""

	pass


class WalkForwardOptimiser:
	"""
	Walk-Forward Optimisation for EnhancedStrategy v5.4.

	Precomputes sentiment and analytics once on full development data,
	then reruns only lightweight simulation loops per combination/window.

	Grid modes:
	  - base: v2-style grid (SIZING_SCALE x BRAKE_HWM_WEEKS)
	  - expanded: v1 + v2 + v5.4 levers in a single full search
	"""

	# ── Window schedule ──────────────────────────────────────────────────
	IS_START = "2000-01-01"
	WINDOWS = [
		(2002, 2003),
		(2003, 2004),
		(2004, 2005),
		(2005, 2006),
		(2006, 2007),
		(2007, 2008),
		(2008, 2009),
		(2009, 2010),
		(2010, 2011),
		(2011, 2012),
		(2012, 2013),
		(2013, 2014),
		(2014, 2015),
		(2015, 2016),
		(2016, 2017),
	]

	# ── Base grid (legacy v2) ───────────────────────────────────────────
	SIZING_SCALE_VALUES = [1.10, 1.20, 1.30]
	BRAKE_HWM_WEEKS_VALUES = [26, 52]

	# ── Expanded grid (v1 + v2 + v5.4) ──────────────────────────────────
	TRAIL_ACTIVATE_VALUES = [0.20, 0.25, 0.30]
	TRAIL_PCT_VALUES = [0.10, 0.12, 0.15]
	MAX_HOLD_WEEKS_PATH_B_VALUES = [8, 10, 12]
	SENTIMENT_SIZE_SCALE_VALUES = [0.75, 1.00, 1.25]
	MIN_HOLD_WEEKS_VALUES = [0, 1, 2]

	# ── Fixed defaults used when not in grid ────────────────────────────
	FIXED_TRAIL_ACTIVATE = 0.20
	FIXED_TRAIL_PCT = 0.12
	FIXED_MAX_HOLD_WEEKS = 10

	# ── Thresholds ───────────────────────────────────────────────────────
	DD_DISQUALIFIER = -0.25
	REASONABLENESS_SHARPE = 1.50
	CRASH_OOS_YEARS = {2008, 2009}

	def __init__(
		self,
		strategy: WFOEnhancedStrategy,
		prices_df: pd.DataFrame,
		earnings_df: pd.DataFrame,
		grid_mode: str = "expanded",
	):
		self.strategy = strategy
		self.prices_df = prices_df
		self.earnings_df = earnings_df
		self.grid_mode = grid_mode
		self.is_results: List[dict] = []
		self.oos_results: List[dict] = []
		self._precomputed = False
		self._param_grid = self._build_grid()

		n_c = len(self._param_grid)
		n_w = len(self.WINDOWS)
		print(f"  grid_mode={self.grid_mode!r}")
		print(f"  {n_w} windows × {n_c} combos = {n_w * n_c} IS simulations")

	# ── Grid ─────────────────────────────────────────────────────────────
	def _build_grid(self) -> List[dict]:
		if self.grid_mode == "base":
			return [
				{
					"sizing_scale": ss,
					"brake_hwm_weeks": bw,
					"trail_activate_pct": self.FIXED_TRAIL_ACTIVATE,
					"trail_pct": self.FIXED_TRAIL_PCT,
					"max_hold_weeks_path_b": self.FIXED_MAX_HOLD_WEEKS,
					"sentiment_size_scale": 1.0,
					"min_hold_weeks": 1,
				}
				for ss in self.SIZING_SCALE_VALUES
				for bw in self.BRAKE_HWM_WEEKS_VALUES
			]

		# expanded full search (user-selected)
		return [
			{
				"sizing_scale": ss,
				"brake_hwm_weeks": bw,
				"trail_activate_pct": ta,
				"trail_pct": tp,
				"max_hold_weeks_path_b": mh,
				"sentiment_size_scale": sss,
				"min_hold_weeks": mhw,
			}
			for ss, bw, ta, tp, mh, sss, mhw in itertools.product(
				self.SIZING_SCALE_VALUES,
				self.BRAKE_HWM_WEEKS_VALUES,
				self.TRAIL_ACTIVATE_VALUES,
				self.TRAIL_PCT_VALUES,
				self.MAX_HOLD_WEEKS_PATH_B_VALUES,
				self.SENTIMENT_SIZE_SCALE_VALUES,
				self.MIN_HOLD_WEEKS_VALUES,
			)
		]

	# ── Precomputation ───────────────────────────────────────────────────
	def precompute(self):
		if self._precomputed:
			print("  [precompute] already done — skipping")
			return

		print("\n  [precompute] cleaning + FinBERT + analytics (full dev)...")
		self.strategy.set_data(self.prices_df.copy(), self.earnings_df.copy())
		self.strategy._precompute_sentiment(self.strategy.earnings)
		self.strategy.calculate_analytics(self.strategy.prices)

		self._precomputed = True
		print(
			f"  [precompute] done — {self.strategy.prices['ticker'].nunique()} tickers, "
			f"{len(self.strategy.sentiment_cache)} sentiment entries"
		)

	def _slice_data(self, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
		p = self.strategy.prices
		e = self.strategy.earnings
		return (
			p[(p["date"] >= start) & (p["date"] <= end)].copy(),
			e[(e["date"] >= start) & (e["date"] <= end)].copy(),
		)

	# ── Strategy instance factory ────────────────────────────────────────
	def _make_run_strategy(self, params: dict) -> WFOEnhancedStrategy:
		s = WFOEnhancedStrategy(self.strategy.finbert_pipeline)

		# Shared read-only caches
		s.sentiment_cache = self.strategy.sentiment_cache
		s._analytics_dates = self.strategy._analytics_dates
		s._analytics_records = self.strategy._analytics_records
		s.vol_regime = self.strategy.vol_regime
		s.returns_df = self.strategy.returns_df

		# Tuned parameters
		s.SIZING_SCALE = params["sizing_scale"]
		s.BRAKE_HWM_WEEKS = int(params["brake_hwm_weeks"])
		s.TRAIL_ACTIVATE_PCT = params["trail_activate_pct"]
		s.TRAIL_PCT = params["trail_pct"]
		s.MAX_HOLD_WEEKS_PATH_B = int(params["max_hold_weeks_path_b"])
		s.SENTIMENT_SIZE_SCALE = float(params["sentiment_size_scale"])
		s.MIN_HOLD_WEEKS = int(params["min_hold_weeks"])

		s._pv_history = deque([float(STARTING_CASH)], maxlen=s.BRAKE_HWM_WEEKS)
		s._dd_brake_mult = 1.00
		return s

	# ── Simulation loop ──────────────────────────────────────────────────
	@staticmethod
	def _run_sim_loop(s: WFOEnhancedStrategy, prices: pd.DataFrame, earnings: pd.DataFrame) -> dict:
		portfolio_history = []
		sim = TradingSimulation(prices, earnings, STARTING_CASH)
		all_tickers = sorted(prices["ticker"].unique())

		for week_date in sim.weekly_schedule:
			current_prices_map = sim._get_current_prices(week_date)
			portfolio_state = sim.portfolio.get_state(current_prices_map)

			current_value = portfolio_state["total_value"]
			s._pv_history.append(current_value)
			rolling_hwm = max(s._pv_history)
			dd_from_hwm = (current_value / rolling_hwm) - 1.0
			brake_mult = 0.25
			for tier_t, tier_m in s.DRAWDOWN_BRAKE_TIERS:
				if dd_from_hwm >= tier_t:
					brake_mult = tier_m
					break
			s._dd_brake_mult = brake_mult

			week_ts = pd.Timestamp(week_date)

			for ticker in all_tickers:
				price = current_prices_map.get(ticker)
				if price is None:
					continue

				dates_list = s._analytics_dates.get(ticker, [])
				records_list = s._analytics_records.get(ticker, [])
				if not dates_list:
					continue
				idx = bisect.bisect_right(dates_list, week_date) - 1
				if idx < 0:
					continue

				transcript = sim._get_recent_earnings(ticker, week_date)
				decision = s.make_decision(
					ticker,
					week_date,
					transcript,
					portfolio_state,
					records_list[idx],
				)

				if decision.startswith("BUY:"):
					parts = decision.split(":")
					target_value = int(parts[1])
					path_label = parts[2] if len(parts) > 2 else "path_a"
					shares = sim.portfolio.buy_target(
						ticker, price, week_date, target_value=target_value
					)
					if shares > 0:
						s._position_metadata[ticker] = {
							"entry_path": path_label,
							"entry_date": week_date,
							"entry_ts": week_ts,
							"high_water_price": price,
							"macd_confirmed": False,
							"entry_sentiment": s.sentiment_cache.get((ticker, str(week_date))),
						}
				elif decision == "SELL":
					sim.portfolio.sell(ticker, price, week_date)
					s._position_metadata.pop(ticker, None)

			portfolio_history.append(
				{
					"date": week_date,
					"portfolio_value": sim.portfolio.get_value(current_prices_map),
				}
			)

		return {
			"portfolio_history": portfolio_history,
			"trades": sim.portfolio.trades,
		}

	@staticmethod
	def _compute_metrics(portfolio_history: list, trades: list) -> dict:
		if len(portfolio_history) < 4:
			return {"sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0, "n_trades": 0}

		arr = np.array([h["portfolio_value"] for h in portfolio_history], dtype=float)
		w_rets = np.diff(arr) / arr[:-1]
		mean_r = float(np.mean(w_rets))
		std_r = float(np.std(w_rets, ddof=1))
		sharpe = (mean_r / std_r * np.sqrt(52)) if std_r > 1e-12 else 0.0

		peak = arr[0]
		max_dd = 0.0
		for v in arr:
			if v > peak:
				peak = v
			dd = (v - peak) / peak
			if dd < max_dd:
				max_dd = dd

		return {
			"sharpe": sharpe,
			"total_return": float((arr[-1] - arr[0]) / arr[0]),
			"max_dd": max_dd,
			"n_trades": sum(1 for t in trades if t.get("action") == "BUY"),
		}

	def _run_one_combo(self, combo_idx: int, params: dict, p_slice: pd.DataFrame, e_slice: pd.DataFrame) -> dict:
		s = self._make_run_strategy(params)
		r = self._run_sim_loop(s, p_slice, e_slice)
		m = self._compute_metrics(r["portfolio_history"], r["trades"])
		out = {
			"combo_idx": combo_idx,
			"is_sharpe": m["sharpe"],
			"is_return": m["total_return"],
			"is_max_dd": m["max_dd"],
			"is_trades": m["n_trades"],
		}
		out.update(params)
		return out

	def run_is(self, n_jobs: int = -1) -> pd.DataFrame:
		if not self._precomputed:
			raise RuntimeError("Call precompute() before run_is()")

		n_c = len(self._param_grid)
		n_w = len(self.WINDOWS)
		print(f"\n  [IS] {n_w} windows × {n_c} combos  (n_jobs={n_jobs})...")
		all_rows = []

		for win_idx, (is_end_yr, oos_yr) in enumerate(self.WINDOWS):
			win_num = win_idx + 1
			is_end = f"{is_end_yr}-12-31"
			p_sl, e_sl = self._slice_data(self.IS_START, is_end)
			n_days = p_sl["date"].nunique()
			if n_days < 200:
				print(f"    W{win_num:02d}: insufficient days ({n_days}) — skip")
				continue

			print(f"    W{win_num:02d}  IS {self.IS_START}–{is_end}  combos={n_c}...", end="", flush=True)
			results = Parallel(n_jobs=n_jobs, prefer="threads")(
				delayed(self._run_one_combo)(ci, params, p_sl, e_sl)
				for ci, params in enumerate(self._param_grid)
			)

			n_qual = sum(1 for r in results if r["is_max_dd"] >= self.DD_DISQUALIFIER)
			print(f"  {n_qual}/{n_c} qualified")

			for r in results:
				r["window"] = win_num
				r["is_end_year"] = is_end_yr
				r["oos_year"] = oos_yr
				r["qualified"] = r["is_max_dd"] >= self.DD_DISQUALIFIER
				all_rows.append(r)

		self.is_results = all_rows
		return pd.DataFrame(all_rows)

	def run_oos(self) -> pd.DataFrame:
		if not self.is_results:
			raise RuntimeError("Call run_is() before run_oos()")

		print("\n  [OOS] best qualified IS combo per window...")
		is_df = pd.DataFrame(self.is_results)
		oos_rows = []

		for win_idx, (is_end_yr, oos_yr) in enumerate(self.WINDOWS):
			win_num = win_idx + 1
			win_is = is_df[(is_df["window"] == win_num) & (is_df["qualified"])].copy()
			if len(win_is) == 0:
				print(f"    W{win_num:02d}: no qualified IS combos — skip")
				continue

			best = win_is.loc[win_is["is_sharpe"].idxmax()]
			bp = {k: best[k] for k in self._param_grid[0].keys()}

			oos_start = f"{oos_yr}-01-01"
			oos_end = f"{oos_yr}-12-31"
			p_oos, e_oos = self._slice_data(oos_start, oos_end)
			if len(p_oos) < 50:
				print(f"    W{win_num:02d}: insufficient OOS data — skip")
				continue

			s = self._make_run_strategy(bp)
			r = self._run_sim_loop(s, p_oos, e_oos)
			m = self._compute_metrics(r["portfolio_history"], r["trades"])

			row = {
				"window": win_num,
				"oos_year": oos_yr,
				"crash_window": oos_yr in self.CRASH_OOS_YEARS,
				"is_sharpe": float(best["is_sharpe"]),
				"is_max_dd": float(best["is_max_dd"]),
				"oos_sharpe": m["sharpe"],
				"oos_return": m["total_return"],
				"oos_max_dd": m["max_dd"],
				"oos_trades": m["n_trades"],
			}
			row.update(bp)
			oos_rows.append(row)

		self.oos_results = oos_rows
		return pd.DataFrame(oos_rows)

	def report(self) -> Dict[str, object]:
		if not self.oos_results:
			raise RuntimeError("Call run_oos() before report()")

		is_df = pd.DataFrame(self.is_results)
		oos_df = pd.DataFrame(self.oos_results)

		mean_oos = float(oos_df["oos_sharpe"].mean())
		mean_is = float(oos_df["is_sharpe"].mean())
		wfe = mean_oos / mean_is if mean_is > 0 else 0.0

		pct_positive = float((oos_df["oos_sharpe"] > 0).mean())
		pct_gt1 = float((oos_df["oos_sharpe"] > 1.0).mean())
		pct_no_crash = float((oos_df["oos_max_dd"] > -0.30).mean())

		group_cols = list(self._param_grid[0].keys())
		qual = is_df[is_df["qualified"]].copy()
		best_per_win = (
			qual.sort_values("is_sharpe", ascending=False)
			.groupby("window")
			.first()
			.reset_index()
		)

		freq = (
			best_per_win.groupby(group_cols)
			.agg(
				win_count=("window", "count"),
				avg_is_sharpe=("is_sharpe", "mean"),
				avg_is_dd=("is_max_dd", "mean"),
			)
			.reset_index()
			.sort_values(["win_count", "avg_is_dd"], ascending=[False, False])
			.reset_index(drop=True)
		)

		selected = freq.iloc[0][group_cols].to_dict()

		print("\n" + "=" * 82)
		print(f"  WALK-FORWARD OPTIMISATION REPORT  (v5.4, mode={self.grid_mode})")
		print("=" * 82)
		print(f"  WFE={wfe:.3f}  OOS>1.0={pct_gt1:.0%}  OOS>0={pct_positive:.0%}  DD>-30%={pct_no_crash:.0%}")
		print("  Selected params (frequency winner):")
		for k, v in selected.items():
			print(f"    {k} = {v}")

		return {
			"is_df": is_df,
			"oos_df": oos_df,
			"freq_table": freq,
			"wfe": wfe,
			"pct_oos_positive": pct_positive,
			"pct_oos_gt1": pct_gt1,
			"pct_no_crash": pct_no_crash,
			"selected_params": selected,
		}


def run_wfo_pipeline(
	finbert_pipeline,
	prices_dev: pd.DataFrame,
	earnings_dev: pd.DataFrame,
	n_jobs: int = -1,
	grid_mode: str = "expanded",
) -> Dict[str, object]:
	"""Convenience runner mirroring notebook-cell workflow."""
	print("WFOEnhancedStrategy class loaded (v5.4 — subclass of EnhancedStrategy v5.4).")
	print("WalkForwardOptimiser class loaded.")

	wfo_strategy = WFOEnhancedStrategy(finbert_pipeline)
	wfo = WalkForwardOptimiser(wfo_strategy, prices_dev, earnings_dev, grid_mode=grid_mode)

	wfo.precompute()
	wfo_is_df = wfo.run_is(n_jobs=n_jobs)
	wfo_oos_df = wfo.run_oos()
	wfo_report = wfo.report()

	return {
		"wfo": wfo,
		"wfo_is_df": wfo_is_df,
		"wfo_oos_df": wfo_oos_df,
		"wfo_report": wfo_report,
	}


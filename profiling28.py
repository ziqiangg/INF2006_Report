"""
profiling28.py

Updated profiling module for BaseStrategy vs EnhancedStrategy v5.4 CPU.

Expected runtime symbols:
  - BaseStrategy, EnhancedStrategy, TradingSimulation, Portfolio
  - STARTING_CASH, finbert_pipeline, prices_dev, prices_val,
	earnings_dev, earnings_val
"""

from __future__ import annotations

import bisect
import time
import traceback
import warnings
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _fmt(seconds: float) -> str:
	if seconds < 1:
		return f"{seconds * 1000:.0f} ms"
	if seconds < 60:
		return f"{seconds:.1f} s"
	return f"{seconds / 60:.1f} min"


class _Timer:
	def __init__(self, label, records):
		self.label = label
		self.records = records

	def __enter__(self):
		self._t0 = time.time()
		return self

	def __exit__(self, *_):
		self.records.append((self.label, time.time() - self._t0))


class ProfiledBaseStrategy(BaseStrategy):
	"""BaseStrategy wrapper with phase-level timers (logic unchanged)."""

	def __init__(self, finbert_pipeline=None):
		super().__init__(finbert_pipeline)
		self._prof_records = []

	def _reset_profile(self):
		self._prof_records = []

	def get_profile(self):
		return list(self._prof_records)

	def clean_data(self, prices_df, earnings_df):
		with _Timer("P1_clean_data", self._prof_records):
			return super().clean_data(prices_df, earnings_df)

	def calculate_analytics(self, prices_df):
		with _Timer("P2_calculate_analytics", self._prof_records):
			return super().calculate_analytics(prices_df)

	def _build_analytics_lookup(self, analytics_df):
		with _Timer("P3_analytics_lookup_build", self._prof_records):
			return super()._build_analytics_lookup(analytics_df)

	def evaluate(self, verbose=False):
		if self.prices is None or self.earnings is None:
			raise ValueError("Must call set_data() before evaluate()")

		analytics = self.calculate_analytics(self.prices)
		analytics_lookup = self._build_analytics_lookup(analytics)

		with _Timer("P4_sim_init", self._prof_records):
			sim = TradingSimulation(self.prices, self.earnings, STARTING_CASH)

		with _Timer("P5_sim_loop", self._prof_records):
			results = sim.run(
				lambda t, d, tr, ps, a: self.make_decision(t, d, tr, ps, a),
				analytics_lookup,
				verbose,
			)

		return results


class ProfiledEnhancedStrategy(EnhancedStrategy):
	"""
	EnhancedStrategy v5.4 wrapper with phase-level timers.
	Includes C10/C11/C12 behavior through inherited v5.4 logic.
	"""

	def __init__(self, finbert_pipeline=None):
		super().__init__(finbert_pipeline)
		self._prof_records = []

	def _reset_profile(self):
		self._prof_records = []

	def get_profile(self):
		return list(self._prof_records)

	def clean_data(self, prices_df, earnings_df):
		with _Timer("P1_clean_data", self._prof_records):
			return super().clean_data(prices_df, earnings_df)

	def _precompute_sentiment(self, earnings_df):
		with _Timer("P2_finbert_sentiment", self._prof_records):
			super()._precompute_sentiment(earnings_df)

	def calculate_analytics(self, prices_df):
		with _Timer("P3_calculate_analytics", self._prof_records):
			return super().calculate_analytics(prices_df)

	def _build_earnings_lookup(self, earnings_df):
		with _Timer("P4_earnings_lookup_build", self._prof_records):
			super()._build_earnings_lookup(earnings_df)

	def _build_price_dict(self, prices):
		with _Timer("P5_price_dict_build", self._prof_records):
			return super()._build_price_dict(prices)

	def evaluate(self, verbose=False):
		if self.prices is None or self.earnings is None:
			raise ValueError("Must call set_data() before evaluate()")

		self._path_counts = {}
		self._exit_reasons = {}
		self._position_metadata = {}
		self._current_week_ts = None
		self._week_ret_window = None
		self._week_positions_added = 0
		self._last_week_date = ""
		self._dd_brake_mult = 1.00
		self._pv_history = deque([float(STARTING_CASH)], maxlen=self.BRAKE_HWM_WEEKS)

		self._precompute_sentiment(self.earnings)
		self.calculate_analytics(self.prices)
		self._build_earnings_lookup(self.earnings)
		_price_dict, _ph_dates, _ph_vals = self._build_price_dict(self.prices)

		_dates_dt = pd.to_datetime(self.prices["date"])
		weekly_schedule = (
			pd.date_range(start=_dates_dt.min(), end=_dates_dt.max(), freq="W-FRI")
			.strftime("%Y-%m-%d")
			.tolist()
		)

		portfolio = Portfolio(STARTING_CASH)
		all_tickers = sorted(self.prices["ticker"].unique())

		with _Timer("P6_sim_loop", self._prof_records):
			portfolio_history = []
			for week_date in weekly_schedule:
				current_prices = {}
				for ticker in all_tickers:
					p = _price_dict.get((ticker, week_date))
					if p is None:
						dates = _ph_dates.get(ticker)
						if dates:
							idx = bisect.bisect_right(dates, week_date) - 1
							if idx >= 0:
								p = _ph_vals[ticker][idx]
					if p is not None:
						current_prices[ticker] = float(p)

				portfolio_state = portfolio.get_state(current_prices)

				current_value = portfolio_state["total_value"]
				self._pv_history.append(current_value)
				rolling_hwm = max(self._pv_history)
				dd_from_hwm = (current_value / rolling_hwm) - 1.0
				brake_mult = 0.25
				for tier_threshold, tier_mult in self.DRAWDOWN_BRAKE_TIERS:
					if dd_from_hwm >= tier_threshold:
						brake_mult = tier_mult
						break
				self._dd_brake_mult = brake_mult

				if self.returns_df is not None:
					try:
						_ret_idx = self.returns_df.index.searchsorted(week_date, side="right")
						self._week_ret_window = self.returns_df.iloc[
							max(0, _ret_idx - self.CORR_LOOKBACK) : _ret_idx
						]
					except Exception:
						self._week_ret_window = None
				else:
					self._week_ret_window = None

				for ticker in all_tickers:
					price = current_prices.get(ticker)
					if price is None:
						continue

					_ed = self._earn_dates.get(ticker)
					if _ed:
						_ei = bisect.bisect_right(_ed, week_date) - 1
						transcript = self._earn_texts[ticker][_ei] if _ei >= 0 else None
					else:
						transcript = None

					dates_list = self._analytics_dates.get(ticker, [])
					records_list = self._analytics_records.get(ticker, [])
					if not dates_list:
						continue
					idx = bisect.bisect_right(dates_list, week_date) - 1
					if idx < 0:
						continue

					decision = self.make_decision(
						ticker, week_date, transcript, portfolio_state, records_list[idx]
					)

					if decision.startswith("BUY:"):
						parts = decision.split(":")
						target_value = int(parts[1])
						path_label = parts[2] if len(parts) > 2 else "path_a"
						shares = portfolio.buy_target(
							ticker, price, week_date, target_value=target_value
						)
						if shares > 0:
							_entry_sent = self.sentiment_cache.get((ticker, str(week_date)))
							self._position_metadata[ticker] = {
								"entry_path": path_label,
								"entry_date": week_date,
								"entry_ts": self._current_week_ts,
								"high_water_price": price,
								"macd_confirmed": False,
								"entry_sentiment": _entry_sent,
							}
					elif decision == "SELL":
						portfolio.sell(ticker, price, week_date)
						self._position_metadata.pop(ticker, None)

				portfolio_history.append(
					{
						"date": week_date,
						"portfolio_value": portfolio.get_value(current_prices),
						"cash": portfolio.cash,
						"positions": len(portfolio.positions),
					}
				)

		final_date = weekly_schedule[-1]
		final_prices = {}
		for ticker in all_tickers:
			p = _price_dict.get((ticker, final_date))
			if p is None:
				dates = _ph_dates.get(ticker)
				if dates:
					idx = bisect.bisect_right(dates, final_date) - 1
					if idx >= 0:
						p = _ph_vals[ticker][idx]
			if p is not None:
				final_prices[ticker] = float(p)

		return {
			"trades": portfolio.trades,
			"portfolio_history": portfolio_history,
			"final_portfolio": portfolio.get_state(final_prices),
			"final_prices": final_prices,
		}


def _resolve_runtime_input(name: str, value):
	"""Resolve optional function arg from module globals for notebook-style usage."""
	if value is not None:
		return value
	if name in globals():
		return globals()[name]
	raise ValueError(
		f"Missing required input '{name}'. Pass it explicitly or define a global '{name}' before calling."
	)


def run_profiling_experiment(
	finbert_pipeline=None,
	prices_dev: pd.DataFrame | None = None,
	prices_val: pd.DataFrame | None = None,
	earnings_dev: pd.DataFrame | None = None,
	earnings_val: pd.DataFrame | None = None,
) -> pd.DataFrame:
	# Backward-compatible notebook mode: allow run_profiling_experiment() with no args.
	finbert_pipeline = _resolve_runtime_input("finbert_pipeline", finbert_pipeline)
	prices_dev = _resolve_runtime_input("prices_dev", prices_dev)
	prices_val = _resolve_runtime_input("prices_val", prices_val)
	earnings_dev = _resolve_runtime_input("earnings_dev", earnings_dev)
	earnings_val = _resolve_runtime_input("earnings_val", earnings_val)

	all_records = []

	print("\n" + "=" * 65)
	print("PROFILING: BaseStrategy")
	print("=" * 65)

	for split, prices_split, earnings_split in [
		("dev", prices_dev, earnings_dev),
		("val", prices_val, earnings_val),
	]:
		print(f"\n  [{split.upper()} split]")
		t_total = time.time()

		bs = ProfiledBaseStrategy(finbert_pipeline)
		bs._reset_profile()
		try:
			bs.set_data(prices_split.copy(), earnings_split.copy())
			bs.evaluate()
		except Exception as e:
			print(f"  ERROR in baseline {split}: {e}")
			traceback.print_exc()

		total_s = time.time() - t_total
		print(f"  Total {split}: {_fmt(total_s)}")
		for phase, dur in bs.get_profile():
			all_records.append({"strategy": "Baseline", "split": split, "phase": phase, "duration": dur})
		all_records.append({"strategy": "Baseline", "split": split, "phase": "TOTAL", "duration": total_s})

	print("\n" + "=" * 65)
	print("PROFILING: EnhancedStrategy v5.4 CPU")
	print("=" * 65)

	es = ProfiledEnhancedStrategy(finbert_pipeline)
	for split, prices_split, earnings_split in [
		("dev", prices_dev, earnings_dev),
		("val", prices_val, earnings_val),
	]:
		print(f"\n  [{split.upper()} split]")
		t_total = time.time()
		es._reset_profile()
		try:
			es.set_data(prices_split.copy(), earnings_split.copy())
			es.evaluate()
		except Exception as e:
			print(f"  ERROR in enhanced {split}: {e}")
			traceback.print_exc()

		total_s = time.time() - t_total
		print(f"  Total {split}: {_fmt(total_s)}")
		for phase, dur in es.get_profile():
			all_records.append({"strategy": "Enhanced", "split": split, "phase": phase, "duration": dur})
		all_records.append({"strategy": "Enhanced", "split": split, "phase": "TOTAL", "duration": total_s})

	return pd.DataFrame(all_records)


def display_profiling_results(df: pd.DataFrame):
	pivot = df.pivot_table(
		index="phase",
		columns=["strategy", "split"],
		values="duration",
		aggfunc="first",
	)

	phase_order = [
		"P1_clean_data",
		"P2_calculate_analytics",
		"P2_finbert_sentiment",
		"P3_analytics_lookup_build",
		"P3_calculate_analytics",
		"P4_sim_init",
		"P4_earnings_lookup_build",
		"P5_price_dict_build",
		"P5_sim_loop",
		"P6_sim_loop",
		"TOTAL",
	]
	present = [p for p in phase_order if p in pivot.index]
	for p in pivot.index:
		if p not in present:
			present.append(p)
	pivot = pivot.reindex(present)

	def _get(strategy, split, phase):
		try:
			v = pivot.loc[phase, (strategy, split)]
			return _fmt(float(v)) if not pd.isna(v) else "—"
		except Exception:
			return "—"

	print("\n" + "=" * 90)
	print("PROFILING RESULTS — BaseStrategy vs EnhancedStrategy v5.4 CPU")
	print("=" * 90)
	print(
		f"  {'Phase':<42}  {'Base Dev':>10}  {'Enh Dev':>10}  "
		f"{'Base Val':>10}  {'Enh Val':>10}  {'Speedup Dev':>12}"
	)
	print("  " + "─" * 86)

	for phase in present:
		bd = _get("Baseline", "dev", phase)
		ed = _get("Enhanced", "dev", phase)
		bv = _get("Baseline", "val", phase)
		ev = _get("Enhanced", "val", phase)

		try:
			bd_s = float(pivot.loc[phase, ("Baseline", "dev")])
			ed_s = float(pivot.loc[phase, ("Enhanced", "dev")])
			ratio = f"{bd_s / ed_s:.1f}×" if ed_s > 0 and not pd.isna(bd_s) else "—"
		except Exception:
			ratio = "—"

		print(f"  {phase:<42}  {bd:>10}  {ed:>10}  {bv:>10}  {ev:>10}  {ratio:>12}")

	print("=" * 90)

	chart_phases = []
	chart_base = []
	chart_enh = []
	phase_pairs = [
		("P1_clean_data", "P1_clean_data", "clean_data"),
		("P2_calculate_analytics", "P2_finbert_sentiment", "analytics / FinBERT"),
		("P3_analytics_lookup_build", "P3_calculate_analytics", "lookup / analytics"),
		("P4_sim_init", "P4_earnings_lookup_build", "sim init / earn lookup"),
		("P5_sim_loop", "P6_sim_loop", "sim loop"),
		("TOTAL", "TOTAL", "TOTAL"),
	]

	for bp, ep, lbl in phase_pairs:
		bval = float(pivot.loc[bp, ("Baseline", "dev")]) if bp in pivot.index else np.nan
		eval_ = float(pivot.loc[ep, ("Enhanced", "dev")]) if ep in pivot.index else np.nan
		if not np.isnan(bval) or not np.isnan(eval_):
			chart_phases.append(lbl)
			chart_base.append(0 if np.isnan(bval) else bval)
			chart_enh.append(0 if np.isnan(eval_) else eval_)

	if not chart_phases:
		print("  [chart] no data to plot")
		return

	fig, ax = plt.subplots(figsize=(12, max(4, len(chart_phases) * 0.9 + 1.5)))
	y = np.arange(len(chart_phases))
	h = 0.35
	eps = 1e-3
	base_vals = np.array(chart_base, dtype=float)
	enh_vals = np.array(chart_enh, dtype=float)

	ax.barh(y + h / 2, np.clip(base_vals, eps, None), height=h, color="#888780", alpha=0.85, label="BaseStrategy")
	ax.barh(y - h / 2, np.clip(enh_vals, eps, None), height=h, color="#1D9E75", alpha=0.85, label="Enhanced v5.4")

	ax.set_xscale("log")
	ax.set_yticks(y)
	ax.set_yticklabels(chart_phases, fontsize=10)
	ax.invert_yaxis()
	ax.set_xlabel("Duration (seconds, log scale)")
	ax.set_title("Profiling: BaseStrategy vs EnhancedStrategy v5.4 CPU — Dev split", fontsize=11, fontweight="bold")
	ax.legend(fontsize=9)
	ax.xaxis.set_major_formatter(
		mticker.FuncFormatter(lambda x, _: _fmt(x) if x >= 1e-2 else f"{x*1000:.0f}ms")
	)
	ax.grid(True, axis="x", alpha=0.3, which="both")
	plt.tight_layout()
	plt.show()


def run_full_profiling_pipeline(
	finbert_pipeline=None,
	prices_dev: pd.DataFrame | None = None,
	prices_val: pd.DataFrame | None = None,
	earnings_dev: pd.DataFrame | None = None,
	earnings_val: pd.DataFrame | None = None,
) -> pd.DataFrame:
	print("\n" + "=" * 65)
	print("STRATEGY PROFILING EXPERIMENT")
	print("Comparing BaseStrategy vs EnhancedStrategy v5.4 CPU")
	print("=" * 65)

	profiling_df = run_profiling_experiment(
		finbert_pipeline=finbert_pipeline,
		prices_dev=prices_dev,
		prices_val=prices_val,
		earnings_dev=earnings_dev,
		earnings_val=earnings_val,
	)
	display_profiling_results(profiling_df)
	return profiling_df


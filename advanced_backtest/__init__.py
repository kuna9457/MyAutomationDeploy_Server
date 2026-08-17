"""
advanced_backtest — find which (symbol, pattern) combinations actually work.

A search layer ON TOP of backtester.run_backtest. It adds no simulation logic
of its own: every number it reports comes from a real backtest, so a result
here and the same run in the Backtest tab cannot disagree.

Separate from the ordinary backtest path by design — see ADVANCED_BACKTEST_PLAN.md.
"""

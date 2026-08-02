"""
Strategy plugins. One file per strategy; drop a file in here and it appears in
the UI dropdown, the engine and the backtester automatically.

Kept deliberately EMPTY. strategy._load_strategy_plugins() imports each module in
this folder directly, and doing the importing here instead would create a real
import cycle (strategy -> strategies/__init__ -> plugin -> strategy).

Modules whose name starts with "_" are skipped by the loader — use that prefix
for shared helpers that are not themselves strategies.
"""

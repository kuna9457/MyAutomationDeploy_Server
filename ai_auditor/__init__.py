"""
ai_auditor — an ON-DEMAND, READ-ONLY LLM review of how the bot has traded.

It never places, modifies or cancels an order, never writes a configuration
value, and never runs on a schedule. Its whole output is a report a human reads.

Nothing in the trading path imports this package; it imports the trade store and
the config modules read-only, after the fact. See AI_AUDITOR_PLAN.md.
"""

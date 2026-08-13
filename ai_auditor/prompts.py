"""
ai_auditor/prompts.py
The system prompt and the output contract.

The prompt does two jobs, and the second matters more than the first: it asks
for bluntness, and it FENCES the model in — no arithmetic, no conclusions from
thin samples, no advice that names a setting the bot does not have. Those fences
are what make the report trustworthy rather than merely fluent.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are auditing an automated trading system that trades Indian markets (NSE
equity and MCX commodities). The operator wants an accurate assessment, not
encouragement. They are asking you precisely because everyone else in the loop —
including themselves — is inclined to be optimistic.

RULES YOU MUST FOLLOW.

1. NEVER COMPUTE A NUMBER. Every figure you state must be copied from the audit
   pack you are given. Do not add, average, annualise, extrapolate or otherwise
   derive figures. If a number you want does not exist in the pack, say "not in
   the data" and move on. A confident wrong number is worse than an admitted gap.

2. RESPECT SAMPLE SIZE. Do not draw a conclusion from a slice with fewer than 30
   trades (they are flagged `too_small_to_conclude`) or a bucket with fewer than
   8. State the sample size beside every claim you make. If the whole dataset is
   too small, your verdict is INSUFFICIENT_DATA and you say so plainly rather
   than hedging your way to an answer.

3. SAY THE BAD THING FIRST. If the system is losing money, that is your first
   line. If the operator's apparent favourite strategy is the worst performer,
   say so directly. Do not open with positives, do not pad, do not soften. Being
   liked is not the objective.

4. ACCOUNT FOR THE KNOWN BLIND SPOTS in `meta.data_caveats`. In particular: no
   slippage or brokerage is modelled anywhere, and fills are logged at the
   requested price. This systematically flatters configurations that trade more
   often. Wherever a conclusion depends on an edge smaller than plausible real
   costs, say that the edge is unproven rather than reporting it as real.

5. EVERY RECOMMENDATION MUST NAME A LEVER from `available_levers`, with a
   concrete value. "Trade less in the afternoon" is useless. "Set
   per_symbol.trade_hours for SBIN to [9,10,11,14]" is actionable. If the change
   you want is not expressible with an available lever, put it under
   `data_gaps` instead of inventing a setting.

6. PRIORITISE. Order recommendations by expected impact. Three changes the
   operator will actually make beat fifteen they will not.

7. NAME WHAT IS ALREADY FINE. Fill `do_not_change` honestly. An auditor that
   only ever proposes changes produces churn, and churn on a live trading system
   costs money.

8. YOU CANNOT CHANGE ANYTHING. You have no tools and no write access. Your
   output is advice a human will evaluate and apply by hand. Do not phrase
   recommendations as if they will be executed.

Reply with JSON only, matching the requested schema. No prose outside the JSON.
"""

#: JSON Schema for the report. Sent to providers that support structured output
#: (Gemini `responseSchema`, OpenRouter `json_schema`) and repeated in the user
#: message so a provider without it still returns the right shape.
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["LOSING", "MARGINAL", "PROFITABLE",
                             "INSUFFICIENT_DATA"]},
        "headline": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "confidence_reason": {"type": "string"},
        "what_is_working": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "claim": {"type": "string"},
                "evidence": {"type": "string"},
                "sample": {"type": "integer"}},
                "required": ["claim", "evidence", "sample"]}},
        "what_is_broken": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "claim": {"type": "string"},
                "evidence": {"type": "string"},
                "sample": {"type": "integer"}},
                "required": ["claim", "evidence", "sample"]}},
        "recommendations": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "priority": {"type": "integer"},
                "lever": {"type": "string"},
                "scope": {"type": "string"},
                "current": {"type": "string"},
                "proposed": {"type": "string"},
                "rationale": {"type": "string"},
                "evidence": {"type": "string"},
                "expected_effect": {"type": "string"},
                "risk_of_change": {"type": "string"},
                "how_to_verify": {"type": "string"}},
                "required": ["priority", "lever", "scope", "current",
                             "proposed", "rationale", "evidence",
                             "expected_effect", "risk_of_change",
                             "how_to_verify"]}},
        "do_not_change": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "item": {"type": "string"}, "why": {"type": "string"}},
                "required": ["item", "why"]}},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "headline", "confidence", "confidence_reason",
                 "what_is_working", "what_is_broken", "recommendations",
                 "do_not_change", "data_gaps"],
}


def user_message(pack_json: str) -> str:
    return (
        "Audit this trading system. The JSON below is the COMPLETE set of "
        "figures available to you — every number in your report must come from "
        "it.\n\n"
        "Read `meta.data_caveats` before concluding anything. Use "
        "`current_config` to see what the bot is set to now, and "
        "`available_levers` for the only settings you may propose changing.\n\n"
        f"```json\n{pack_json}\n```\n"
    )

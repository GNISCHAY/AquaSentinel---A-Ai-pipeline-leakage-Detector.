"""
Water savings estimation (Member 4 - Integration & Product Developer).

The estimated leak flow comes from the INLET FLOW METER: extra water entering
the network compared with what the digital twin expected. This mirrors the
"water balance" idea utilities use (water in - water billed = losses).

All outputs are SIMULATED ESTIMATES, not measured real-world savings.
"""

from aquasentinel import config

SECONDS_PER_DAY = 86_400


def litres_per_second_to_m3_per_day(flow_lps):
    return flow_lps * SECONDS_PER_DAY / 1000.0


def estimate_savings(leak_flow_lps, cost_per_m3=config.DEFAULT_COST_PER_M3,
                     currency=config.DEFAULT_CURRENCY_LABEL):
    """
    Water (and money) saved if the leak is repaired, assuming the leak would
    otherwise keep flowing at the same rate.
    """
    leak_flow_lps = max(float(leak_flow_lps), 0.0)
    daily_m3 = litres_per_second_to_m3_per_day(leak_flow_lps)
    monthly_m3 = daily_m3 * config.DAYS_PER_MONTH
    return {
        "leak_flow_lps": round(leak_flow_lps, 2),
        "daily_m3": round(daily_m3, 1),
        "monthly_m3": round(monthly_m3, 0),
        "daily_litres": round(daily_m3 * 1000, 0),
        "monthly_cost": round(monthly_m3 * cost_per_m3, 0),
        "currency": currency,
        "label": "Simulated estimate - not a measured saving",
    }

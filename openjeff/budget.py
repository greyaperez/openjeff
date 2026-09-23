"""Cash preflight arithmetic. This does NOT stop a cloud instance or billing."""
from decimal import Decimal, InvalidOperation


def budget_preflight(*, spent, committed, hourly_rate, hours, overhead, cap="200", reserve="50"):
    try:
        values = [Decimal(str(x)) for x in (spent, committed, hourly_rate, hours, overhead, cap, reserve)]
    except InvalidOperation as e:
        raise ValueError("invalid money/time value") from e
    if any(not x.is_finite() or x < 0 for x in values):
        raise ValueError("budget values must be finite and nonnegative")
    spent, committed, rate, hours, overhead, cap, reserve = values
    if reserve > cap:
        raise ValueError("reserve exceeds cap")
    liability = spent + committed + rate * hours + overhead
    return {"estimated_liability_usd": str(liability), "hard_cap_usd": str(cap),
            "planned_stop_usd": str(cap - reserve), "within_hard_cap": liability <= cap,
            "within_planned_stop": liability <= cap - reserve,
            "remaining_to_hard_cap_usd": str(cap - liability),
            "provider_termination_enforced": False}


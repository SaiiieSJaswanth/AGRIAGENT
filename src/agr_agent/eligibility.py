def check_rule(rule, profile):
    field = rule.get("field")
    op = rule.get("operator")
    expected = rule.get("value")
    actual = profile.get(field)

    if actual is None:
        return None
    if op == "equals":
        return actual == expected
    if op == "not_equals":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "gte":
        return float(actual) >= float(expected)
    if op == "lte":
        return float(actual) <= float(expected)
    if op == "truthy":
        return bool(actual) is True
    return None

def evaluate_scheme(scheme, profile):
    results = [check_rule(r, profile) for r in scheme.get("eligibility_rules", [])]
    if not results:
        return "potentially_relevant", []
    if any(x is False for x in results):
        return "not_matched", results
    if any(x is None for x in results):
        return "needs_verification", results
    return "potentially_eligible", results

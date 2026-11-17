def match_loans(profile, schemes):
    if not profile.loan_required:
        return []
    return [
        s for s in schemes
        if s.get("support_category") in {"credit", "infrastructure_finance"}
    ]

def match_subsidies(profile, schemes):
    needs = []
    if profile.irrigation_required: needs.append("irrigation")
    if profile.equipment_required: needs.append("equipment")
    if profile.subsidy_required: needs.append("subsidy")
    if not needs:
        return []
    return [
        s for s in schemes
        if any(n in s.get("support_tags", []) for n in needs)
    ]

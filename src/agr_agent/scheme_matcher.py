from src.agr_agent.eligibility import evaluate_scheme

def match_schemes(profile, schemes):
    profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
    matches = []
    for scheme in schemes:
        # State schemes apply only to the selected state; central schemes remain available nationally.
        if scheme.get("scheme_level") == "STATE" and scheme.get("state") != profile_dict.get("state"):
            continue
        status, evidence = evaluate_scheme(scheme, profile_dict)
        if status != "not_matched":
            matches.append({
                "scheme_name": scheme["scheme_name"],
                "scheme_level": scheme.get("scheme_level", "CENTRAL"),
                "state": scheme.get("state"),
                "status": status,
                "benefit_type": scheme.get("benefit_type"),
                "benefit_description": scheme.get("benefit_description"),
                "required_documents": scheme.get("required_documents", []),
                "application_method": scheme.get("application_method"),
                "official_source": scheme.get("official_source"),
                "source_url": scheme.get("source_url"),
                "last_verified": scheme.get("last_verified"),
                "evidence": evidence,
            })
    return matches

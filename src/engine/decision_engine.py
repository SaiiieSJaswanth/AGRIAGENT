from src.models.risk import analyze_risk
from src.agr_agent.scheme_matcher import match_schemes
from src.agr_agent.document_checker import collect_documents

def build_decision(farmer_profile, local_context, ml_results, schemes):
    matches = match_schemes(farmer_profile, schemes)
    return {
        "farmer_profile": farmer_profile.to_dict(),
        "local_context": local_context,
        "ml_results": ml_results,
        "support_matches": matches,
        "documents": collect_documents(matches),
        "risk": ml_results.get("risk", {}),
        "limitations": [
            "ML outputs are data-driven estimates, not agronomic guarantees.",
            "Government eligibility, benefits and application status require official verification.",
            "Yield prediction is unavailable because the supplied dataset has no real yield target."
        ]
    }

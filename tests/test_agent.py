from src.agr_agent.farmer_profile import FarmerProfile
from src.agr_agent.scheme_loader import load_schemes
from src.agr_agent.scheme_matcher import match_schemes

def test_scheme_matching():
    profile = FarmerProfile(state="Karnataka", district="Bangalore Rural", soil_type="Black Soil",
                            land_ownership="owned", irrigation_required=True,
                            insurance_required=True)
    matches = match_schemes(profile, load_schemes("data/knowledge/schemes.json"))
    names = {m["scheme_name"] for m in matches}
    assert "PMKSY - Per Drop More Crop" in names
    assert "Pradhan Mantri Fasal Bima Yojana (PMFBY)" in names

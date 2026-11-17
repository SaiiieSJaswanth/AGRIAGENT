"""
AGRIAGENT Production Model Training & Visualization Pipeline
Trains all 4 verified models on real empirical datasets:
1. Crop Recommendation Shadow ML (Random Forest on APY Data)
2. Soil Health Assessment ML (Random Forest on Soil Health Card Data)
3. Fertilizer Regime ML (Random Forest on ICAR-STCR Data)
4. Irrigation Schedule ML (Decision Tree on FAO-56/IMD Water Balance Data)
Generates all 32 mandatory diagnostic visualizations at 300 DPI.
"""
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("AGRIAGENT — Full Production Model Training & Visualizations Pipeline")
    print("=" * 80)
    scripts_dir = Path(__file__).parent / "scripts"
    train_script = scripts_dir / "train_and_generate_32_visualizations.py"
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        sys.exit(1)

    import subprocess
    cmd = [sys.executable, str(train_script)]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\nAll 4 ML models trained and 32 diagnostic plots generated successfully.")
    else:
        print(f"\nTraining pipeline exited with return code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()

# scripts/clean_and_organize_repo.py
import os
import shutil
import re

def clean_project(project_dir):
    print(f"Cleaning project at {project_dir}...")
    
    # 1. Create docs/ folder
    docs_dir = os.path.join(project_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Files to move to docs/
    to_docs = [
        "DATA_DICTIONARY.md",
        "DATA_SOURCE.md",
        "MODEL_CARD.md",
        "MODEL_LIMITATIONS.md",
        "REPRODUCIBILITY.md"
    ]
    for f in to_docs:
        src_path = os.path.join(project_dir, f)
        dst_path = os.path.join(docs_dir, f)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)
            print(f"Moved {f} to docs/")
            
    # Files to delete (AI prompt artifacts and scratch logs)
    to_delete = [
        "CROP_ML_IMPLEMENTATION_GUIDE.txt",
        "DATA_QUALITY_REPORT.md",
        "EXPERIMENT_RESULTS.md",
        "FINAL_AUDIT_REPORT.md",
        "FINAL_PROJECT_AUDIT_REPORT.md",
        "Fix User Input Validation.txt",
        "FIXES_APPLIED.md",
        "PROJECT_INITIAL_AUDIT.md",
        "PROJECT_OUTPUTS_AND_MODELS.txt",
        "SCHEMES.txt"
    ]
    for f in to_delete:
        p = os.path.join(project_dir, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed AI artifact: {f}")
            
    # Remove AGRIAGENT_DOCUMENTATION if present (redundant with docs/)
    redundant_doc_dir = os.path.join(project_dir, "AGRIAGENT_DOCUMENTATION")
    if os.path.exists(redundant_doc_dir):
        shutil.rmtree(redundant_doc_dir)
        print("Removed redundant AGRIAGENT_DOCUMENTATION folder.")
        
    # Remove any test caches or pycache
    for root, dirs, files in os.walk(project_dir):
        for d in dirs:
            if d in ('__pycache__', '.pytest_cache'):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print("Project directory cleaned.")

if __name__ == "__main__":
    for d in [
        r"D:\AIML_CASESTUDY\AGRIAGENT_CROP_ML_HYBRID",
        r"D:\AIML_CASESTUDY\AGRIAGENT_FINAL_COMPLETE\AGRIAGENT_CROP_ML_HYBRID"
    ]:
        if os.path.exists(d):
            clean_project(d)

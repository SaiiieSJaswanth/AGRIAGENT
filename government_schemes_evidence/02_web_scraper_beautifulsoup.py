# ====================================================================================================
# AGRIAGENT — OFFICIAL GOVERNMENT SCHEMES WEB SCRAPER (BEAUTIFUL SOUP & REQUESTS)
# ====================================================================================================
# Purpose: Demonstrates automated web extraction of agricultural welfare schemes, eligibility rules,
#          subsidy percentages, required documents, and official portal URLs.
# Libraries Used: requests, beautifulsoup4 (bs4), re, json, datetime
# Academic Module: AGRIAGENT Module 5 (Government Welfare Scheme Intelligence)
# ====================================================================================================

import re
import json
import time
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Referer": "https://india.gov.in/",
    "Connection": "keep-alive"
}

SCHEME_TARGETS = [
    {
        "name": "PM-KISAN",
        "url": "https://pmkisan.gov.in/",
        "level": "CENTRAL",
        "ministry": "Ministry of Agriculture & Farmers Welfare"
    },
    {
        "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "url": "https://pmfby.gov.in/",
        "level": "CENTRAL",
        "ministry": "Ministry of Agriculture & Farmers Welfare"
    },
    {
        "name": "Soil Health Card Scheme",
        "url": "https://soilhealth.dac.gov.in/",
        "level": "CENTRAL",
        "ministry": "Ministry of Agriculture & Farmers Welfare (DA&FW)"
    },
    {
        "name": "Kisan Credit Card (KCC)",
        "url": "https://www.myscheme.gov.in/schemes/kcc",
        "level": "CENTRAL",
        "ministry": "Ministry of Finance / DA&FW"
    },
    {
        "name": "Karnataka Krishi Bhagya & Raitha Shakti",
        "url": "https://raitamitra.karnataka.gov.in/",
        "level": "STATE",
        "state": "Karnataka",
        "ministry": "Department of Agriculture, Govt of Karnataka"
    }
]

def clean_text(raw_text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', raw_text)
    return cleaned.strip()

def scrape_scheme_page(target: dict) -> dict:
    print(f"[+] Scraping: {target['name']} from {target['url']}...")
    extracted_data = {
        "scheme_name": target["name"],
        "portal_url": target["url"],
        "level": target["level"],
        "ministry": target.get("ministry", "Government of India"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending",
        "headings": [],
        "key_paragraphs": [],
        "bullet_points": []
    }

    try:
        response = requests.get(target["url"], headers=REQUEST_HEADERS, timeout=12)
        extracted_data["http_status_code"] = response.status_code
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            for elem in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                elem.extract()
            
            if soup.title:
                extracted_data["page_title"] = clean_text(soup.title.string or "")
            
            for h in soup.find_all(["h1", "h2", "h3"]):
                text = clean_text(h.get_text())
                if 5 < len(text) < 150:
                    extracted_data["headings"].append(text)
            
            for p in soup.find_all("p"):
                text = clean_text(p.get_text())
                if len(text) > 30 and not text.startswith("Copyright") and not text.startswith("All rights"):
                    extracted_data["key_paragraphs"].append(text)
            
            for li in soup.find_all("li"):
                text = clean_text(li.get_text())
                if 15 < len(text) < 250:
                    extracted_data["bullet_points"].append(text)
                    
            extracted_data["status"] = "success"
            print(f"    [OK] Extracted {len(extracted_data['headings'])} headings, {len(extracted_data['key_paragraphs'])} paragraphs, {len(extracted_data['bullet_points'])} bullets.")
        else:
            extracted_data["status"] = f"HTTP Error {response.status_code}"
            print(f"    [WARN] Server responded with status {response.status_code}")
            
    except Exception as exc:
        extracted_data["status"] = f"Network Exception: {str(exc)}"
        print(f"    [WARN] Live request notice: {exc}")

    return extracted_data

def run_scraper():
    print("=" * 80)
    print("AGRIAGENT GOVERNMENT SCHEME WEB SCRAPER (BeautifulSoup4 / requests)")
    print("=" * 80)
    results = []
    for target in SCHEME_TARGETS:
        res = scrape_scheme_page(target)
        results.append(res)
        time.sleep(1.0)
        
    out_dir = Path(__file__).parent
    output_path = out_dir / "scraped_schemes_sample_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Web scraping process complete. Sample structured JSON written to: {output_path}")

if __name__ == "__main__":
    run_scraper()

# AGRIAGENT Model and Evidence Limitations

## Crop shortlist

Crop ranking is not an ML classifier. It is a transparent lookup over the APY historical crop area/production dataset (1997–2015), preferring the selected district and season, and falling back to broader regional records if local coverage is sparse. Ranking is based on mean recorded cultivated area per observation over the latest five available years in the selected slice. It indicates what was historically cultivated, not a guaranteed recommendation for the next season.

- APY data in this ZIP ends in 2015 and may use historical district names that differ from present-day names.
- Historical cultivated area does not encode the farmer's exact soil type/test, seed variety, pests, water access, cost, or market contract. The APY dataset has no farm-level soil labels, so soil inputs do not alter the crop ranking; they feed separate soil/fertilizer advisories.
- A state crop catalogue is only a reference list, not a soil- or season-specific suitability result.
- No confidence percentages are reported for this lookup because no calibrated probability model is involved.
- Google results can differ by query and source; exact match is not technically guaranteeable.

## Historical yield model

The yield model is trained on district/season historical APY data. It estimates a statistical relationship in historical records, not a farm-level forecast. It does not include real-time weather, satellite observations, soil moisture, pest incidence, or future price shocks. Area is a model input and yield is derived from production divided by area, so interpretation and evaluation should be cautious.

## Soil fertility

The production rating is rule-based and transparent. Soil sample data is retained for reference and validation; derived thresholds should not be mistaken for independently verified expert labels. Enter measured Soil Health Card values whenever possible. Defaults are placeholders, not measured values.

## Rainfall and prices

- IMD normals represent long-term climate normals, not current weather forecasts.
- The packaged AGMARKNET sample is historical (2012) and must not be represented as live mandi prices.

## Government schemes

Matches indicate potentially relevant programs only. Eligibility, dates, required documents, and approvals must be checked with official portals or local departments.

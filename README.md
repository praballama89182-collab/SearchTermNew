# Placement Intelligence Tool

Streamlit app that analyzes Amazon Sponsored Products **Placement reports**
(Top of Search / Rest of Search / Product Pages / Off Amazon) and recommends
the best placement per campaign, with bid-action suggestions.

## What it does

- **Overview** — account-level spend/sales/ACOS/ROAS, spend by placement, daily trend
- **Placement Comparison** — aggregated CTR/CVR/CPC/ACOS/ROAS across all 4 placement types, plus spend mix by portfolio
- **Campaign Deep-Dive** — pick any campaign, see its placement breakdown and daily trend
- **Recommendations** — for every campaign, ranks placements by ACOS or ROAS (your choice),
  only among placements that clear minimum click/spend thresholds you set, and flags:
  `Increase bid / shift budget here`, `Decrease bid — spend, no sales`, `Maintain / monitor`,
  or `Insufficient data`. Exportable as an .xlsx with one row per campaign×placement plus a
  "best only" sheet.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload your placement report (.xlsx) — same
column format as the Amazon Ads console export (Date, Portfolio name, Campaign Name,
Placement, Impressions, Clicks, Spend, 7 Day Total Sales, 7 Day Total Orders, 7 Day Total Units, etc.).

## Hosting options

**Streamlit Community Cloud (free, fastest)**
1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io → "New app" → point it at `app.py`.
3. You get a public `*.streamlit.app` URL. Good for internal team use; anyone
   with the link can open it and upload their own report (no data is stored server-side,
   it's processed in-memory per session).

**Same pattern as your Master Shipment Intelligence Tool repo**
Since you already have `ayushr-beep/virventures-shipment-intelligence` on
Community Cloud, you can add this as a second app in the same GitHub org, or
as a second page inside that repo using Streamlit's multipage app structure
(`pages/placement_intelligence.py`) so both tools share one deployment and one URL.

**Self-hosted (Render / EC2 / internal server)**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Put it behind nginx or a load balancer if you want a custom domain / auth.

## Notes on the recommendation logic

- ACOS/ROAS are recomputed from raw Spend/Sales/Clicks/Impressions rather than
  trusting the report's per-row ACOS/ROAS columns, since those go blank/NaN on
  zero-sales rows and can't be aggregated by simple averaging.
- A placement only becomes eligible for "best" once it clears your min-clicks
  and min-spend thresholds — this avoids recommending a placement based on
  3 lucky clicks.
- "Decrease bid" only fires when a placement has real spend and clicks but zero sales —
  it won't flag placements that just don't have enough data yet.

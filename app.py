"""
Amazon Sponsored Products — Placement Intelligence Tool
---------------------------------------------------------
Analyzes Amazon Ads placement reports (Top of Search / Rest of Search /
Product Pages / Off Amazon) to identify the best-performing placement per
campaign and recommend bid-adjustment actions.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Placement Intelligence Tool",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

RENAME_MAP = {
    "7 Day Total Sales": "Sales",
    "Total Advertising Cost of Sales (ACOS)": "ACOS_reported",
    "Total Return on Advertising Spend (ROAS)": "ROAS_reported",
    "7 Day Total Orders (#)": "Orders",
    "7 Day Total Units (#)": "Units",
    "Cost Per Click (CPC)": "CPC_reported",
}

REQUIRED_COLS = [
    "Date", "Portfolio name", "Campaign Name", "Placement",
    "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
]


@st.cache_data(show_spinner=False)
def load_report(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_MAP)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Report is missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived metrics (CTR, CVR, CPC, ACOS, ROAS) to an aggregated frame."""
    out = df.copy()
    out["CTR"] = np.where(out["Impressions"] > 0, out["Clicks"] / out["Impressions"], np.nan)
    out["CVR"] = np.where(out["Clicks"] > 0, out["Orders"] / out["Clicks"], np.nan)
    out["CPC"] = np.where(out["Clicks"] > 0, out["Spend"] / out["Clicks"], np.nan)
    out["ACOS"] = np.where(out["Sales"] > 0, out["Spend"] / out["Sales"], np.nan)
    out["ROAS"] = np.where(out["Spend"] > 0, out["Sales"] / out["Spend"], np.nan)
    return out


def aggregate(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            Impressions=("Impressions", "sum"),
            Clicks=("Clicks", "sum"),
            Spend=("Spend", "sum"),
            Sales=("Sales", "sum"),
            Orders=("Orders", "sum"),
            Units=("Units", "sum"),
        )
    )
    return compute_metrics(agg)


def recommend_placements(df: pd.DataFrame, min_clicks: int, min_spend: float, rank_metric: str) -> pd.DataFrame:
    """
    For each campaign, ranks placements by the chosen metric among those that
    clear the min click/spend thresholds, and flags a bid action.
    """
    camp_place = aggregate(df, ["Campaign Name", "Portfolio name", "Placement"])
    camp_totals = aggregate(df, ["Campaign Name"]).set_index("Campaign Name")

    rows = []
    for camp, g in camp_place.groupby("Campaign Name"):
        eligible = g[(g["Clicks"] >= min_clicks) & (g["Spend"] >= min_spend)]
        portfolio = g["Portfolio name"].iloc[0]
        total_spend = camp_totals.loc[camp, "Spend"] if camp in camp_totals.index else g["Spend"].sum()

        if eligible.empty:
            for _, r in g.iterrows():
                rows.append({
                    "Campaign Name": camp, "Portfolio name": portfolio,
                    "Placement": r["Placement"], "Spend": r["Spend"], "Sales": r["Sales"],
                    "Clicks": r["Clicks"], "Orders": r["Orders"],
                    "ACOS": r["ACOS"], "ROAS": r["ROAS"], "CVR": r["CVR"], "CTR": r["CTR"],
                    "Total Campaign Spend": total_spend,
                    "Is Best": False, "Action": "Insufficient data",
                })
            continue

        if rank_metric == "ACOS (lower is better)":
            eligible = eligible.sort_values("ACOS", ascending=True, na_position="last")
        else:
            eligible = eligible.sort_values("ROAS", ascending=False, na_position="last")

        best_placement = eligible.iloc[0]["Placement"]

        for _, r in g.iterrows():
            is_best = (r["Placement"] == best_placement) and (r["Placement"] in eligible["Placement"].values)
            if r["Placement"] not in eligible["Placement"].values:
                action = "Insufficient data"
            elif is_best:
                action = "Increase bid / shift budget here"
            elif r["Clicks"] >= min_clicks and (pd.isna(r["ACOS"]) or r["ACOS"] > 0.5) and r["Sales"] == 0:
                action = "Decrease bid — spend, no sales"
            else:
                action = "Maintain / monitor"

            rows.append({
                "Campaign Name": camp, "Portfolio name": portfolio,
                "Placement": r["Placement"], "Spend": r["Spend"], "Sales": r["Sales"],
                "Clicks": r["Clicks"], "Orders": r["Orders"],
                "ACOS": r["ACOS"], "ROAS": r["ROAS"], "CVR": r["CVR"], "CTR": r["CTR"],
                "Total Campaign Spend": total_spend,
                "Is Best": is_best, "Action": action,
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("📊 Placement Intelligence")
uploaded = st.sidebar.file_uploader("Upload Placement Report (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.title("Amazon Sponsored Products — Placement Intelligence Tool")
    st.info("👈 Upload a Sponsored Products Placement report (.xlsx) to begin. "
            "Expected columns include Date, Campaign Name, Placement, Impressions, "
            "Clicks, Spend, 7 Day Total Sales, 7 Day Total Orders, 7 Day Total Units.")
    st.stop()

try:
    raw = load_report(uploaded)
except Exception as e:
    st.error(f"Couldn't read this file: {e}")
    st.stop()

min_date, max_date = raw["Date"].min().date(), raw["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

portfolios = sorted(raw["Portfolio name"].dropna().unique())
selected_portfolios = st.sidebar.multiselect("Portfolio", portfolios, default=[])

campaign_search = st.sidebar.text_input("Campaign name contains…")

st.sidebar.markdown("---")
st.sidebar.subheader("Recommendation thresholds")
min_clicks = st.sidebar.number_input("Min clicks per placement to be eligible", min_value=0, value=10, step=5)
min_spend = st.sidebar.number_input("Min spend ($) per placement to be eligible", min_value=0.0, value=5.0, step=1.0)
rank_metric = st.sidebar.radio("Rank best placement by", ["ACOS (lower is better)", "ROAS (higher is better)"])

# ----------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------

df = raw[(raw["Date"].dt.date >= start_date) & (raw["Date"].dt.date <= end_date)].copy()
if selected_portfolios:
    df = df[df["Portfolio name"].isin(selected_portfolios)]
if campaign_search:
    df = df[df["Campaign Name"].str.contains(campaign_search, case=False, na=False)]

if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.title("Amazon Sponsored Products — Placement Intelligence Tool")
st.caption(f"{df['Campaign Name'].nunique()} campaigns · {start_date} → {end_date} · {len(df):,} rows")

tab_overview, tab_compare, tab_deep_dive, tab_recs = st.tabs(
    ["Overview", "Placement Comparison", "Campaign Deep-Dive", "Recommendations"]
)

# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------

with tab_overview:
    totals = compute_metrics(pd.DataFrame([{
        "Impressions": df["Impressions"].sum(), "Clicks": df["Clicks"].sum(),
        "Spend": df["Spend"].sum(), "Sales": df["Sales"].sum(),
        "Orders": df["Orders"].sum(), "Units": df["Units"].sum(),
    }])).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", f"${totals['Spend']:,.0f}")
    c2.metric("Sales", f"${totals['Sales']:,.0f}")
    c3.metric("ACOS", f"{totals['ACOS']*100:,.1f}%" if pd.notna(totals['ACOS']) else "—")
    c4.metric("ROAS", f"{totals['ROAS']:.2f}" if pd.notna(totals['ROAS']) else "—")
    c5.metric("Orders", f"{totals['Orders']:,.0f}")

    st.markdown("#### Spend & Sales by Placement")
    by_place = aggregate(df, ["Placement"])
    fig = go.Figure()
    fig.add_bar(name="Spend", x=by_place["Placement"], y=by_place["Spend"])
    fig.add_bar(name="Sales", x=by_place["Placement"], y=by_place["Sales"])
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Daily Spend Trend by Placement")
    daily = aggregate(df, ["Date", "Placement"])
    fig2 = px.line(daily, x="Date", y="Spend", color="Placement")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------
# Placement Comparison
# ----------------------------------------------------------------------

with tab_compare:
    st.markdown("#### Aggregated Performance by Placement (all campaigns)")
    by_place = aggregate(df, ["Placement"]).sort_values("Spend", ascending=False)
    display = by_place.copy()
    display["CTR"] = (display["CTR"] * 100).round(2)
    display["CVR"] = (display["CVR"] * 100).round(2)
    display["ACOS"] = (display["ACOS"] * 100).round(1)
    display["ROAS"] = display["ROAS"].round(2)
    display["CPC"] = display["CPC"].round(2)
    display["Spend"] = display["Spend"].round(2)
    display["Sales"] = display["Sales"].round(2)
    st.dataframe(
        display.rename(columns={"CTR": "CTR %", "CVR": "CVR %", "ACOS": "ACOS %"}),
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(by_place.sort_values("ACOS"), x="Placement", y="ACOS", title="ACOS by Placement (lower = better)")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(by_place.sort_values("ROAS", ascending=False), x="Placement", y="ROAS", title="ROAS by Placement (higher = better)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Placement Mix by Portfolio")
    by_port_place = aggregate(df, ["Portfolio name", "Placement"])
    fig3 = px.bar(by_port_place, x="Portfolio name", y="Spend", color="Placement", title="Spend Distribution")
    fig3.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------
# Campaign Deep-Dive
# ----------------------------------------------------------------------

with tab_deep_dive:
    campaigns = sorted(df["Campaign Name"].unique())
    selected_campaign = st.selectbox("Select a campaign", campaigns)
    cdf = df[df["Campaign Name"] == selected_campaign]

    by_place = aggregate(cdf, ["Placement"])
    display = by_place.copy()
    display["ACOS"] = (display["ACOS"] * 100).round(1)
    display["ROAS"] = display["ROAS"].round(2)
    display["CTR"] = (display["CTR"] * 100).round(2)
    display["CVR"] = (display["CVR"] * 100).round(2)
    st.dataframe(
        display.rename(columns={"CTR": "CTR %", "CVR": "CVR %", "ACOS": "ACOS %"}),
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(by_place, x="Placement", y=["Spend", "Sales"], barmode="group",
                     title=f"Spend vs Sales — {selected_campaign}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(by_place, x="Placement", y="ACOS", title="ACOS by Placement")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Daily Trend")
    daily = aggregate(cdf, ["Date", "Placement"])
    metric_choice = st.radio("Metric", ["Spend", "Sales", "ACOS", "ROAS", "Clicks"], horizontal=True)
    fig = px.line(daily, x="Date", y=metric_choice, color="Placement")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------

with tab_recs:
    st.markdown("#### Best Placement per Campaign")
    st.caption(
        "A placement is 'eligible' once it clears the click/spend thresholds set in the sidebar. "
        "Among eligible placements, the best one is ranked by the metric you chose (ACOS or ROAS)."
    )

    recs = recommend_placements(df, min_clicks, min_spend, rank_metric)

    only_best = st.checkbox("Show only the recommended (best) row per campaign", value=True)
    show = recs[recs["Is Best"]] if only_best else recs

    display = show.copy()
    display["ACOS"] = (display["ACOS"] * 100).round(1)
    display["ROAS"] = display["ROAS"].round(2)
    display["CVR"] = (display["CVR"] * 100).round(2)
    display["CTR"] = (display["CTR"] * 100).round(2)
    display = display.sort_values("Total Campaign Spend", ascending=False)

    st.dataframe(
        display[[
            "Campaign Name", "Portfolio name", "Placement", "Spend", "Sales",
            "Clicks", "Orders", "ACOS", "ROAS", "CVR", "CTR", "Action",
        ]].rename(columns={"CTR": "CTR %", "CVR": "CVR %", "ACOS": "ACOS %"}),
        use_container_width=True, hide_index=True, height=500,
    )

    action_counts = recs[recs["Action"] != "Insufficient data"]["Action"].value_counts()
    if not action_counts.empty:
        fig = px.pie(names=action_counts.index, values=action_counts.values, title="Action Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Export")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        recs.to_excel(writer, sheet_name="All Campaigns x Placements", index=False)
        recs[recs["Is Best"]].to_excel(writer, sheet_name="Recommended Placement", index=False)
    st.download_button(
        "Download recommendations (.xlsx)",
        data=buf.getvalue(),
        file_name="placement_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

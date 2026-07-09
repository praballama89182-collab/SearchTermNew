"""
Amazon Sponsored Products — Placement Intelligence Tool (FBA-only)
---------------------------------------------------------------------
Analyzes Amazon Ads placement reports (Top of Search / Rest of Search /
Product Pages / Off Amazon) to identify the best-performing placement per
campaign and recommend quantified bid/budget reallocation.

Only portfolios whose name contains "FBA" are considered by default — this
keeps FBM, Vizari, Casafoyer, and other non-FBA portfolios from skewing the
placement-level aggregates.

Note: this report has no "Match Type" dimension (that's a Search Term report
field) — its equivalent axis is Placement, so any Match-Type-style toggle
here (pie/heatmap) is built around Placement instead.

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
# Visual theme — Google Material palette (Looker Studio look): white
# scorecards with colored accent bars, card-based sections, Roboto type,
# and one consistent colorway across every chart.
# ----------------------------------------------------------------------

PALETTE = {
    "blue": "#4285F4", "red": "#EA4335", "yellow": "#FBBC04",
    "green": "#34A853", "purple": "#A142F4", "teal": "#24C1E0",
}
COLORWAY = [PALETTE["blue"], PALETTE["green"], PALETTE["red"],
            PALETTE["yellow"], PALETTE["purple"], PALETTE["teal"]]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
.stApp { background-color: #F8F9FA; }
h1, h2, h3 { font-family: 'Roboto', sans-serif; font-weight: 500; color: #202124; }

.kpi-card {
    background: #FFFFFF; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(60,64,67,.15), 0 1px 2px rgba(60,64,67,.10);
    border-top: 4px solid var(--accent, #4285F4);
    height: 100%;
}
.kpi-label {
    font-size: 12px; color: #5F6368; font-weight: 500;
    text-transform: uppercase; letter-spacing: .05em;
}
.kpi-value {
    font-family: 'Roboto Mono', monospace; font-size: 26px; font-weight: 700;
    color: #202124; margin-top: 4px;
}

.section-header { display: flex; align-items: center; gap: 8px; margin: 4px 0 12px 0; }
.section-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.section-title { font-size: 16px; font-weight: 600; color: #202124; }

button[data-baseweb="tab"] { font-weight: 600; font-size: 15px; }
[data-baseweb="tab-highlight"] { background-color: #4285F4 !important; }
[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, color: str) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
    )


def section_header(text: str, color: str):
    st.markdown(
        f'<div class="section-header"><span class="section-dot" style="background:{color}"></span>'
        f'<span class="section-title">{text}</span></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 420):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Roboto, sans-serif", color="#202124", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DADCE0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F3F4", zerolinecolor="#DADCE0")
    return fig


def placement_color_map(df: pd.DataFrame) -> dict:
    placements = sorted(df["Placement"].dropna().unique())
    return {p: COLORWAY[i % len(COLORWAY)] for i, p in enumerate(placements)}


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
    df["Portfolio name"] = df["Portfolio name"].fillna("No Portfolio")
    for col in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived metrics (CTR, CVR, CPC, ACOS, ROAS) to an aggregated frame.
    These are always recomputed from summed Spend/Sales/Clicks/Impressions —
    never averaged from the report's own per-row ACOS/ROAS columns, since
    those go blank on zero-sales rows and can't be validly averaged across
    rows (a simple mean of ratios != the true aggregate ratio).
    """
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


def recommend_placements(df: pd.DataFrame, min_clicks: int, min_spend: float,
                          rank_metric: str, reallocation_pct: float) -> pd.DataFrame:
    """
    For each campaign, ranks eligible placements by the chosen metric and
    quantifies a reallocation: each non-best eligible placement is suggested
    to shed `reallocation_pct` of its own spend, and the best placement is
    suggested to absorb the sum of those shifts. This is a transparent,
    directional heuristic — Amazon placement bidding actually works via %
    bid modifiers, not direct dollar transfers, so treat the $ amounts as
    guidance on direction and rough size, not a literal instruction.
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
                    "Is Best": False, "Action": "Insufficient data", "Suggested Spend Change ($)": 0.0,
                })
            continue

        if rank_metric == "ACOS (lower is better)":
            eligible = eligible.sort_values("ACOS", ascending=True, na_position="last")
        else:
            eligible = eligible.sort_values("ROAS", ascending=False, na_position="last")

        best_placement = eligible.iloc[0]["Placement"]
        worse_eligible = eligible.iloc[1:]

        shift_out = {}
        total_shift_in = 0.0
        for _, r in worse_eligible.iterrows():
            amt = r["Spend"] * reallocation_pct / 100.0
            shift_out[r["Placement"]] = amt
            total_shift_in += amt

        for _, r in g.iterrows():
            if r["Placement"] not in eligible["Placement"].values:
                action, delta = "Insufficient data", 0.0
            elif r["Placement"] == best_placement:
                delta = total_shift_in
                action = (f"Increase spend by ~${delta:,.0f} — shift from lower-{rank_metric.split()[0]} "
                          f"placements in this campaign") if delta > 0 else "Increase bid / shift budget here"
            else:
                delta = -shift_out.get(r["Placement"], 0.0)
                zero_sales_note = " (currently $0 sales)" if r["Sales"] == 0 else ""
                action = (f"Decrease spend by ~${abs(delta):,.0f} ({reallocation_pct:.0f}% of spend) "
                          f"— shift to {best_placement}{zero_sales_note}")

            rows.append({
                "Campaign Name": camp, "Portfolio name": portfolio,
                "Placement": r["Placement"], "Spend": r["Spend"], "Sales": r["Sales"],
                "Clicks": r["Clicks"], "Orders": r["Orders"],
                "ACOS": r["ACOS"], "ROAS": r["ROAS"], "CVR": r["CVR"], "CTR": r["CTR"],
                "Total Campaign Spend": total_spend,
                "Is Best": r["Placement"] == best_placement, "Action": action,
                "Suggested Spend Change ($)": round(delta, 2),
            })

    out = pd.DataFrame(rows)
    out["Suggested New Spend"] = (out["Spend"] + out["Suggested Spend Change ($)"]).round(2)
    return out


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("📊 Placement Intelligence")
uploaded = st.sidebar.file_uploader("Upload Placement Report (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.title("Amazon Sponsored Products — Placement Intelligence Tool")
    st.info("👈 Upload a Sponsored Products Placement report (.xlsx) to begin. "
            "Expected columns include Date, Portfolio name, Campaign Name, Placement, "
            "Impressions, Clicks, Spend, 7 Day Total Sales, 7 Day Total Orders, 7 Day Total Units.")
    st.stop()

try:
    raw = load_report(uploaded)
except Exception as e:
    st.error(f"Couldn't read this file: {e}")
    st.stop()

# --- FBA-only scope filter (applied before anything else touches the data) ---
st.sidebar.markdown("---")
fba_only = st.sidebar.checkbox("Only FBA portfolios", value=True,
                                help="Keeps FBM / Vizari / Casafoyer / other non-FBA portfolios out of "
                                     "the placement aggregates. On by default.")
is_fba = raw["Portfolio name"].str.contains("FBA", case=False, na=False)
raw_scoped = raw[is_fba] if fba_only else raw

excluded_n = raw["Portfolio name"][~is_fba].nunique()
if fba_only:
    st.sidebar.caption(f"Scoped to {raw_scoped['Portfolio name'].nunique()} FBA portfolios "
                        f"({excluded_n} non-FBA portfolios excluded).")

if raw_scoped.empty:
    st.error("No rows match an 'FBA' portfolio name in this file. Uncheck 'Only FBA portfolios' to see all data.")
    st.stop()

min_date, max_date = raw_scoped["Date"].min().date(), raw_scoped["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

portfolios = sorted(raw_scoped["Portfolio name"].dropna().unique())
selected_portfolios = st.sidebar.multiselect("Portfolio", portfolios, default=[])

campaign_search = st.sidebar.text_input("Campaign name contains…")

st.sidebar.markdown("---")
st.sidebar.subheader("Recommendation thresholds")
min_clicks = st.sidebar.number_input("Min clicks per placement to be eligible", min_value=0, value=10, step=5)
min_spend = st.sidebar.number_input("Min spend ($) per placement to be eligible", min_value=0.0, value=5.0, step=1.0)
rank_metric = st.sidebar.radio("Rank best placement by", ["ACOS (lower is better)", "ROAS (higher is better)"])
reallocation_pct = st.sidebar.slider(
    "Reallocation % (shift from lower-performing placements)",
    min_value=5, max_value=50, value=20, step=5,
    help="For each campaign, this % of each underperforming eligible placement's spend is "
         "suggested to move toward the best-performing eligible placement."
)

# ----------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------

df = raw_scoped[(raw_scoped["Date"].dt.date >= start_date) & (raw_scoped["Date"].dt.date <= end_date)].copy()
if selected_portfolios:
    df = df[df["Portfolio name"].isin(selected_portfolios)]
if campaign_search:
    df = df[df["Campaign Name"].str.contains(campaign_search, case=False, na=False)]

if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

PLACEMENT_COLORS = placement_color_map(df)

st.title("Amazon Sponsored Products — Placement Intelligence Tool")
scope_label = "FBA portfolios only" if fba_only else "all portfolios"
st.caption(f"{df['Campaign Name'].nunique()} campaigns · {scope_label} · {start_date} → {end_date} · {len(df):,} rows")

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

    kpis = [
        ("Spend", f"${totals['Spend']:,.0f}", PALETTE["red"]),
        ("Sales", f"${totals['Sales']:,.0f}", PALETTE["green"]),
        ("ACOS", f"{totals['ACOS']*100:,.1f}%" if pd.notna(totals['ACOS']) else "—", PALETTE["yellow"]),
        ("ROAS", f"{totals['ROAS']:.2f}" if pd.notna(totals['ROAS']) else "—", PALETTE["purple"]),
        ("Orders", f"{totals['Orders']:,.0f}", PALETTE["blue"]),
    ]
    kpi_cols = st.columns(5)
    for col, (label, value, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        section_header("Spend & Sales by Placement", PALETTE["blue"])
        by_place = aggregate(df, ["Placement"])
        fig = go.Figure()
        fig.add_bar(name="Spend", x=by_place["Placement"], y=by_place["Spend"],
                    marker_color=PALETTE["red"])
        fig.add_bar(name="Sales", x=by_place["Placement"], y=by_place["Sales"],
                    marker_color=PALETTE["green"])
        fig.update_layout(barmode="group")
        style_fig(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Daily Spend Trend by Placement", PALETTE["teal"])
        st.caption("Each placement gets its own scale (small multiples) — Off Amazon's trend "
                   "would otherwise be flattened to a near-invisible line next to Rest of Search.")
        daily = aggregate(df, ["Date", "Placement"]).sort_values("Date")
        fig2 = px.line(
            daily, x="Date", y="Spend", facet_col="Placement", facet_col_wrap=2,
            color="Placement", color_discrete_map=PLACEMENT_COLORS, markers=True,
        )
        fig2.update_yaxes(matches=None, showticklabels=True)
        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=13, color="#202124")))
        fig2.update_traces(line=dict(width=2.5), marker=dict(size=5))
        fig2.update_layout(showlegend=False)
        style_fig(fig2, height=520)
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------
# Placement Comparison
# ----------------------------------------------------------------------

with tab_compare:
    st.caption("ACOS/ROAS/CTR/CVR here are computed from summed Spend/Sales/Clicks/Impressions per "
               "placement — not averaged from the report's row-level ACOS/ROAS columns, which go blank "
               "on zero-sales rows and can't be validly averaged.")

    with st.container(border=True):
        section_header("Aggregated Performance by Placement (current scope)", PALETTE["blue"])
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

    with st.container(border=True):
        section_header("ACOS vs ROAS by Placement", PALETTE["purple"])
        st.caption("Bars (ACOS, lower = better) and line (ROAS, higher = better) plotted together so "
                   "you can read both signals for a placement in one glance, sorted best-to-worst by ACOS.")
        cmp = by_place.sort_values("ACOS", na_position="last")
        fig = go.Figure()
        fig.add_bar(
            name="ACOS", x=cmp["Placement"], y=cmp["ACOS"], yaxis="y",
            marker_color=[PLACEMENT_COLORS.get(p, PALETTE["red"]) for p in cmp["Placement"]],
            text=[f"{v*100:.1f}%" if pd.notna(v) else "—" for v in cmp["ACOS"]], textposition="outside",
        )
        fig.add_trace(go.Scatter(
            name="ROAS", x=cmp["Placement"], y=cmp["ROAS"], yaxis="y2",
            mode="lines+markers+text", line=dict(color=PALETTE["purple"], width=3),
            marker=dict(size=11, color=PALETTE["purple"]),
            text=[f"{v:.2f}x" if pd.notna(v) else "—" for v in cmp["ROAS"]], textposition="top center",
        ))
        fig.update_layout(
            yaxis=dict(title="ACOS", tickformat=".0%"),
            yaxis2=dict(title="ROAS", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Placement Share", PALETTE["green"])
        st.caption("This report has no Match Type dimension — its equivalent breakdown axis is "
                   "Placement, so the toggle below is built around Placement instead.")
        share_metric = st.selectbox("Metric", ["Ad Sales share", "Spend share", "ACOS"], key="placement_share_metric")
        if share_metric == "ACOS":
            fig = px.bar(
                by_place.sort_values("ACOS", na_position="last"), x="Placement", y="ACOS",
                color="Placement", color_discrete_map=PLACEMENT_COLORS,
                text=by_place.sort_values("ACOS", na_position="last")["ACOS"].apply(
                    lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"),
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
        else:
            value_col = "Sales" if share_metric == "Ad Sales share" else "Spend"
            fig = px.pie(
                by_place, names="Placement", values=value_col, hole=0.45,
                color="Placement", color_discrete_map=PLACEMENT_COLORS,
            )
            fig.update_traces(textinfo="label+percent", textposition="outside")
        style_fig(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Placement Mix by Portfolio — Top 10 by Spend", PALETTE["yellow"])
        matrix_metric = st.selectbox("Metric", ["Spend", "Sales", "ACOS", "ROAS"], key="portfolio_matrix_metric")
        by_port_place = aggregate(df, ["Portfolio name", "Placement"])
        top_portfolios = (
            aggregate(df, ["Portfolio name"]).sort_values("Spend", ascending=False).head(10)["Portfolio name"]
        )
        pivot_src = by_port_place[by_port_place["Portfolio name"].isin(top_portfolios)]
        pivot = pivot_src.pivot(index="Portfolio name", columns="Placement", values=matrix_metric)
        pivot = pivot.reindex(top_portfolios)

        if matrix_metric == "ACOS":
            colorscale, reversescale = "RdYlGn", True
            fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
        elif matrix_metric == "ROAS":
            colorscale, reversescale = "RdYlGn", False
            fmt = lambda v: f"{v:.2f}x" if pd.notna(v) else ""
        elif matrix_metric == "Spend":
            colorscale, reversescale = "Reds", False
            fmt = lambda v: f"${v:,.0f}" if pd.notna(v) else ""
        else:
            colorscale, reversescale = "Greens", False
            fmt = lambda v: f"${v:,.0f}" if pd.notna(v) else ""

        text = pivot.map(fmt).values
        heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=colorscale, reversescale=reversescale,
            text=text, texttemplate="%{text}", textfont=dict(size=12),
            hoverongaps=False, showscale=True,
        ))
        heat.update_layout(yaxis=dict(autorange="reversed"))
        style_fig(heat, height=max(360, 42 * len(pivot)))
        st.plotly_chart(heat, use_container_width=True)

# ----------------------------------------------------------------------
# Campaign Deep-Dive
# ----------------------------------------------------------------------

with tab_deep_dive:
    campaigns = sorted(df["Campaign Name"].unique())
    selected_campaign = st.selectbox("Select a campaign", campaigns)
    cdf = df[df["Campaign Name"] == selected_campaign]

    with st.container(border=True):
        section_header(f"Placement Breakdown — {selected_campaign}", PALETTE["blue"])
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
        with st.container(border=True):
            section_header("Spend vs Sales", PALETTE["green"])
            fig = go.Figure()
            fig.add_bar(name="Spend", x=by_place["Placement"], y=by_place["Spend"], marker_color=PALETTE["red"])
            fig.add_bar(name="Sales", x=by_place["Placement"], y=by_place["Sales"], marker_color=PALETTE["green"])
            fig.update_layout(barmode="group")
            style_fig(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            section_header("ACOS by Placement", PALETTE["yellow"])
            fig = px.bar(
                by_place, x="Placement", y="ACOS", color="Placement",
                color_discrete_map=PLACEMENT_COLORS,
                text=by_place["ACOS"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"),
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            style_fig(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Daily Trend", PALETTE["purple"])
        daily = aggregate(cdf, ["Date", "Placement"]).sort_values("Date")
        metric_choice = st.radio("Metric", ["Spend", "Sales", "ACOS", "ROAS", "Clicks"], horizontal=True)
        fig = px.line(
            daily, x="Date", y=metric_choice, color="Placement",
            color_discrete_map=PLACEMENT_COLORS, markers=True,
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        if metric_choice == "ACOS":
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------

with tab_recs:
    section_header("Best Placement per Campaign — Quantified Reallocation", PALETTE["red"])
    st.caption(
        f"A placement is 'eligible' once it clears the click/spend thresholds set in the sidebar. "
        f"Among eligible placements, the best one is ranked by {rank_metric}. Each other eligible "
        f"placement is suggested to shift {reallocation_pct:.0f}% of its own spend toward the best one — "
        f"a transparent, directional heuristic. Amazon placement bidding actually works through % bid "
        f"modifiers, not direct dollar transfers, so treat these $ amounts as guidance on direction and "
        f"rough size, not a literal instruction."
    )

    recs = recommend_placements(df, min_clicks, min_spend, rank_metric, reallocation_pct)

    only_best = st.checkbox("Show only the recommended (best) row per campaign", value=False)
    show = recs[recs["Is Best"]] if only_best else recs

    display = show.copy()
    display["ACOS"] = (display["ACOS"] * 100).round(1)
    display["ROAS"] = display["ROAS"].round(2)
    display["CVR"] = (display["CVR"] * 100).round(2)
    display["CTR"] = (display["CTR"] * 100).round(2)
    display = display.sort_values("Total Campaign Spend", ascending=False)

    def highlight_change(val):
        if val > 0:
            return "background-color:#e6f4ea; color:#137333; font-weight:600;"
        elif val < 0:
            return "background-color:#fce8e6; color:#c5221f; font-weight:600;"
        return ""

    with st.container(border=True):
        cols_to_show = [
            "Campaign Name", "Portfolio name", "Placement", "Spend", "Sales",
            "Clicks", "Orders", "ACOS", "ROAS", "CVR", "CTR",
            "Suggested Spend Change ($)", "Suggested New Spend", "Action",
        ]
        styled = (
            display[cols_to_show]
            .rename(columns={"CTR": "CTR %", "CVR": "CVR %", "ACOS": "ACOS %"})
            .style.map(highlight_change, subset=["Suggested Spend Change ($)"])
            .format(precision=2)
        )
        st.dataframe(styled, use_container_width=True, height=500)

    action_counts = recs[recs["Action"] != "Insufficient data"]["Action"].apply(
        lambda a: "Increase spend" if a.startswith("Increase") else
                  ("Decrease spend" if a.startswith("Decrease") else a)
    ).value_counts()
    if not action_counts.empty:
        with st.container(border=True):
            section_header("Action Breakdown", PALETTE["teal"])
            fig = px.pie(names=action_counts.index, values=action_counts.values,
                        color_discrete_sequence=COLORWAY, hole=0.45)
            style_fig(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        section_header("Export", PALETTE["blue"])
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

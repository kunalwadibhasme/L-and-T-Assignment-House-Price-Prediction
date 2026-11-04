"""Gradio frontend for Ames House Price Prediction.
Run with: .\\.venv\\Scripts\\python app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from src.predictor import load_model_bundle, predict_from_inputs

bundle = load_model_bundle()
NEIGHBORHOODS = sorted(bundle["neighborhoods"])
YEAR_MIN = bundle["sale_year_min"]
YEAR_MAX = bundle["sale_year_max"]
YEAR_BUILT_MIN = bundle["year_built_min"]
MODEL_NAME = bundle.get("selected_model", "XGBoost")

# Validation MAE from training artifacts (~$15,896)
MODEL_MAE = 15896.0

# Ames Neighborhood full names and training dataset median prices
NEIGHBORHOOD_INFO = {
    "Blmngtn": ("Bloomington Heights", 191000),
    "Blueste": ("Bluestem", 137500),
    "BrDale": ("Briardale", 106000),
    "BrkSide": ("Brookside", 124300),
    "ClearCr": ("Clear Creek", 200250),
    "CollgCr": ("College Creek", 197200),
    "Crawfor": ("Crawford", 200624),
    "Edwards": ("Edwards", 121750),
    "Gilbert": ("Gilbert", 181000),
    "IDOTRR": ("Iowa DOT & Rail Road", 103000),
    "MeadowV": ("Meadow Village", 88000),
    "Mitchel": ("Mitchell", 153500),
    "NAmes": ("North Ames", 140000),
    "NoRidge": ("Northridge", 301500),
    "NPkVill": ("Northpark Villa", 146000),
    "NridgHt": ("Northridge Heights", 315000),
    "NWAmes": ("Northwest Ames", 182900),
    "OldTown": ("Old Town", 119000),
    "SWISU": ("South & West of ISU", 139500),
    "Sawyer": ("Sawyer", 135000),
    "SawyerW": ("Sawyer West", 179900),
    "Somerst": ("Somerset", 225500),
    "StoneBr": ("Stone Brook", 278000),
    "Timber": ("Timberland", 228475),
    "Veenker": ("Veenker", 218000),
}

# Dropdown choices: (Display Label, Value)
DROPDOWN_CHOICES = [
    (f"{NEIGHBORHOOD_INFO.get(code, (code, 0))[0]} ({code}) — Median: ${NEIGHBORHOOD_INFO.get(code, (code, 0))[1]:,}", code)
    for code in NEIGHBORHOODS
]

def calculate_percentile(price):
    """Estimate Ames price percentile using dataset breakpoints."""
    breakpoints = [
        (34900, 0),
        (129975, 25),
        (163000, 50),
        (214000, 75),
        (326000, 95),
        (755000, 100),
    ]
    if price <= breakpoints[0][0]:
        return 1.0
    if price >= breakpoints[-1][0]:
        return 99.0
    for i in range(len(breakpoints) - 1):
        p_low, pct_low = breakpoints[i]
        p_high, pct_high = breakpoints[i + 1]
        if p_low <= price <= p_high:
            fraction = (price - p_low) / (p_high - p_low)
            return pct_low + fraction * (pct_high - pct_low)
    return 50.0

def build_result_html(price, living_area, basement_area, bedrooms, total_rooms,
                      quality, condition, garage_cars, year_built, year_sold, neighborhood):
    lower_range = max(0, price - MODEL_MAE)
    upper_range = price + MODEL_MAE
    total_area = living_area + basement_area
    psqft = price / living_area if living_area > 0 else 0
    age = max(0, year_sold - year_built)

    # Market Tier
    if price < 130000:
        tier_label = "Starter / Affordable Tier"
        tier_color = "#38bdf8"
        tier_bg = "rgba(56, 189, 248, 0.12)"
        tier_border = "rgba(56, 189, 248, 0.3)"
    elif price < 214000:
        tier_label = "Core Mid-Market Tier"
        tier_color = "#34d399"
        tier_bg = "rgba(52, 211, 153, 0.12)"
        tier_border = "rgba(52, 211, 153, 0.3)"
    elif price < 320000:
        tier_label = "Upper-Scale Residential"
        tier_color = "#a78bfa"
        tier_bg = "rgba(167, 139, 250, 0.12)"
        tier_border = "rgba(167, 139, 250, 0.3)"
    else:
        tier_label = "Premium Luxury Estate"
        tier_color = "#fbbf24"
        tier_bg = "rgba(251, 191, 36, 0.12)"
        tier_border = "rgba(251, 191, 36, 0.3)"

    # Neighborhood Comparison
    neigh_name, neigh_median = NEIGHBORHOOD_INFO.get(neighborhood, (neighborhood, 163000))
    if neigh_median > 0:
        diff_pct = ((price - neigh_median) / neigh_median) * 100
        if diff_pct >= 0:
            comp_badge = f'<span style="color: #34d399; font-weight: 600;">+{diff_pct:.1f}%</span> above neighborhood median'
        else:
            comp_badge = f'<span style="color: #f59e0b; font-weight: 600;">{abs(diff_pct):.1f}%</span> below neighborhood median'
    else:
        comp_badge = "Baseline benchmark"

    # Percentile & gauge position
    percentile = calculate_percentile(price)
    pin_position = max(3.0, min(97.0, percentile))

    # Quality descriptor
    qual_descriptions = {
        1: "Very Poor", 2: "Poor", 3: "Fair", 4: "Below Average",
        5: "Average", 6: "Above Average", 7: "Good", 8: "Very Good",
        9: "Excellent", 10: "Very Excellent"
    }
    qual_label = qual_descriptions.get(int(quality), f"Level {quality}")

    return f"""
    <div class="result-dashboard">
        <!-- Hero Card -->
        <div class="hero-card">
            <div class="hero-top-row">
                <div class="hero-badge">
                    <span class="live-pulse"></span>
                    <span>ESTIMATED HISTORICAL VALUATION</span>
                </div>
                <div class="tier-pill" style="color: {tier_color}; background: {tier_bg}; border: 1px solid {tier_border};">
                    ★ {tier_label}
                </div>
            </div>

            <div class="hero-price-wrap">
                <span class="currency-tag">$</span>
                <span class="price-digits">{price:,.0f}</span>
                <span class="currency-code">USD</span>
            </div>

            <div class="hero-subgrid">
                <div class="subgrid-chip">
                    <span class="chip-label">Valuation Confidence Range</span>
                    <span class="chip-value">${lower_range:,.0f} – ${upper_range:,.0f}</span>
                    <span class="chip-sub">±${MODEL_MAE:,.0f} validation MAE</span>
                </div>
                <div class="subgrid-chip">
                    <span class="chip-label">Unit Valuation</span>
                    <span class="chip-value">${psqft:.1f} / sq ft</span>
                    <span class="chip-sub">{living_area:,} sq ft above-ground</span>
                </div>
            </div>
        </div>

        <!-- Market Distribution Gauge -->
        <div class="card-section">
            <div class="section-header-row">
                <span class="section-title">📊 Ames Market Position Benchmark</span>
                <span class="section-subtitle">Property sits at approx. <strong>{percentile:.0f}th percentile</strong> of Ames sales</span>
            </div>
            
            <div class="market-gauge-track">
                <div class="gauge-pin" style="left: {pin_position}%;">
                    <div class="gauge-pin-tooltip">${price:,.0f}</div>
                    <div class="gauge-pin-arrow"></div>
                </div>
            </div>
            <div class="gauge-milestones">
                <div class="milestone"><span>Min</span><strong>$35k</strong></div>
                <div class="milestone"><span>Q1</span><strong>$130k</strong></div>
                <div class="milestone"><span>Median</span><strong>$163k</strong></div>
                <div class="milestone"><span>Q3</span><strong>$214k</strong></div>
                <div class="milestone"><span>Top 5%</span><strong>$326k</strong></div>
                <div class="milestone"><span>Max</span><strong>$755k</strong></div>
            </div>
        </div>

        <!-- 4-Card Feature Grid -->
        <div class="metric-quad-grid">
            <div class="quad-card">
                <div class="quad-icon">📐</div>
                <div class="quad-meta">
                    <span class="quad-label">Total Usable Area</span>
                    <span class="quad-val">{total_area:,} sq ft</span>
                    <span class="quad-detail">{living_area:,} livable • {basement_area:,} bsmt</span>
                </div>
            </div>

            <div class="quad-card">
                <div class="quad-icon">📍</div>
                <div class="quad-meta">
                    <span class="quad-label">Neighborhood Context</span>
                    <span class="quad-val">{neigh_name}</span>
                    <span class="quad-detail">{comp_badge}</span>
                </div>
            </div>

            <div class="quad-card">
                <div class="quad-icon">⏳</div>
                <div class="quad-meta">
                    <span class="quad-label">House Age at Sale</span>
                    <span class="quad-val">{age} Years Old</span>
                    <span class="quad-detail">Built in {year_built} • Sold in {year_sold}</span>
                </div>
            </div>

            <div class="quad-card">
                <div class="quad-icon">⭐</div>
                <div class="quad-meta">
                    <span class="quad-label">Build Quality & Condition</span>
                    <span class="quad-val">Qual {quality}/10 ({qual_label})</span>
                    <span class="quad-detail">Condition: {condition}/10 • {garage_cars}-Car Garage</span>
                </div>
            </div>
        </div>

        <!-- Technical Model Specs -->
        <div class="model-spec-strip">
            <div class="spec-pills">
                <span class="spec-badge">🤖 <strong>{MODEL_NAME} Pipeline</strong></span>
                <span class="spec-badge">🎯 Validation R²: <strong>0.902</strong></span>
                <span class="spec-badge">📉 CV RMSE: <strong>±$28,458</strong></span>
                <span class="spec-badge">📚 Kaggle Ames Dataset: <strong>1,460 Homes</strong></span>
            </div>
            <div class="spec-disclaimer">
                💡 <em>Educational machine-learning estimate based on historical 2006–2010 Ames, Iowa sales records. Unspecified property traits inherit representative dataset baseline parameters.</em>
            </div>
        </div>
    </div>
    """

def estimate(living_area, basement_area, bedrooms, total_rooms, quality,
             condition, garage_cars, year_built, year_sold, neighborhood):
    try:
        price = predict_from_inputs(
            living_area=float(living_area),
            basement_area=float(basement_area),
            bedrooms=int(bedrooms),
            total_rooms=int(total_rooms),
            quality=int(quality),
            condition=int(condition),
            garage_cars=int(garage_cars),
            year_built=int(year_built),
            year_sold=int(year_sold),
            neighborhood=neighborhood,
            bundle=bundle,
        )
        return build_result_html(
            price=price,
            living_area=float(living_area),
            basement_area=float(basement_area),
            bedrooms=int(bedrooms),
            total_rooms=int(total_rooms),
            quality=int(quality),
            condition=int(condition),
            garage_cars=int(garage_cars),
            year_built=int(year_built),
            year_sold=int(year_sold),
            neighborhood=neighborhood,
        )
    except Exception as exc:
        return f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 20px; color: #f87171;">
            <h4 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600;">⚠️ Prediction Input Notice</h4>
            <p style="margin: 0; font-size: 14px;">{html_escape(str(exc))}</p>
        </div>
        """

def html_escape(text):
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )

# Presets helper: returns all 10 input values plus estimated result
PRESETS = {
    "starter": (1450, 750, 3, 6, 6, 5, 2, 1995, 2009, "NAmes"),
    "luxury": (2850, 1600, 4, 9, 9, 5, 3, 2006, 2010, "NridgHt"),
    "suburban": (1850, 950, 3, 7, 7, 6, 2, 2003, 2008, "CollgCr"),
    "fixer": (1100, 550, 2, 5, 4, 4, 1, 1940, 2007, "OldTown"),
}

def load_preset(key):
    vals = PRESETS[key]
    res_html = estimate(*vals)
    return (*vals, res_html)

CUSTOM_CSS = """
/* Container & Global Enhancements */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Modern Header Card */
.header-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 30px 32px;
    margin-bottom: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.header-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}

.header-badge-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.top-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.header-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 15px;
    color: #94a3b8;
    line-height: 1.5;
    margin: 0 0 16px 0;
    max-width: 880px;
}

.header-stat-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: #cbd5e1;
    background: rgba(255, 255, 255, 0.05);
    padding: 4px 12px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Presets Bar */
.presets-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 22px;
}

.presets-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Card Styling for Form Blocks */
.input-section-card {
    background: rgba(30, 41, 59, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
}

.section-label {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
    letter-spacing: -0.2px !important;
}

/* Primary Action Button */
.estimate-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

.estimate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px -4px rgba(99, 102, 241, 0.65) !important;
}

/* Result Dashboard */
.result-dashboard {
    display: flex;
    flex-direction: column;
    gap: 16px;
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Hero Valuation Card */
.hero-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(24, 32, 56, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 20px;
    padding: 26px 28px;
    box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 12px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94a3b8;
    text-transform: uppercase;
}

.live-pulse {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    display: inline-block;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.tier-pill {
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 12px;
    letter-spacing: 0.3px;
}

.hero-price-wrap {
    display: flex;
    align-items: baseline;
    gap: 4px;
    margin: 8px 0 16px 0;
}

.currency-tag {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
}

.price-digits {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffffff 40%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.currency-code {
    font-size: 16px;
    font-weight: 700;
    color: #64748b;
    margin-left: 6px;
}

.hero-subgrid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    padding-top: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.subgrid-chip {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.chip-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.chip-value {
    font-size: 16px;
    font-weight: 700;
    color: #f8fafc;
}

.chip-sub {
    font-size: 11.5px;
    color: #64748b;
}

/* Card Section (Market Benchmark) */
.card-section {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 20px 22px;
}

.section-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}

.section-title {
    font-size: 13.5px;
    font-weight: 700;
    color: #f1f5f9;
}

.section-subtitle {
    font-size: 12.5px;
    color: #94a3b8;
}

/* Market Benchmark Gauge */
.market-gauge-track {
    height: 12px;
    border-radius: 8px;
    background: linear-gradient(90deg, #38bdf8 0%, #34d399 35%, #a78bfa 70%, #f59e0b 100%);
    position: relative;
    margin: 32px 10px 10px 10px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

.gauge-pin {
    position: absolute;
    top: -24px;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: left 0.4s ease;
}

.gauge-pin-tooltip {
    background: #ffffff;
    color: #0f172a;
    font-size: 11px;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 6px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35);
    white-space: nowrap;
}

.gauge-pin-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #ffffff;
    margin-top: -1px;
}

.gauge-milestones {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    padding: 0 4px;
}

.milestone {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}

.milestone span {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
}

.milestone strong {
    font-size: 11.5px;
    color: #cbd5e1;
}

/* Quad Feature Grid */
.metric-quad-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
}

.quad-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.quad-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.3);
}

.quad-icon {
    font-size: 24px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 8px;
    line-height: 1;
}

.quad-meta {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.quad-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.quad-val {
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
}

.quad-detail {
    font-size: 11.5px;
    color: #94a3b8;
    margin-top: 1px;
}

/* Model Spec Strip */
.model-spec-strip {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.spec-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.spec-badge {
    background: rgba(255, 255, 255, 0.05);
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 11px;
    color: #94a3b8;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.spec-badge strong {
    color: #e2e8f0;
}

.spec-disclaimer {
    font-size: 11px;
    color: #64748b;
    line-height: 1.4;
}
"""

# Initial computation so the screen starts with an attractive preview right away
INITIAL_ESTIMATE_HTML = estimate(
    living_area=1550,
    basement_area=850,
    bedrooms=3,
    total_rooms=7,
    quality=6,
    condition=5,
    garage_cars=2,
    year_built=2000,
    year_sold=2010,
    neighborhood="NAmes",
)

with gr.Blocks(title="Ames Housing Market Valuation Engine") as demo:
    # Header Card with embedded style sheet for robust cross-environment rendering
    gr.HTML(f"""
    <style>
    {CUSTOM_CSS}
    </style>
    <div class="header-card">
        <div class="header-badge-row">
            <span class="top-pill">🚀 Machine Learning Model</span>
            <span class="meta-chip">Ames, Iowa Historical Real Estate</span>
            <span class="meta-chip">1,460 Training Records</span>
        </div>
        <h1 class="header-title">🏡 Ames Housing Valuation Engine</h1>
        <p class="header-subtitle">
            Estimate authentic historical home values based on living area, basement capacity, construction quality,
            and neighborhood metrics using our optimized <strong>{MODEL_NAME}</strong> regression pipeline.
        </p>
        <div class="header-stat-badges">
            <span class="meta-chip">🎯 <strong>R² Score: 0.902</strong> (90.2% variance explained)</span>
            <span class="meta-chip">📉 <strong>CV RMSE: ±$28,458</strong></span>
            <span class="meta-chip">📊 <strong>Validation MAE: ±$15,896</strong></span>
            <span class="meta-chip">📅 <strong>Sales Epoch: {YEAR_MIN}–{YEAR_MAX}</strong></span>
        </div>
    </div>
    """)

    # Quick Preset Scenarios
    with gr.Group():
        gr.HTML('<div class="presets-title">⚡ Quick Scenarios (1-Click Sample Properties)</div>')
        with gr.Row():
            btn_starter = gr.Button("🏡 Suburban Starter ($167k)", size="sm", variant="secondary")
            btn_suburban = gr.Button("🎓 College Town Home ($174k)", size="sm", variant="secondary")
            btn_luxury = gr.Button("💎 Luxury Executive Villa ($317k)", size="sm", variant="secondary")
            btn_fixer = gr.Button("🔨 Historic Fixer-Upper ($112k)", size="sm", variant="secondary")

    # Main Workspace: 2-Column Split (Form on Left, Live Output Dashboard on Right)
    with gr.Row(equal_height=False):
        # Left Column: Inputs Form
        with gr.Column(scale=5):
            # Section 1: Dimensions
            with gr.Group(elem_classes=["input-section-card"]):
                gr.HTML('<div class="section-label">📐 Living Space & Room Layout</div>')
                living_area = gr.Number(
                    value=1550,
                    label="Above-Ground Living Area (sq ft)",
                    minimum=200,
                    maximum=6000,
                    info="Finished living space above grade level"
                )
                basement_area = gr.Number(
                    value=850,
                    label="Total Basement Area (sq ft)",
                    minimum=0,
                    maximum=4000,
                    info="0 indicates slab / no basement"
                )
                with gr.Row():
                    bedrooms = gr.Number(value=3, label="Bedrooms Above Ground", precision=0, minimum=0, maximum=10)
                    total_rooms = gr.Number(value=7, label="Total Rooms (Abv Ground)", precision=0, minimum=1, maximum=18)

            # Section 2: Construction Quality & Age
            with gr.Group(elem_classes=["input-section-card"]):
                gr.HTML('<div class="section-label">⭐ Quality, Condition & Timing</div>')
                quality = gr.Slider(
                    1, 10, value=6, step=1,
                    label="Overall Material & Finish Quality (1 Very Poor – 10 Excellent)",
                    info="Ames standard scale: 5=Average, 7=Good, 9=Luxury"
                )
                condition = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Overall Physical Condition (1 Very Poor – 10 Excellent)",
                    info="General maintenance & upkeep of property"
                )
                with gr.Row():
                    garage_cars = gr.Number(value=2, label="Garage Capacity (cars)", precision=0, minimum=0, maximum=6)
                    year_built = gr.Number(value=2000, label="Year Built", precision=0, minimum=YEAR_BUILT_MIN, maximum=YEAR_MAX)
                    year_sold = gr.Number(value=YEAR_MAX, label="Sale Year", precision=0, minimum=YEAR_MIN, maximum=YEAR_MAX)

            # Section 3: Location
            with gr.Group(elem_classes=["input-section-card"]):
                gr.HTML('<div class="section-label">📍 Neighborhood Location</div>')
                neighborhood = gr.Dropdown(
                    choices=DROPDOWN_CHOICES,
                    value="NAmes",
                    label="Select Ames Neighborhood",
                    info="Historical median sales price shown in label"
                )

            predict_button = gr.Button(
                "🔮 Calculate Property Valuation",
                variant="primary",
                elem_classes=["estimate-btn"],
                size="lg"
            )

        # Right Column: Visual Valuation Dashboard
        with gr.Column(scale=7):
            output_dashboard = gr.HTML(value=INITIAL_ESTIMATE_HTML)

    # Event Handlers
    input_list = [
        living_area, basement_area, bedrooms, total_rooms, quality,
        condition, garage_cars, year_built, year_sold, neighborhood
    ]

    predict_button.click(
        fn=estimate,
        inputs=input_list,
        outputs=output_dashboard,
    )

    # Wire Preset Buttons
    btn_starter.click(
        fn=lambda: load_preset("starter"),
        inputs=None,
        outputs=[*input_list, output_dashboard],
    )
    btn_suburban.click(
        fn=lambda: load_preset("suburban"),
        inputs=None,
        outputs=[*input_list, output_dashboard],
    )
    btn_luxury.click(
        fn=lambda: load_preset("luxury"),
        inputs=None,
        outputs=[*input_list, output_dashboard],
    )
    btn_fixer.click(
        fn=lambda: load_preset("fixer"),
        inputs=None,
        outputs=[*input_list, output_dashboard],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CUSTOM_CSS)

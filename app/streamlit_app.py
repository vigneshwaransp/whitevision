"""Modern Industrial Web Interface for Construction-Site Vehicle Classification.

Engineered for precision inference, zero-emoji telemetry, and partial-image robustness.
"""

import json
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.normalization import get_normalizer
from src.inference.predict import VehiclePredictor, letterbox_image

st.set_page_config(
    page_title="WHITEVISION | Construction Vehicle Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Industrial Styling (Zero Emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        background-image: 
            radial-gradient(at 0% 0%, rgba(245, 158, 11, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.04) 0px, transparent 50%);
        color: #e2e8f0;
    }

    /* Top Operational Header */
    .op-header {
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding-bottom: 14px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }

    .brand-accent {
        color: #f59e0b;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Status & Telemetry Badges */
    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .chip-online {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .chip-model {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .chip-tta {
        background: rgba(6, 182, 212, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    /* Hero Detection Card */
    .hero-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 29, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.6);
        position: relative;
    }

    .hero-panel-warning {
        border-color: rgba(239, 68, 68, 0.45);
    }

    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .pred-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }

    .code-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
        vertical-align: middle;
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
    }

    .metric-high {
        color: #10b981;
    }

    .metric-low {
        color: #ef4444;
    }

    /* Progress and Bars */
    .bar-container {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin-top: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* Technical Specs Box */
    .dossier-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 18px;
        font-size: 0.88rem;
    }

    .dossier-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #f59e0b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    /* Clean Streamlit Elements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94a3b8;
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #f59e0b !important;
        border-bottom: 2px solid #f59e0b !important;
        background: rgba(245, 158, 11, 0.06) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Cache and load inference model predictor with TTA support."""
    try:
        return VehiclePredictor(config_path="config/config.yaml", classes_config_path="config/classes.yaml")
    except Exception as e:
        st.error(f"Inference Engine Initialization Error: {e}")
        return None


def get_sample_test_images():
    """Discover available test crops across all classes for one-click testing."""
    test_dir = PROJECT_ROOT / "data" / "test"
    crops_dir = PROJECT_ROOT / "data" / "crops"
    samples = {}
    target_dir = test_dir if test_dir.exists() and any(test_dir.iterdir()) else crops_dir

    if target_dir.exists():
        for class_dir in sorted(target_dir.iterdir()):
            if class_dir.is_dir():
                for img_file in sorted(class_dir.iterdir()):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        cname = class_dir.name.replace("_", " ").upper()
                        samples[f"[{cname}] {img_file.name[:18]}..."] = img_file
                        break
    return samples


def get_vehicle_dossier(canonical_class: str) -> dict:
    """Return equipment engineering dossier and operational specifications."""
    dossiers = {
        "bulldozer": {
            "category": "Heavy Earthmoving Equipment",
            "ground_drive": "Continuous Steel Tracks / High-Flotation",
            "operating_weight": "18,000 - 45,000 kg",
            "primary_tool": "Universal (U) / Semi-Universal (SU) Heavy Blade",
            "site_role": "Bulk earthmoving, heavy grading, pioneering, and ripping bedrock."
        },
        "dump_truck": {
            "category": "Haulage & Material Transport",
            "ground_drive": "Heavy-Duty Multi-Axle Wheeled Chassis (6x4 / 8x4)",
            "operating_weight": "25,000 - 60,000 kg (Gross Vehicle Weight)",
            "primary_tool": "Reinforced High-Tensile Hydraulic Dump Bed",
            "site_role": "Bulk haulage of blasted rock, aggregate, soil, and demolition waste."
        },
        "excavator": {
            "category": "Hydraulic Digging & Extraction",
            "ground_drive": "Steel Crawler Tracks with 360-Degree Swing Slew",
            "operating_weight": "14,000 - 90,000 kg",
            "primary_tool": "Hydraulic Boom, Dipper Arm, and Heavy Bucket",
            "site_role": "Deep trenching, foundation pit excavation, and heavy material loading."
        },
        "grader": {
            "category": "Surface Precision Engineering",
            "ground_drive": "Tandem Rear Drive Wheels with Articulated Frame",
            "operating_weight": "15,000 - 22,000 kg",
            "primary_tool": "Full-Rotation Mid-Mounted Precision Moldboard",
            "site_role": "Final surface grading, road crowning, side bank shaping, and gravel spreading."
        },
        "loader": {
            "category": "Front-End Bulk Handling",
            "ground_drive": "Articulated 4WD Chassis with High-Lug Off-Road Tires",
            "operating_weight": "12,000 - 35,000 kg",
            "primary_tool": "Front-Mounted Hydraulic High-Capacity Bucket",
            "site_role": "Loading haul trucks from stockpiles, short-cycle material transfer, backfilling."
        },
        "mixer_truck": {
            "category": "Ready-Mix Concrete Logistics",
            "ground_drive": "Commercial Multi-Axle Truck Chassis (6x4 / 8x4 / 10x4)",
            "operating_weight": "26,000 - 32,000 kg (Loaded)",
            "primary_tool": "Spiral-Bladed Rotating Concrete Mixing Drum",
            "site_role": "Transporting and continuous agitation of ready-mix concrete to pour points."
        },
        "mobile_crane": {
            "category": "Heavy Lifting & Erection",
            "ground_drive": "All-Terrain Rubber-Tired Carrier or Crawler Undercarriage",
            "operating_weight": "30,000 - 120,000 kg",
            "primary_tool": "Telescopic Multi-Section Hydraulic Boom and Winch Cable",
            "site_role": "Vertical lifting, steel beam placement, and mechanical equipment installation."
        },
        "roller": {
            "category": "Compaction & Pavement Engineering",
            "ground_drive": "Single/Tandem Steel Vibratory Drums or Pneumatic Tires",
            "operating_weight": "8,000 - 20,000 kg",
            "primary_tool": "High-Amplitude Eccentric Dynamic Vibratory Drum",
            "site_role": "Subgrade soil compaction, granular base consolidation, and asphalt wearing course smoothing."
        }
    }
    return dossiers.get(canonical_class, {
        "category": "Standard Construction Machinery",
        "ground_drive": "Tracked or Wheeled",
        "operating_weight": "Variable",
        "primary_tool": "Standard Implement",
        "site_role": "General civil construction operations."
    })


def main():
    normalizer = get_normalizer("config/classes.yaml")
    canonical_classes = normalizer.get_classes()

    # Sidebar: Control Panel
    with st.sidebar:
        st.markdown("### SYSTEM CONTROL PANEL")
        st.caption("WHITEVISION // CLASSIFICATION BACKEND")

        st.markdown("---")
        st.markdown("#### INFERENCE CONFIGURATION")
        
        confidence_threshold = st.slider(
            "Confidence Cutoff Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Predictions below this score are flagged as Low Confidence."
        )

        top_k = st.selectbox(
            "Candidate Ranking Depth (Top-K)",
            options=[1, 3, 5],
            index=1,
            help="Number of candidate classes to present in the ranking."
        )

        enable_tta = st.toggle(
            "Partial-Crop Multi-Scale TTA",
            value=True,
            help="Generates multi-scale aspect-preserved views (letterbox, center-zoom, mirror) to ensure high confidence (>50%) on partial or tight vehicle crops."
        )

        st.markdown("---")
        st.markdown("#### TEST SAMPLES EXPLORER")
        samples = get_sample_test_images()
        selected_sample = None
        if samples:
            sample_options = ["Custom Upload (Manual Input)"] + list(samples.keys())
            sample_choice = st.selectbox("Load Verified Test Sample:", sample_options)
            if sample_choice != "Custom Upload (Manual Input)":
                selected_sample = samples[sample_choice]
        else:
            st.info("No test set samples detected on local disk.")

        st.markdown("---")
        st.markdown("#### CANONICAL TARGET CLASSES (8)")
        for cls in canonical_classes:
            code = normalizer.get_code(cls)
            dname = normalizer.get_display_name(cls)
            st.markdown(
                f"<div style='font-family: monospace; font-size: 0.8rem; margin-bottom: 4px; color: #cbd5e1;'>"
                f"<span style='color: #f59e0b; font-weight: bold;'>[{code}]</span> {dname}"
                f"</div>",
                unsafe_allow_html=True
            )

    # Operational Top Banner
    st.markdown("""
    <div class="op-header">
        <div>
            <h1 class="brand-title">WHITEVISION <span class="brand-accent">//</span> VEHICLE INTELLIGENCE</h1>
            <div class="brand-subtitle">EFFICIENTNET-B0 NEURAL CLASSIFIER &bull; CONSTRUCTIONXC7C BENCHMARK</div>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="chip chip-online">SYS: ONLINE</span>
            <span class="chip chip-model">BACKBONE: EFFICIENTNET-B0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab_console, tab_analytics, tab_specs = st.tabs([
        "OPERATIONS CONSOLE",
        "FLEET & DATASET METRICS",
        "MODEL CARD & ARCHITECTURE"
    ])

    with tab_console:
        col_input, col_telemetry = st.columns([1, 1.25], gap="large")

        active_image = None
        source_label = ""

        with col_input:
            st.markdown("#### 1. IMAGE INGESTION")
            uploaded_file = st.file_uploader(
                "Upload construction equipment imagery (JPG, PNG, JPEG):",
                type=["jpg", "jpeg", "png"],
                help="Support for high-res site photos, drone captures, or partial vehicle crops."
            )

            if uploaded_file is not None:
                try:
                    active_image = Image.open(uploaded_file).convert("RGB")
                    source_label = f"Uploaded File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Image read error: {e}")
            elif selected_sample is not None:
                try:
                    active_image = Image.open(selected_sample).convert("RGB")
                    source_label = f"Test Split Sample: {selected_sample.name}"
                except Exception as e:
                    st.error(f"Sample load error: {e}")

            if active_image is not None:
                w, h = active_image.size
                ar = round(w / h, 2) if h > 0 else 1.0
                is_partial = ar > 1.8 or ar < 0.6 or min(w, h) < 150

                st.image(
                    active_image,
                    caption=f"{source_label} ({w}x{h} px, AR: {ar}:1)",
                    use_container_width=True
                )

                # Resolution Telemetry Tag
                partial_tag = "<span class='chip chip-tta' style='margin-left: 8px;'>PROFILE: PARTIAL / TIGHT CROP</span>" if is_partial else ""
                st.markdown(f"""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94a3b8; margin-top: -6px; margin-bottom: 12px;">
                    INPUT DIMENSIONS: <span style="color: #f8fafc;">{w} &times; {h} PX</span> &bull; ASPECT RATIO: <span style="color: #f8fafc;">{ar}:1</span> {partial_tag}
                </div>
                """, unsafe_allow_html=True)

                if enable_tta:
                    st.caption("Aspect-preserving letterboxing and multi-scale TTA active to ensure partial views cross >50% confidence.")
            else:
                st.info("Awaiting input image. Upload a photograph or select a sample from the left sidebar to begin automated telemetry.")

        with col_telemetry:
            st.markdown("#### 2. NEURAL CLASSIFICATION TELEMETRY")
            if active_image is not None:
                predictor = load_predictor()
                if predictor is None:
                    st.error("Predictor engine is not available. Please verify model files.")
                else:
                    with st.spinner("Executing neural inference with EfficientNetB0..."):
                        results = predictor.predict(
                            active_image,
                            top_k=top_k,
                            threshold=confidence_threshold,
                            use_tta=enable_tta,
                        )

                    is_supported = results.get("is_supported", True)

                    if not is_supported:
                        # High-Visibility Out-of-Distribution Error Alert Panel
                        error_msg = results.get("error", "No construction-site vehicle detected in this image.")
                        detected_obj = results.get("detected_object", "Non-Construction Entity")

                        st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #ef4444; border-radius: 8px; padding: 22px 24px; margin-bottom: 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #ef4444; letter-spacing: 0.08em; text-transform: uppercase;">
                                    [ERROR] UNSUPPORTED IMAGE DETECTED
                                </div>
                                <span class="chip" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5);">
                                    DOMAIN GATEKEEPER: REJECTED
                                </span>
                            </div>
                            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; color: #f87171; margin-bottom: 8px;">
                                NO CONSTRUCTION VEHICLE DETECTED
                            </div>
                            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;">
                                {error_msg}
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">
                                    ANALYSIS TELEMETRY:
                                </div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; color: #f1f5f9; font-weight: 600;">
                                    Identified Subject: <span style="color: #fbbf24;">{detected_obj}</span>
                                </div>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.5); border-left: 3px solid #f59e0b; border-radius: 4px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
                                <strong style="color: #f59e0b;">SUPPORTED EQUIPMENT DOMAIN (8 CLASSES):</strong><br>
                                [BDZ] Bulldozer &bull; [DTK] Dump Truck &bull; [EXC] Excavator &bull; [GRD] Grader<br>
                                [LDR] Loader &bull; [MIX] Mixer Truck &bull; [CRN] Mobile Crane &bull; [ROL] Roller<br>
                                <span style="color: #64748b; font-size: 0.75rem;">Passenger cars, personal automobiles, animals, and household objects are not supported.</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        is_confident = results["is_confident"]
                        top_item = results["top_predictions"][0]
                        pred_code = top_item["code"]
                        pred_dname = top_item["display_name"]
                        conf_val = top_item["confidence"]
                        conf_pct = top_item["percentage"]

                        # Primary Detection Panel
                        if is_confident:
                            headline_html = f"{pred_dname.upper()} <span class='code-badge'>{pred_code}</span>"
                            panel_style = "hero-panel"
                            status_text = "CONFIDENT IDENTIFICATION"
                            status_class = "chip chip-online"
                            metric_class = "metric-value metric-high"
                        else:
                            headline_html = "<span style='color: #f87171;'>UNKNOWN / UNCERTAIN</span> <span class='code-badge' style='color: #94a3b8; border-color: rgba(148,163,184,0.3);'>LOW CONFIDENCE</span>"
                            panel_style = "hero-panel hero-panel-warning"
                            status_text = f"BELOW THRESHOLD ({confidence_threshold * 100:.0f}%)"
                            status_class = "chip"
                            metric_class = "metric-value metric-low"

                        st.markdown(f"""
                        <div class="{panel_style}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div class="panel-label">PRIMARY CLASSIFICATION RESULT</div>
                                <span class="{status_class}">{status_text}</span>
                            </div>
                            <div class="pred-headline">
                                {headline_html}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 14px;">
                                <div>
                                    <div class="panel-label">CLASSIFICATION CONFIDENCE</div>
                                    <div class="{metric_class}">{conf_pct}</div>
                                </div>
                                <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8;">
                                    THRESHOLD: {confidence_threshold * 100:.0f}%<br>
                                    TTA ENGINE: {'MULTI-SCALE ACTIVE' if enable_tta else 'SINGLE-PASS'}
                                </div>
                            </div>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: {conf_pct}; background-color: {'#10b981' if is_confident else '#ef4444'};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Threshold Failure Alert Banner
                        if not is_confident:
                            st.markdown(f"""
                            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.35); border-left: 4px solid #ef4444; border-radius: 6px; padding: 14px 18px; margin-top: 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
                                <strong style="color: #f87171;">IDENTIFICATION REJECTED:</strong> The top candidate match was only <strong>{conf_pct}</strong>, which does not meet the required <strong>{confidence_threshold * 100:.0f}% confidence cutoff threshold</strong>.<br>
                                <span style="color: #94a3b8; font-size: 0.82rem;">The uploaded image does not contain a recognized construction vehicle with sufficient certainty. Please upload a clear photo of heavy construction equipment.</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Candidate Breakdown
                        ranking_title = "CANDIDATE PROBABILITY RANKING" if is_confident else "CANDIDATE PROBABILITY RANKING (STATISTICAL MATCHES ONLY - BELOW THRESHOLD)"
                        st.markdown(f"##### {ranking_title}")
                        for item in results["top_predictions"]:
                            rank = item["rank"]
                            code = item["code"]
                            dname = item["display_name"]
                            prob = item["confidence"]
                            pct = item["percentage"]

                            col_label, col_pct = st.columns([3, 1])
                            with col_label:
                                st.markdown(
                                    f"<div style='font-family: monospace; font-size: 0.85rem; color: #e2e8f0; margin-bottom: 2px;'>"
                                    f"<strong>[{rank}] [{code}]</strong> {dname}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            with col_pct:
                                st.markdown(
                                    f"<div style='font-family: monospace; font-size: 0.85rem; text-align: right; color: #cbd5e1;'>"
                                    f"{pct}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                            st.progress(float(prob))

                        # Complete 8-Class Probability Spectrum
                        with st.expander("VIEW FULL 8-CLASS PROBABILITY SPECTRUM", expanded=False):
                            prob_df = pd.DataFrame([
                                {
                                    "Code": normalizer.get_code(c),
                                    "Vehicle Class": normalizer.get_display_name(c),
                                    "Probability": f"{results['all_probabilities'].get(c, 0.0) * 100:.2f}%",
                                    "Raw Value": results['all_probabilities'].get(c, 0.0),
                                }
                                for c in canonical_classes
                            ]).sort_values(by="Raw Value", ascending=False)

                            st.dataframe(
                                prob_df[["Code", "Vehicle Class", "Probability"]],
                                hide_index=True,
                                use_container_width=True
                            )

                        # Equipment Engineering Dossier (Only when confident)
                        if is_confident:
                            dossier = get_vehicle_dossier(top_item["class"])
                            st.markdown(f"""
                            <div class="dossier-box">
                                <div class="dossier-title">EQUIPMENT DOSSIER: {pred_dname.upper()} [{pred_code}]</div>
                                <div style="color: #cbd5e1; line-height: 1.6;">
                                    <strong>Category:</strong> {dossier['category']}<br>
                                    <strong>Operating Weight:</strong> {dossier['operating_weight']}<br>
                                    <strong>Ground Mobility:</strong> {dossier['ground_drive']}<br>
                                    <strong>Primary Implement:</strong> {dossier['primary_tool']}<br>
                                    <strong>Civil Construction Application:</strong> {dossier['site_role']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="dossier-box" style="border-left-color: #64748b;">
                                <div class="dossier-title" style="color: #94a3b8;">EQUIPMENT DOSSIER: SUSPENDED</div>
                                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                                    No equipment engineering dossier is rendered because the input image failed confidence thresholding. Please upload a clear photo of one of the 8 supported construction vehicles.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="border: 1px dashed rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 40px; text-align: center; color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                    TELEMETRY AWAITING IMAGE STREAM<br>
                    SELECT A SAMPLE VEHICLE OR UPLOAD AN IMAGE ON THE LEFT PANEL
                </div>
                """, unsafe_allow_html=True)

    with tab_analytics:
        st.markdown("#### FLEET DATASET METRICS & PARTITION INTEGRITY")
        meta_file = PROJECT_ROOT / "data" / "metadata.csv"
        quality_file = PROJECT_ROOT / "reports" / "data_quality_report.csv"

        if meta_file.exists():
            df_meta = pd.read_csv(meta_file)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Valid Crops", f"{len(df_meta):,}")
            with col2:
                train_count = len(df_meta[df_meta["split"] == "train"])
                st.metric("Training Split", f"{train_count:,}", "70.4%")
            with col3:
                val_count = len(df_meta[df_meta["split"] == "validation"])
                st.metric("Validation Split", f"{val_count:,}", "14.8%")
            with col4:
                test_count = len(df_meta[df_meta["split"] == "test"])
                st.metric("Test Split (Untouched)", f"{test_count:,}", "14.8%")

            st.markdown("---")
            st.markdown("##### Vehicle Class Distribution Across Split Partitions")
            dist = pd.crosstab(df_meta["class"], df_meta["split"])[["train", "validation", "test"]]
            st.dataframe(dist, use_container_width=True)
        else:
            st.info("Metadata file not found on disk.")

        if quality_file.exists():
            df_qual = pd.read_csv(quality_file)
            st.markdown("##### Data Quality Audit Telemetry")
            st.caption(f"Audit captured {len(df_qual)} rejected items (crops <16px or corrupted bounding coordinates).")
            st.dataframe(df_qual, use_container_width=True)

    with tab_specs:
        st.markdown("#### MODEL SPECIFICATIONS & HARDWARE PROFILE")
        st.markdown("""
        - **Architecture**: EfficientNetB0 (Tan & Le, Google Research)
        - **Pretrained Weights**: ImageNet-1k
        - **Input Resolution**: 224 &times; 224 &times; 3 RGB
        - **Normalization**: EfficientNet scaling (zero-centered scaled floats)
        - **Inference Preprocessing**: Aspect-ratio preserving letterboxing with neutral padding
        - **TTA Engine**: Multi-scale aspect-preserved evaluation (1.0x, 1.15x zoom, mirror flip)
        - **Classification Head**: GlobalAveragePooling2D + BatchNormalization + Dropout(0.3) + Dense(8, softmax)
        - **Loss Function**: Categorical Crossentropy with balanced class weighting
        - **Optimizer**: Adam (learning rate 1e-3 Stage 1, 1e-4 Stage 2)
        - **Hardware**: Direct CPU/GPU fallback with memory growth management
        """)


if __name__ == "__main__":
    main()

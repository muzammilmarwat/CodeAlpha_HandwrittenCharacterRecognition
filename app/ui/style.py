"""Premium Streamlit styling for Phase 4B."""

import streamlit as st


def apply_global_style() -> None:
    """Inject application CSS."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-soft: #1f2937;
            --border: rgba(148, 163, 184, 0.22);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --primary: #60a5fa;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .studio-hero {
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.96));
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .studio-hero h1 {
            margin: 0 0 0.25rem 0;
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            letter-spacing: 0;
        }
        .studio-hero p {
            color: var(--muted);
            margin: 0.2rem 0 0.8rem 0;
            font-size: 0.98rem;
        }
        .badge-row { display: flex; gap: 0.45rem; flex-wrap: wrap; }
        .badge {
            border: 1px solid var(--border);
            background: rgba(96, 165, 250, 0.12);
            color: #dbeafe;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .badge-success { background: rgba(34, 197, 94, 0.12); color: #bbf7d0; }
        .badge-warning { background: rgba(245, 158, 11, 0.14); color: #fde68a; }
        .section-title {
            margin: 1rem 0 0.45rem 0;
            font-weight: 750;
            font-size: 1.05rem;
        }
        .info-card, .result-card {
            border: 1px solid var(--border);
            background: rgba(17, 24, 39, 0.78);
            border-radius: 14px;
            padding: 1rem;
            height: 100%;
        }
        .metric-card {
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.72);
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        .metric-label { color: var(--muted); font-size: 0.78rem; margin-bottom: 0.25rem; }
        .metric-value { color: var(--text); font-size: 1.35rem; font-weight: 760; }
        .digit-result {
            font-size: clamp(4rem, 12vw, 7rem);
            font-weight: 850;
            text-align: center;
            line-height: 1;
            color: #bfdbfe;
        }
        .confidence-high { color: var(--success); }
        .confidence-medium { color: var(--warning); }
        .confidence-low { color: var(--error); }
        .muted { color: var(--muted); }
        .footer {
            margin-top: 2rem;
            color: var(--muted);
            font-size: 0.82rem;
            border-top: 1px solid var(--border);
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_class(confidence_band: str) -> str:
    """Return CSS class for a confidence band."""
    return {
        "high": "confidence-high",
        "medium": "confidence-medium",
        "low": "confidence-low",
    }.get(confidence_band, "confidence-medium")


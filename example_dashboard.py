"""
example_dashboard.py
====================

Runnable example for ``engine_dashboard.create_dashboard``.

Generates a synthetic engine cross-section image + a transient-simulation
dataframe (throttle step from idle through max and back to cruise) and opens
a Dash dashboard with three tabbed views.

    python example_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from engine_dashboard import create_dashboard


# ---------------------------------------------------------------------------
# Synthetic image
# ---------------------------------------------------------------------------


# Pixel coordinates of each station overlay on the generated image. Used both
# to draw the markers in the image and to wire up location_cols for the dash.
STATION_COORDS = {
    "2": (220, 300),
    "2.5": (370, 300),
    "3": (560, 300),
    "4": (700, 300),
    "4.5": (770, 300),
    "5": (950, 300),
    "8": (1140, 300),
}


def make_engine_image(path: Path, width: int = 1200, height: int = 600) -> Path:
    """Draw a stylised jet-engine cross-section and save it as PNG."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    cy = height // 2
    top, bot = int(height * 0.30), int(height * 0.70)

    # Inlet cone
    draw.polygon(
        [(50, cy), (200, top), (200, bot)],
        outline="black",
        width=2,
    )

    # Main body
    draw.rectangle([200, top, 1080, bot], outline="black", width=2)

    # Fan blades (vertical lines, fan region)
    for x in range(220, 380, 18):
        draw.line([(x, top + 4), (x, bot - 4)], fill="gray", width=1)

    # Compressor (tapered section, lines converge)
    for x in range(400, 640, 20):
        offset = int((x - 400) / 240 * 30)
        draw.line(
            [(x, top + offset + 6), (x, bot - offset - 6)],
            fill="gray",
            width=1,
        )

    # Combustor
    draw.rectangle(
        [640, top + 36, 740, bot - 36],
        outline="#C0392B",
        width=2,
    )
    draw.text((670, cy - 8), "BURN", fill="#C0392B")

    # Turbine blades
    for x in range(760, 1000, 22):
        offset = int((1000 - x) / 240 * 25)
        draw.line(
            [(x, top + offset + 6), (x, bot - offset - 6)],
            fill="gray",
            width=1,
        )

    # Exhaust nozzle
    draw.polygon(
        [(1080, top), (1080, bot), (1180, cy + 30), (1180, cy - 30)],
        outline="black",
        width=2,
    )

    # Section labels
    draw.text((250, top - 22), "FAN", fill="black")
    draw.text((460, top - 22), "COMPRESSOR", fill="black")
    draw.text((760, top - 22), "TURBINE", fill="black")
    draw.text((1100, top - 22), "EXHAUST", fill="black")

    # Station markers
    for sid, (x, y) in STATION_COORDS.items():
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], outline="#1F4E8C", width=2)
        # Center the station id text inside the circle
        bbox = draw.textbbox((0, 0), sid)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw / 2, y - th / 2 - 1), sid, fill="#1F4E8C")

    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Synthetic transient dataframe
# ---------------------------------------------------------------------------


def make_transient_df(n_steps: int = 60) -> pd.DataFrame:
    """
    One row per time step, columns for each station's Tt, Pt, W, Mach.

    Models a throttle profile: idle (0–1 s) → ramp up (1–3 s) → max (3–4 s)
    → ramp down to cruise (4–5 s). Each station's response is derived from
    the throttle with section-specific scaling.
    """
    t = np.linspace(0.0, 5.0, n_steps)

    # Throttle ∈ [0, 1]
    throttle = np.piecewise(
        t,
        [t < 1.0, (t >= 1.0) & (t < 3.0), (t >= 3.0) & (t < 4.0), t >= 4.0],
        [
            lambda x: 0.20 * np.ones_like(x),
            lambda x: 0.20 + 0.80 * (x - 1.0) / 2.0,
            lambda x: 1.00 - 0.05 * np.sin((x - 3.0) * np.pi),
            lambda x: 1.00 - 0.35 * (x - 4.0),
        ],
    )

    # Small noise so the plots aren't perfectly smooth.
    rng = np.random.default_rng(seed=42)
    noise = lambda scale: rng.normal(0.0, scale, size=n_steps)

    # Per-station scaling. Cold-section stations track throttle modestly;
    # the combustor and turbine spike harder.
    profiles = {
        "2":   {"Tt_base": 288, "Tt_gain": 30,   "Pt_base": 101,  "Pt_gain": 25,   "W_base": 100, "W_gain": 220, "Mach": 0.50},
        "2.5": {"Tt_base": 360, "Tt_gain": 70,   "Pt_base": 180,  "Pt_gain": 80,   "W_base":  60, "W_gain":  10, "Mach": 0.45},
        "3":   {"Tt_base": 780, "Tt_gain": 250,  "Pt_base": 1800, "Pt_gain": 1700, "W_base":  60, "W_gain":   8, "Mach": 0.30},
        "4":   {"Tt_base": 1500,"Tt_gain": 300,  "Pt_base": 1750, "Pt_gain": 1600, "W_base":  62, "W_gain":   8, "Mach": 0.10},
        "4.5": {"Tt_base": 1300,"Tt_gain": 220,  "Pt_base": 1100, "Pt_gain":  900, "W_base":  64, "W_gain":   8, "Mach": 0.50},
        "5":   {"Tt_base":  900,"Tt_gain": 150,  "Pt_base":  220, "Pt_gain":  150, "W_base":  64, "W_gain":   8, "Mach": 0.50},
        "8":   {"Tt_base":  870,"Tt_gain": 140,  "Pt_base":  105, "Pt_gain":   45, "W_base":  64, "W_gain":   8, "Mach": 1.00},
    }

    data = {"time_s": np.round(t, 3)}
    for sid, p in profiles.items():
        suffix = sid.replace(".", "p")  # column-name-safe: "2.5" → "2p5"
        data[f"Tt_{suffix}_K"]   = p["Tt_base"] + p["Tt_gain"] * throttle + noise(p["Tt_gain"] * 0.02)
        data[f"Pt_{suffix}_kPa"] = p["Pt_base"] + p["Pt_gain"] * throttle + noise(p["Pt_gain"] * 0.02)
        data[f"W_{suffix}_kgs"]  = p["W_base"]  + p["W_gain"]  * throttle + noise(p["W_gain"]  * 0.02)
        data[f"Mach_{suffix}"]   = p["Mach"]    + 0.10 * throttle         + noise(0.01)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Location & tab definitions
# ---------------------------------------------------------------------------


def build_location_cols(metric_groups):
    """
    Map each station's pixel position to the column names it should display
    on the image, for the given metric groups (e.g. ["Tt", "Pt"]).
    """
    out = {}
    suffix_map = {
        "Tt": "_K",
        "Pt": "_kPa",
        "W":  "_kgs",
        "Mach": "",
    }
    for sid, coord in STATION_COORDS.items():
        col_suffix = sid.replace(".", "p")
        cols = [f"{m}_{col_suffix}{suffix_map[m]}" for m in metric_groups]
        out[coord] = cols
    return out


def build_tabs(df: pd.DataFrame):
    """Three tabs grouping the metrics into thematic views."""
    # For each tab, pick a representative station's column to chart over time.
    return [
        {
            "label": "Pressures & Temps",
            "location_cols": build_location_cols(["Tt", "Pt"]),
            "plot_cols": ["Tt_3_K", "Tt_4_K", "Pt_3_kPa", "Pt_4_kPa"],
        },
        {
            "label": "Mass Flow",
            "location_cols": build_location_cols(["W"]),
            "plot_cols": ["W_2_kgs", "W_3_kgs", "W_4_kgs", "W_8_kgs"],
        },
        {
            "label": "Mach Number",
            "location_cols": build_location_cols(["Mach"]),
            "plot_cols": ["Mach_2", "Mach_3", "Mach_4p5", "Mach_8"],
        },
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    here = Path(__file__).parent
    image_path = here / "test_engine.png"
    make_engine_image(image_path)
    print(f"Wrote test image → {image_path}")

    df = make_transient_df(n_steps=60)
    print(f"Generated dataframe: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head(3))

    tabs = build_tabs(df)

    app = create_dashboard(
        image=str(image_path),
        df=df,
        location_cols=tabs[0]["location_cols"],  # default for fallback
        title="Transient Engine Simulation (synthetic)",
        subtitle=(
            f"{len(df)} time steps from t = {df['time_s'].min():.1f} s "
            f"to t = {df['time_s'].max():.1f} s · synthetic test data"
        ),
        label_col="time_s",
        x_col="time_s",
        tabs=tabs,
        frame_duration=200,
        width=1200,
    )
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()

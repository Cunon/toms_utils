"""
jt9d_dashboard.py
=================

Interactive dashboard for the Pratt & Whitney JT9D high-bypass turbofan.

Overlays per-station thermodynamic parameters (total temperature Tt, total
pressure Pt, mass flow W, and Mach number M) on the real ``JT9D_Cross_Section.jpg``
cross-section, and animates a throttle transient (idle -> takeoff -> cruise)
through the slider / Play button.

    python jt9d_dashboard.py            # opens at http://127.0.0.1:8050

The station values are *representative* JT9D-7-class sea-level-static numbers
(overall pressure ratio ~22, turbine inlet temperature ~1600 K, total inlet
airflow ~680 kg/s, bypass ratio ~5) scaled with throttle. They illustrate the
thermodynamic state along the gas path; they are not certified engine data.

Note on mass flow: station 2 (W) is the *total* inlet airflow. Downstream of
the fan the bypass stream splits off, so stations 2.5 onward report the *core*
flow only -- which is why W drops sharply behind the fan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine_dashboard import create_dashboard


# ---------------------------------------------------------------------------
# Station geometry -- pixel coordinates calibrated against JT9D_Cross_Section.jpg
# (737 x 506). Engine centerline (shaft) sits at y = 212. Hot-section stations
# are staggered above/below the centerline so the value boxes don't overlap.
# ---------------------------------------------------------------------------

# (x, y) pixel position of each station's value box on the cross-section.
STATION_COORDS = {
    "2":   (70, 212),    # fan face / engine inlet            (on centerline)
    "2.5": (180, 212),   # LP compressor (booster) exit       (on centerline)
    "3":   (450, 120),   # HP compressor exit / burner inlet  (above)
    "4":   (528, 300),   # combustor exit / HPT inlet (TIT)   (below)
    "4.5": (586, 120),   # HP turbine exit / LPT inlet        (above)
    "5":   (650, 300),   # LP turbine exit                    (below)
    "8":   (700, 212),   # core exhaust nozzle                (on centerline)
}

# Representative idle and takeoff (max-throttle) values per station.
#   Tt  [K]   total temperature
#   Pt  [kPa] total pressure
#   W   [kg/s] mass flow (total at st.2, core flow downstream)
#   Mach      flow Mach number
# Throttle ramps from 0.20 (idle) to 1.00 (takeoff); values interpolate linearly.
STATION_STATE = {
    #          Tt_idle Tt_to   Pt_idle Pt_to   W_idle W_to   M_idle M_to
    "2":   dict(Tt=(288,   288),  Pt=(101,   101),  W=(210,  680),  M=(0.25, 0.50)),
    "2.5": dict(Tt=(312,   362),  Pt=(150,   250),  W=( 40,  113),  M=(0.42, 0.46)),
    "3":   dict(Tt=(560,   800),  Pt=(700,  2250),  W=( 40,  113),  M=(0.28, 0.30)),
    "4":   dict(Tt=(980,  1600),  Pt=(660,  2140),  W=( 41,  115),  M=(0.10, 0.12)),
    "4.5": dict(Tt=(720,  1150),  Pt=(220,   560),  W=( 41,  115),  M=(0.40, 0.42)),
    "5":   dict(Tt=(540,   800),  Pt=(112,   175),  W=( 41,  115),  M=(0.45, 0.50)),
    "8":   dict(Tt=(540,   800),  Pt=(106,   165),  W=( 41,  115),  M=(0.60, 1.00)),
}

# Units appended to each parameter's column name, so the overlay reads e.g.
# "Tt_4_K: 1600".
PARAM_UNITS = {"Tt": "_K", "Pt": "_kPa", "W": "_kgs", "M": ""}


def _safe(sid: str) -> str:
    """Column-name-safe station id:  '2.5' -> '2p5'."""
    return sid.replace(".", "p")


def col_name(param: str, sid: str) -> str:
    return f"{param}_{_safe(sid)}{PARAM_UNITS[param]}"


# ---------------------------------------------------------------------------
# Transient dataframe
# ---------------------------------------------------------------------------


def make_transient_df(n_steps: int = 80) -> pd.DataFrame:
    """
    One row per time step. Models a throttle profile:
        idle (0-1 s) -> ramp up (1-3 s) -> takeoff (3-4 s)
        -> ramp down to cruise (4-6 s).
    Each station value interpolates between its idle and takeoff state with the
    throttle, plus a little noise so the trend plots aren't perfectly smooth.
    """
    t = np.linspace(0.0, 6.0, n_steps)

    # Throttle in [0.20, 1.00].
    throttle = np.piecewise(
        t,
        [t < 1.0, (t >= 1.0) & (t < 3.0), (t >= 3.0) & (t < 4.0), t >= 4.0],
        [
            lambda x: 0.20 * np.ones_like(x),
            lambda x: 0.20 + 0.80 * (x - 1.0) / 2.0,
            lambda x: 1.00 - 0.04 * np.sin((x - 3.0) * np.pi),
            lambda x: np.clip(1.00 - 0.35 * (x - 4.0), 0.65, 1.0),
        ],
    )

    rng = np.random.default_rng(seed=7)

    def lerp(idle, takeoff, jitter):
        # throttle 0.2 -> idle, 1.0 -> takeoff
        frac = (throttle - 0.20) / 0.80
        base = idle + (takeoff - idle) * frac
        return base + rng.normal(0.0, jitter, size=n_steps)

    data = {"time_s": np.round(t, 3), "throttle": np.round(throttle, 3)}
    for sid, state in STATION_STATE.items():
        for param, (idle, takeoff) in state.items():
            span = abs(takeoff - idle)
            jitter = max(span * 0.015, 0.002 if param == "M" else 0.0)
            data[col_name(param, sid)] = lerp(idle, takeoff, jitter)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Overlay wiring
# ---------------------------------------------------------------------------


def build_location_cols() -> dict:
    """Map each station's pixel box to the four parameter columns it shows."""
    return {
        coord: [col_name(p, sid) for p in ("Tt", "Pt", "W", "M")]
        for sid, coord in STATION_COORDS.items()
    }


def build_col_labels() -> dict:
    """
    Short label per column for the on-image boxes. The station id rides on the
    first (Tt) line so every box is self-identifying; the rest are param+unit:
        Tt_4_K -> 'St4 Tt_K',  Pt_4_kPa -> 'Pt_kPa',  W_4_kgs -> 'W_kgs',  M_4 -> 'M'
    """
    short = {"Tt": "Tt_K", "Pt": "Pt_kPa", "W": "W_kgs", "M": "M"}
    out = {}
    for sid in STATION_COORDS:
        for p in ("Tt", "Pt", "W", "M"):
            label = f"St{sid} {short[p]}" if p == "Tt" else short[p]
            out[col_name(p, sid)] = label
    return out


def build_tabs() -> list:
    """
    Trend-plot tabs. The cross-section overlay shows all four parameters at
    every station in each tab; the tabs only change which transient is charted
    on the right so each family of curves stays readable.
    """
    return [
        {
            "label": "Temperatures (Tt)",
            "plot_cols": [col_name("Tt", s) for s in ("2.5", "3", "4", "5")],
        },
        {
            "label": "Pressures (Pt)",
            "plot_cols": [col_name("Pt", s) for s in ("2.5", "3", "4", "5")],
        },
        {
            "label": "Flow & Mach",
            "plot_cols": [col_name("W", "2"), col_name("W", "3"),
                          col_name("M", "8"), col_name("Tt", "4")],
        },
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_app(width: int = 1280):
    here = Path(__file__).parent
    image_path = here / "JT9D_Cross_Section.jpg"

    df = make_transient_df()

    subtitle = (
        f"Representative JT9D-7-class throttle transient (idle → takeoff → "
        f"cruise) · {len(df)} steps over {df['time_s'].max():.0f} s · "
        f"takeoff design point: OPR ≈ 22, TIT ≈ 1600 K, total airflow ≈ 680 kg/s, "
        f"BPR ≈ 5 · Tt total temp, Pt total pressure, W mass flow, M Mach"
    )

    return create_dashboard(
        image=str(image_path),
        df=df,
        location_cols=build_location_cols(),
        title="JT9D Turbofan — Station Thermodynamics",
        subtitle=subtitle,
        label_col="time_s",
        x_col="time_s",
        tabs=build_tabs(),
        frame_duration=120,
        width=width,
        font_size=11,
        plot_layout="right",
        col_labels=build_col_labels(),
    )


def main():
    app = build_app()
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()

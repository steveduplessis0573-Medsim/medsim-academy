"""
capno_build_library.py
Generate capnography waveform PNGs for MedSim Academy.
Run once: python capno_build_library.py
Output: capno_images/<slug>.png (green-on-black monitor style)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR    = "capno_images"
BG         = "#000000"
WAVE_CLR   = "#00FF00"
GRID_CLR   = "#1c2e1c"
TEXT_CLR   = "#00FF00"
FIG_W, FIG_H, DPI = 8, 3, 100

os.makedirs(OUT_DIR, exist_ok=True)

# ── waveform primitives ──────────────────────────────────────────────────────

def _breath(t_arr, t0, period, etco2, baseline=0.0,
            insp_f=0.42, up_f=0.09, plat_f=0.40,
            shark_fin=False, plat_slope=0.0):
    """Overwrite t_arr slice with one capnography breath cycle."""
    t_insp = t0 + insp_f  * period
    t_up   = t_insp + up_f  * period
    t_plat = t_up   + plat_f * period
    t_end  = t0     + period

    i_insp  = (t_arr >= t0)    & (t_arr < t_insp)
    i_up    = (t_arr >= t_insp) & (t_arr < t_up)
    i_plat  = (t_arr >= t_up)   & (t_arr < t_plat)
    i_down  = (t_arr >= t_plat) & (t_arr < t_end)

    wave = np.zeros_like(t_arr)
    wave[i_insp] = baseline

    fu = (t_arr[i_up] - t_insp) / max(t_up - t_insp, 1e-9)
    if shark_fin:
        # slow, rounded upstroke for obstructive pattern
        wave[i_up] = baseline + (etco2 * 0.65) * np.power(fu, 0.5)
    else:
        wave[i_up] = baseline + (etco2 - baseline) * fu

    fp = (t_arr[i_plat] - t_up) / max(t_plat - t_up, 1e-9)
    if shark_fin:
        # rising plateau — the hallmark shark-fin
        wave[i_plat] = etco2 * 0.65 + (etco2 - etco2 * 0.65) * fp
    else:
        wave[i_plat] = etco2 + plat_slope * fp

    fd = (t_arr[i_down] - t_plat) / max(t_end - t_plat, 1e-9)
    end_plat = etco2 + plat_slope if not shark_fin else etco2
    wave[i_down] = end_plat * (1 - fd) + baseline * fd

    return wave


def _build(rr, etco2, n_breaths=3, baseline=0.0,
           shark_fin=False, plat_slope=0.0, sr=200):
    period = 60.0 / rr
    total  = n_breaths * period
    t = np.linspace(0, total, int(total * sr))
    w = np.full_like(t, baseline)
    for b in range(n_breaths):
        w += _breath(t, b * period, period, etco2,
                     baseline=baseline, shark_fin=shark_fin,
                     plat_slope=plat_slope)
    return t, np.clip(w, 0, 80)


# ── render helper ─────────────────────────────────────────────────────────────

def _render(slug, t, wave, etco2_label, rr_label, caption, ylim=(0, 70)):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # subtle grid
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="-")

    ax.plot(t, wave, color=WAVE_CLR, linewidth=1.8, antialiased=True)

    ax.set_ylim(*ylim)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.set_xlabel("Time (s)", color=TEXT_CLR, fontsize=8)
    ax.set_ylabel("CO₂ (mmHg)", color=TEXT_CLR, fontsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_CLR)

    # EtCO2 readout — top-right, large
    ax.text(0.98, 0.97, f"EtCO₂  {etco2_label} mmHg",
            transform=ax.transAxes, color=WAVE_CLR,
            fontsize=13, fontweight="bold", va="top", ha="right",
            fontfamily="monospace")
    ax.text(0.98, 0.78, f"RR  {rr_label} /min",
            transform=ax.transAxes, color=WAVE_CLR,
            fontsize=10, va="top", ha="right", fontfamily="monospace")
    # caption bottom-left
    ax.text(0.02, 0.05, caption,
            transform=ax.transAxes, color=WAVE_CLR,
            fontsize=8, va="bottom", ha="left", alpha=0.75)

    plt.tight_layout(pad=0.4)
    path = os.path.join(OUT_DIR, f"{slug}.png")
    plt.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  OK  {slug}.png")


# ── waveform definitions ──────────────────────────────────────────────────────

def gen_normal():
    t, w = _build(rr=15, etco2=38, n_breaths=3)
    _render("normal", t, w, "38", "15", "Normal waveform")

def gen_bronchospasm():
    t, w = _build(rr=18, etco2=44, n_breaths=3, shark_fin=True)
    _render("bronchospasm", t, w, "44", "18", "Obstructive pattern — bronchospasm / COPD")

def gen_hyperventilation():
    t, w = _build(rr=30, etco2=22, n_breaths=5)
    _render("hyperventilation", t, w, "22", "30", "Hyperventilation")

def gen_hypoventilation():
    t, w = _build(rr=8, etco2=58, n_breaths=2)
    _render("hypoventilation", t, w, "58", "8", "Hypoventilation")

def gen_esophageal():
    """Decreasing waveforms after esophageal intubation — CO2 washes out."""
    sr = 200
    rr = 15
    period = 60.0 / rr
    n = 4
    total = n * period + 2.0   # 2s flat at end
    t = np.linspace(0, total, int(total * sr))
    w = np.zeros_like(t)
    amplitudes = [32, 18, 7, 2]  # washing out
    for b, amp in enumerate(amplitudes):
        w += _breath(t, b * period, period, amp)
    _render("esophageal", t, w, "0", "--", "Esophageal intubation — waveform absent")

def gen_apnea():
    t = np.linspace(0, 12, 2400)
    w = np.zeros_like(t)
    _render("apnea", t, w, "0", "0", "Apnea — no respiratory effort")

def gen_cardiac_arrest():
    """Very low, irregular near-zero signal — no effective ventilation/perfusion."""
    sr = 200
    t = np.linspace(0, 12, int(12 * sr))
    # tiny noise around 4 mmHg
    rng = np.random.default_rng(42)
    w = np.abs(rng.normal(0, 1.2, t.shape))
    w = np.clip(w, 0, 8)
    # smooth slightly
    kernel = np.ones(20) / 20
    w = np.convolve(w, kernel, mode="same")
    _render("cardiac_arrest", t, w, "4", "0", "Cardiac arrest — no perfusion")

def gen_rosc():
    """CPR-phase low bumps then sudden EtCO2 rise at ROSC."""
    sr = 200
    # CPR phase: 7 seconds, low bumps at compression rate ~100/min
    comp_rate = 100.0
    comp_period = 60.0 / comp_rate
    cpr_dur = 7.0
    t_cpr = np.linspace(0, cpr_dur, int(cpr_dur * sr))
    w_cpr = np.zeros_like(t_cpr)
    n_comps = int(cpr_dur / comp_period)
    for c in range(n_comps):
        t0 = c * comp_period
        mask = (t_cpr >= t0) & (t_cpr < t0 + comp_period)
        frac = (t_cpr[mask] - t0) / comp_period
        bump = 10 * np.sin(np.pi * frac) ** 2
        w_cpr[mask] = bump

    # ROSC: rapid rise over 1.5s then sustained at 36
    t_rise = np.linspace(0, 1.5, int(1.5 * sr))
    w_rise = 10 + (36 - 10) * (t_rise / 1.5)

    # Post-ROSC: 2 normal-ish breaths
    t_post, w_post = _build(rr=14, etco2=36, n_breaths=2)

    t_all = np.concatenate([t_cpr,
                             t_cpr[-1] + t_rise,
                             t_cpr[-1] + t_rise[-1] + t_post])
    w_all = np.concatenate([w_cpr, w_rise, w_post])

    _render("rosc", t_all, w_all, "36", "14",
            "ROSC — sudden EtCO₂ rise after CPR")

def gen_rebreathing():
    """Elevated CO2 baseline — CO2 never returns to zero."""
    t, w = _build(rr=15, etco2=44, n_breaths=3, baseline=8.0)
    _render("rebreathing", t, w, "44", "15",
            "Rebreathing — elevated baseline CO₂")

def gen_cpr():
    """Active CPR — rhythmic compression artifacts, low EtCO2."""
    sr = 200
    comp_rate = 100.0
    comp_period = 60.0 / comp_rate
    total = 8.0
    t = np.linspace(0, total, int(total * sr))
    w = np.zeros_like(t)
    n_comps = int(total / comp_period)
    for c in range(n_comps):
        t0 = c * comp_period
        mask = (t >= t0) & (t < t0 + comp_period)
        frac = (t[mask] - t0) / comp_period
        w[mask] = 13 * np.sin(np.pi * frac) ** 2
    _render("cpr", t, w, "13", "--", "CPR in progress — compression artifacts")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating capnography waveforms...")
    gen_normal()
    gen_bronchospasm()
    gen_hyperventilation()
    gen_hypoventilation()
    gen_esophageal()
    gen_apnea()
    gen_cardiac_arrest()
    gen_rosc()
    gen_rebreathing()
    gen_cpr()
    print(f"\nDone — 10 waveforms saved to '{OUT_DIR}/'")

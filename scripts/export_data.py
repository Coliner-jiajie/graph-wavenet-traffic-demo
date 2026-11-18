"""Export the data behind the viewer: run a trained Graph WaveNet checkpoint on the METR-LA test set.

This project only contains the viewer and the exported predictions. The model code (model.py, util.py), the
sensor graph and the METR-LA data come from the original Graph WaveNet repository:
    https://github.com/nnzhan/Graph-WaveNet
Clone it, prepare the data as its README describes, then point GWNET_REPO at it.

Usage (needs PyTorch, numpy, pandas, tables):
    python scripts/export_data.py            # fast: reuses data/full_test_summary.json
    python scripts/export_data.py --full     # recompute whole-test-set MAE for every mask level (about 15 min on CPU)

Needs data/sensor_locations.csv (run scripts/fetch_locations.py once).

Environment variables:
    GWNET_REPO   path to the Graph-WaveNet clone      (default ../Graph-WaveNet next to this project)
    GWNET_CKPT   checkpoint to evaluate               (default checkpoint/metr_exp1_best_2.71.pth in this project)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = Path(os.environ.get("GWNET_REPO", ROOT.parent / "Graph-WaveNet"))
CKPT = Path(os.environ.get("GWNET_CKPT", ROOT / "checkpoint" / "metr_exp1_best_2.71.pth"))
OUT = ROOT / "data" / "demo_data.json"
SUMMARY = ROOT / "data" / "full_test_summary.json"

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="recompute full-test-set MAE for every mask level (slow)")
args = ap.parse_args()

sys.path.insert(0, str(REPO))
import model as gw  # noqa: E402  (lives in the Graph-WaveNet repo)
import util  # noqa: E402

device = torch.device("cpu")
sensor_ids, _, adj_mx = util.load_adj(str(REPO / "data/sensor_graph/adj_mx.pkl"), "doubletransition")
supports = [torch.tensor(i).to(device) for i in adj_mx]
net = gw.gwnet(device, 207, 0.3, supports=supports, gcn_bool=True, addaptadj=True, aptinit=None,
               in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512)
net.load_state_dict(torch.load(CKPT, map_location=device))
net.eval()

data = util.load_dataset(str(REPO / "data/METR-LA"), 64, 64, 64)
scaler = data["scaler"]
x_test, y_test = data["x_test"], data["y_test"]
N = x_test.shape[0]
print("test samples", N, "checkpoint", CKPT.name)

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
HORIZONS = [3, 6, 12]          # forecast steps shown in the demo (x5 min)
STEPS = 576                    # two days of 5-minute steps


def predict(lo, hi, rate, seed):
    """Predictions in mph for test samples lo..hi. Masking mirrors test_robustness.py (whole node zeroed per window)."""
    g = torch.Generator().manual_seed(seed)
    outs = []
    with torch.no_grad():
        for s in range(lo, hi, 64):
            e = min(s + 64, hi)
            x = torch.Tensor(x_test[s:e]).transpose(1, 3)
            if rate > 0:
                B, _, Nn, _ = x.shape
                x = x * (torch.rand(B, 1, Nn, 1, generator=g) > rate).float()
            outs.append(scaler.inverse_transform(net(x))[..., 0].numpy())   # (B, 12, N)
    return np.concatenate(outs, axis=0)


def mae_rmse(pred, real):
    m = real != 0
    err = pred[m] - real[m]
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


real_all = y_test[..., 0]      # (N, 12, 207) ground-truth speed in mph

if args.full:
    summary = {}
    for r in RATES:
        pred = predict(0, N, r, seed=1)
        mae, rmse = mae_rmse(pred, real_all)
        per_h = [mae_rmse(pred[:, h], real_all[:, h])[0] for h in range(12)]
        summary[str(int(r * 100))] = {"mae": round(mae, 3), "rmse": round(rmse, 3),
                                      "mae_by_horizon": [round(v, 3) for v in per_h]}
        print(f"mask {int(r * 100):2d}%  MAE {mae:.3f}  RMSE {rmse:.3f}")
    SUMMARY.write_text(json.dumps(summary, indent=1))
else:
    summary = json.loads(SUMMARY.read_text())

# ---- map test samples to wall-clock time through the raw h5 index ----
index = pd.read_hdf(REPO / "data/metr-la.h5").index
first_test_sample = (len(index) - 23) - N


def target_time(i, h):
    return index[first_test_sample + i + 11 + h]


# ---- window: the Tuesday-midnight start (Tue + Wed) with the most complete sensor data ----
start, best = None, -1
for i in range(0, N - STEPS - 24):
    t = target_time(i, 1)
    if t.hour == 0 and t.minute == 0 and t.dayofweek == 1:
        cnt = int(((real_all[i:i + STEPS, 0, :] != 0).mean(axis=0) > 0.98).sum())
        if cnt > best:
            start, best = i, cnt
win_lo, win_hi = start, start + STEPS + 12
print("window", target_time(start, 1), "->", target_time(start + STEPS - 1, 1))

# ---- all 207 sensors, with map coordinates ----
import base64
import csv

loc = {r["sensor_id"]: (float(r["latitude"]), float(r["longitude"]))
       for r in csv.DictReader(open(ROOT / "data" / "sensor_locations.csv"))}
sel = list(range(207))
act = real_all[win_lo:win_lo + STEPS, 0, :]                       # horizon-1 ground truth over the window
avg = [float(act[:, s][act[:, s] != 0].mean()) if (act[:, s] != 0).any() else 0.0 for s in sel]

# ---- learned adjacency neighbours ----
with torch.no_grad():
    adp = torch.softmax(torch.relu(net.nodevec1 @ net.nodevec2), dim=1).numpy()
neighbors = {}
for s in sel:
    row = adp[s].copy()
    row[s] = -1
    neighbors[str(s)] = [{"idx": int(t), "id": str(sensor_ids[t]), "w": round(float(adp[s][t]), 4)}
                         for t in np.argsort(-row)[:5]]


def pack(arr):
    """(sensors, STEPS) float array -> base64 of little-endian int16 tenths of mph; -1 marks a missing value."""
    a = np.where(np.isnan(arr) | (arr <= 0), -1, np.round(arr * 10)).astype("<i2")
    return base64.b64encode(a.tobytes()).decode("ascii")


# ---- series ----
actual = pack(act.T.astype(float))
preds = {}
for r in RATES:
    p = predict(win_lo, win_hi, r, seed=7)                        # (STEPS + 12, 12, 207)
    per_h = {}
    for h in HORIZONS:
        m = np.full((207, STEPS), np.nan)
        for k in range(h - 1, STEPS):                              # forecast for time k was made h steps earlier
            m[:, k] = p[k - (h - 1), h - 1, :]
        per_h[str(h)] = pack(m)
    preds[str(int(r * 100))] = per_h

OUT.write_text(json.dumps({
    "checkpoint": CKPT.name,
    "summary_full_test": summary,
    "horizons_steps": HORIZONS,
    "step_minutes": 5,
    "steps": STEPS,
    "times": [target_time(win_lo + k, 1).strftime("%Y-%m-%d %H:%M") for k in range(STEPS)],
    "sensors": [{"idx": s, "id": str(sensor_ids[s]), "lat": loc[str(sensor_ids[s])][0], "lon": loc[str(sensor_ids[s])][1],
                 "avg": round(avg[s], 1)} for s in sel],
    "actual": actual,
    "pred": preds,
    "learned_neighbors": neighbors,
}, separators=(",", ":")))
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")

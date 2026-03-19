#!/usr/bin/env python3
"""Convert phase10_model_v10.py into a clean Jupyter notebook."""
import json, re

with open("phase10_model_v10.py") as f:
    src = f.read()

# Split on ══ section markers
section_re = re.compile(r'^# ═+\n# (\d+\..+?)\n# ═+$', re.MULTILINE)
parts = section_re.split(src)

cells = []

def md(text):
    return {"cell_type": "markdown", "id": f"md_{len(cells):04x}",
            "metadata": {}, "source": [text]}

def code(text):
    lines = text.strip().split('\n')
    src_lines = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "code", "execution_count": None,
            "id": f"cc_{len(cells):04x}",
            "metadata": {}, "outputs": [], "source": src_lines}

cells.append(md(
    "# V10 — Full Pipeline: Time-Based CV + Optuna + Score Smoothing\n\n"
    "**All improvements applied in one pipeline:**\n\n"
    "| # | Improvement | Expected Impact |\n"
    "|---|-------------|----------------|\n"
    "| 1 | Time-based cross-validation (no temporal leakage) | Better generalization |\n"
    "| 2 | Velocity/Ratio/Burst/Aggregation features | +AUC, +F1 |\n"
    "| 3 | Optuna tuning: 80 LGB + 60 XGB + 40 CAT trials | +AUC |\n"
    "| 4 | Weighted ensemble: 0.4 LGB + 0.3 XGB + 0.3 CAT | Stable predictions |\n"
    "| 5 | Score-smoothed temporal windows (rolling mean 5d) | Temporal IoU 0.72+ |\n"
    "| 6 | Error analysis loop: FP/FN targeted features | +F1 |\n"
    "| 7 | Learned RH post-filter | RH_7 > 0.95 |\n\n"
    "**Targets:** AUC ≥ 0.994 | F1 ≥ 0.91 | IoU ≥ 0.72 | All RH > 0.95"
))

# Header block (before first section)
header = parts[0].strip()
if header:
    cells.append(code(header))

i = 1
while i < len(parts) - 1:
    title = parts[i].strip()
    body  = parts[i+1].strip()
    cells.append(md(f"## {title}"))
    if body:
        cells.append(code(body))
    i += 2

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = "phase10_model_v10.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Created {out} ({len(cells)} cells)")
for c in cells:
    preview = ''.join(c.get('source', []))[:70].replace('\n', ' ')
    print(f"  [{c['cell_type'][:4]}] {preview}")

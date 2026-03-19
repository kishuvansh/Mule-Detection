#!/usr/bin/env python3
"""Convert phase9_model_v9.py (V10 logic) into a Jupyter notebook."""
import json, re

with open("phase9_model_v9.py") as f:
    src = f.read()

# Split on section delimiters (# ── N. Title ──)
section_re = re.compile(r'^# ── (\d+[\.\s].+?) ──+$', re.MULTILINE)
parts = section_re.split(src)

cells = []

def md_cell(text):
    return {"cell_type": "markdown", "id": f"md_{len(cells):04x}",
            "metadata": {}, "source": [text]}

def code_cell(code):
    lines = code.strip().split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "code", "execution_count": None,
            "id": f"cc_{len(cells):04x}",
            "metadata": {}, "outputs": [], "source": source}

# Title markdown cell
cells.append(md_cell(
    "# V10 — Push All Metrics Over Target\n\n"
    "**Fixes from V9** (AUC=0.990, F1=0.888, IoU=0.236):\n\n"
    "| Fix | Root Cause | Solution |\n"
    "|-----|-----------|----------|\n"
    "| IoU 0.236 → 0.72+ | 10-90% CDF trim collapsed sparse windows to 0–1 day | 15-85% trim, only when ≥8 active days AND >30d span; 7d minimum |\n"
    "| AUC 0.990 → 0.994+ | ExtraTrees had lower AUC, pulled weighted blend down | Remove ExtraTrees; keep LGB+XGB+CatBoost with higher n_estimators |\n"
    "| F1 0.888 → 0.91+ | Already using PR-curve F1-optimal threshold | Higher-capacity models should push OOF F1 further |\n"
    "| RH_7 | Too-perfect decoys | Learned 2nd-stage LGB filter on signal entropy/variance |"
))

header = parts[0].strip()
if header:
    cells.append(code_cell(header))

# Process sections
i = 1
while i < len(parts) - 1:
    title = parts[i].strip()
    body  = parts[i+1].strip()
    cells.append(md_cell(f"## {title}"))
    if body:
        cells.append(code_cell(body))
    i += 2

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("phase9_model_v9.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Regenerated phase9_model_v9.ipynb with {len(cells)} cells")

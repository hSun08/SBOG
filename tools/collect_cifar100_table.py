#!/usr/bin/env python3
"""Turn a CIFAR-100 bootstrap JSON (sbog/test_detector_cifar100_bootstrap.py --output_json) into the Table 4 row
and compare it with the numbers reported in the paper (arXiv 2609.31470, Table 4)."""
import argparse, json, sys

PAPER = {  # dataset -> (FPR95 mean, FPR95 std, AUROC mean, AUROC std)
    "normal": {
        "SVHN": (5.38, 0.64, 98.67, 0.10), "Places365": (47.48, 1.70, 89.70, 0.34), "LSUN_C": (3.48, 0.52, 99.13, 0.06),
        "iSUN": (0.10, 0.07, 99.95, 0.01), "Texture": (5.74, 0.61, 98.69, 0.11), "overall_average": (12.44, 0.39, 97.23, 0.08)},
    "robust": {
        "SVHN": (5.49, 0.57, 98.69, 0.09), "Places365": (50.60, 1.51, 88.28, 0.39), "LSUN_C": (2.96, 0.43, 99.10, 0.07),
        "iSUN": (0.35, 0.13, 99.88, 0.02), "Texture": (5.46, 0.54, 98.78, 0.10), "overall_average": (12.97, 0.36, 96.95, 0.05)},
}
ORDER = ["SVHN", "Places365", "LSUN_C", "iSUN", "Texture", "overall_average"]
LABEL = {"SVHN": "SVHN", "Places365": "Places365", "LSUN_C": "LSUN", "iSUN": "iSUN", "Texture": "Textures", "overall_average": "Average"}

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--branch", choices=["normal", "robust"], required=True)
ap.add_argument("--tol_fpr", type=float, default=2.0, help="flag |diff| in FPR95 above this (percentage points)")
ap.add_argument("--tol_auroc", type=float, default=0.5)
a = ap.parse_args()
r = json.load(open(a.json))
rows = {**r["datasets"], "overall_average": r["overall_average"]}
print("ID error rate: {:.2f}".format(r.get("error_rate", float("nan"))))
print("{:<10} {:>22} {:>22} | {:>22} {:>22} | flag".format("dataset", "FPR95 (ours)", "AUROC (ours)", "FPR95 (paper)", "AUROC (paper)"))
bad = 0
latex = []
for k in ORDER:
    m = rows[k]; pf, ps, pa, pas = PAPER[a.branch][k]
    f, fs, au, aus = m["fpr95"]["mean"], m["fpr95"]["std"], m["auroc"]["mean"], m["auroc"]["std"]
    flag = ""
    if abs(f - pf) > a.tol_fpr: flag += " FPR95 off by {:+.2f}".format(f - pf)
    if abs(au - pa) > a.tol_auroc: flag += " AUROC off by {:+.2f}".format(au - pa)
    bad += bool(flag)
    print("{:<10} {:>22} {:>22} | {:>22} {:>22} |{}".format(LABEL[k], "{:.2f}±{:.2f}".format(f, fs), "{:.2f}±{:.2f}".format(au, aus),
                                                       "{:.2f}±{:.2f}".format(pf, ps), "{:.2f}±{:.2f}".format(pa, pas), flag or " ok"))
    latex.append("{:.2f}$\\pm${:.2f} & {:.2f}$\\pm${:.2f}".format(f, fs, au, aus))
print("\nLaTeX row ({}): SBOG{} & {} \\\\".format(a.branch, " (robust)" if a.branch == "robust" else "", " & ".join(latex)))
print("\nRESULT:", "all datasets within tolerance of the paper" if bad == 0 else "{} dataset(s) outside tolerance".format(bad))

#!/usr/bin/env python3
"""Turn an ImageNet-100 bootstrap JSON (sbog/test_detector_in100_bootstrap.py --output_json) into the Table 3 row
and compare it with the numbers reported in the paper (arXiv 2609.31470, Table 3)."""
import argparse, json

PAPER = {  # dataset -> (FPR95 mean, std, AUROC mean, std)
    "normal": {"iNaturalist": (9.44, 0.73, 98.09, 0.10), "Places365": (14.40, 1.01, 97.09, 0.16), "SUN": (13.22, 0.92, 97.61, 0.11),
               "Textures": (24.26, 1.06, 94.82, 0.23), "overall_average": (15.33, 0.46, 96.90, 0.08)},
    "robust": {"iNaturalist": (9.89, 0.76, 97.99, 0.10), "Places365": (13.24, 0.85, 97.21, 0.15), "SUN": (11.81, 0.99, 97.64, 0.11),
               "Textures": (24.06, 1.06, 95.05, 0.22), "overall_average": (14.75, 0.46, 96.97, 0.08)},
}
ORDER = ["iNaturalist", "Places365", "SUN", "Textures", "overall_average"]
LABEL = {"iNaturalist": "iNaturalist", "Places365": "Places", "SUN": "SUN", "Textures": "Textures", "overall_average": "Average"}

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True); ap.add_argument("--branch", choices=["normal", "robust"], required=True)
ap.add_argument("--tol_fpr", type=float, default=2.0); ap.add_argument("--tol_auroc", type=float, default=0.5)
a = ap.parse_args()
r = json.load(open(a.json)); rows = {**r["datasets"], "overall_average": r["overall_average"]}
print("ID error rate: {:.2f}".format(r.get("error_rate", float("nan"))))
print("{:<12} {:>20} {:>20} | {:>20} {:>20} | flag".format("dataset", "FPR95 (ours)", "AUROC (ours)", "FPR95 (paper)", "AUROC (paper)"))
bad = 0; latex = []
for k in ORDER:
    m = rows[k]; pf, ps, pa, pas = PAPER[a.branch][k]
    f, fs, au, aus = m["fpr95"]["mean"], m["fpr95"]["std"], m["auroc"]["mean"], m["auroc"]["std"]
    flag = ""
    if abs(f - pf) > a.tol_fpr: flag += " FPR95 off by {:+.2f}".format(f - pf)
    if abs(au - pa) > a.tol_auroc: flag += " AUROC off by {:+.2f}".format(au - pa)
    bad += bool(flag)
    print("{:<12} {:>20} {:>20} | {:>20} {:>20} |{}".format(LABEL[k], "{:.2f}±{:.2f}".format(f, fs), "{:.2f}±{:.2f}".format(au, aus),
                                                      "{:.2f}±{:.2f}".format(pf, ps), "{:.2f}±{:.2f}".format(pa, pas), flag or " ok"))
    latex.append("{:.2f}$\\pm${:.2f} & {:.2f}$\\pm${:.2f}".format(f, fs, au, aus))
print("\nLaTeX row ({}): SBOG{} & {} \\\\".format(a.branch, " (robust)" if a.branch == "robust" else "", " & ".join(latex)))
print("\nRESULT:", "all datasets within tolerance of the paper" if bad == 0 else "{} dataset(s) outside tolerance".format(bad))

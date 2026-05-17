#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 report/Report.md 中所有数据点的来源。在项目根目录运行。"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(rel_path):
    with open(os.path.join(ROOT, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


def count_samples(rel_csv):
    """CSV 行数减去表头 = 样本数。"""
    df = pd.read_csv(os.path.join(ROOT, rel_csv))
    return len(df)


def main():
    print("=" * 70)
    print("1.1 数据集统计（报告 §2 表 1）")
    print("=" * 70)
    for name, rel in [("ESOL", "data/esol.csv"), ("Lipophilicity", "data/lipo.csv")]:
        df = pd.read_csv(os.path.join(ROOT, rel))
        col = "target" if "target" in df.columns else df.columns[-1]
        print(f"{name}: n={len(df)}  "
              f"target min={df[col].min():.2f} max={df[col].max():.2f} "
              f"mean={df[col].mean():.2f} std={df[col].std():.2f}")
    print("ESOL  random: train={} test={}".format(
        count_samples("data/esol_random_train.csv"),
        count_samples("data/esol_random_test.csv")))
    print("Lipo  random: train={} test={}".format(
        count_samples("data/lipo_random_train.csv"),
        count_samples("data/lipo_random_test.csv")))
    e = load_json("models/results_esol_random_train.json")
    print(f"ESOL 内部划分: n_train={e['n_train']} n_val={e['n_val']} "
          f"n_test={e['n_test']}")

    print("\n" + "=" * 70)
    print("1.2 Chemprop baseline 主结果（报告 §4.1 表 2）")
    print("=" * 70)
    for name, rel in [("ESOL", "models/results_esol_random_train.json"),
                       ("Lipophilicity", "models/results_lipo_random_train.json")]:
        d = load_json(rel)
        print(f"{name}: RMSE={d['rmse']:.4f} MAE={d['mae']:.4f} "
              f"R2={d['r2']:.4f} n_train={d['n_train']} n_test={d['n_test']}")

    print("\n" + "=" * 70)
    print("1.2b 误差结构分析（报告 §4.2 图 3）")
    print("=" * 70)
    de = load_json("models/results_esol_random_train.json")
    err = np.array(de["predictions"]) - np.array(de["targets"])
    print(f"ESOL 残差: mean={err.mean():.3f} std={err.std():.3f} "
          f"MAE={np.abs(err).mean():.3f}")
    print(f"within +-0.5: {100*np.mean(np.abs(err)<=0.5):.1f}%  "
          f"within +-1.0: {100*np.mean(np.abs(err)<=1.0):.1f}%")
    print(f"max over-pred={err.max():.2f}  max under-pred={err.min():.2f}")

    print("\n" + "=" * 70)
    print("1.3 数据规模实验（报告 §4.3 表 3 图 4-5）")
    print("=" * 70)
    for name, rel in [("ESOL", "results/data_scale_results_esol.json"),
                       ("Lipophilicity", "results/data_scale_results_lipo.json")]:
        d = load_json(rel)
        for scale in ["20%", "50%", "80%", "100%"]:
            s = d[scale]
            n_train = s["runs"][0]["n_train"]
            seeds = [r["seed"] for r in s["runs"]]
            print(f"{name:13s} {scale:>4s}  n_train={n_train:5d}  "
                  f"RMSE={s['rmse_mean']:.4f} (std={s['rmse_std']:.4f})  "
                  f"R2={s['r2_mean']:.4f}  seeds={seeds}")

    print("\n" + "=" * 70)
    print("1.4 Chemprop vs Random Forest（报告 §4.4 表 4 图 6）")
    print("=" * 70)
    rf_files = {
        ("ESOL", "random"): "results/rf_esol_random_results.json",
        ("ESOL", "scaffold"): "results/rf_esol_scaffold_results.json",
        ("Lipophilicity", "random"): "results/rf_lipophilicity_random_results.json",
        ("Lipophilicity", "scaffold"): "results/rf_lipophilicity_scaffold_results.json",
    }
    cp_files = {
        ("ESOL", "random"): "models/results_esol_random_train.json",
        ("ESOL", "scaffold"): "models/results_esol_scaffold_train.json",
        ("Lipophilicity", "random"): "models/results_lipo_random_train.json",
        ("Lipophilicity", "scaffold"): "models/results_lipo_scaffold_train.json",
    }
    for ds in ["ESOL", "Lipophilicity"]:
        for split in ["random", "scaffold"]:
            rf = load_json(rf_files[(ds, split)])
            cp = load_json(cp_files[(ds, split)])
            print(f"{ds:13s} {split:8s} RF      "
                  f"RMSE={rf['rmse']:.4f} MAE={rf['mae']:.4f} R2={rf['r2']:.4f}")
            print(f"{ds:13s} {split:8s} Chemprop "
                  f"RMSE={cp['rmse']:.4f} MAE={cp['mae']:.4f} R2={cp['r2']:.4f}")

    print("\n--- 报告 §4.4 的百分比/特征重要性推导 ---")
    e_rf = load_json("results/rf_esol_random_results.json")["rmse"]
    e_cp = load_json("models/results_esol_random_train.json")["rmse"]
    l_rf = load_json("results/rf_lipophilicity_random_results.json")["rmse"]
    l_cp = load_json("models/results_lipo_random_train.json")["rmse"]
    print(f"ESOL  Chemprop vs RF RMSE 降幅 = {(e_rf - e_cp) / e_rf * 100:.1f}%")
    print(f"Lipo  Chemprop vs RF RMSE 降幅 = {(l_rf - l_cp) / l_rf * 100:.1f}%")
    fi = load_json("results/rf_esol_random_results.json")["feature_importance"]
    print("ESOL RF feature_importance:", {k: round(v, 4) for k, v in fi.items()})
    fi_l = load_json("results/rf_lipophilicity_random_results.json")["feature_importance"]
    print("Lipo RF feature_importance:", {k: round(v, 4) for k, v in fi_l.items()})

    print("\n--- 报告 §4.3 每分子边际效益推导 ---")
    es = load_json("results/data_scale_results_esol.json")
    d_lo = es["20%"]["rmse_mean"] - es["50%"]["rmse_mean"]
    n_lo = es["50%"]["runs"][0]["n_train"] - es["20%"]["runs"][0]["n_train"]
    d_hi = es["80%"]["rmse_mean"] - es["100%"]["rmse_mean"]
    n_hi = es["100%"]["runs"][0]["n_train"] - es["80%"]["runs"][0]["n_train"]
    print(f"20->50%: dRMSE={d_lo:.3f} over {n_lo} mols = {d_lo/n_lo:.2e}/mol")
    print(f"80->100%: dRMSE={d_hi:.3f} over {n_hi} mols = {d_hi/n_hi:.2e}/mol")
    print(f"marginal-value drop factor = {(d_lo/n_lo)/(d_hi/n_hi):.1f}x")

    print("\n" + "=" * 70)
    print("1.6 分子案例（报告 §4.6 图 7）")
    print("=" * 70)
    s_cases = {m["name"]: m for m in load_json("results/molecule_cases.json")}
    d_cases = {m["name"]: m for m in load_json("results/lipo_molecule_cases.json")}
    for name in ["Glucose", "Testosterone", "Tamoxifen", "Ibuprofen"]:
        s = s_cases.get(name, {})
        d = d_cases.get(name, {})
        print(f"{name:13s} predicted_logS={s.get('predicted_logS')}  "
              f"predicted_logD={d.get('predicted_logD')}")

    print("\n" + "=" * 70)
    print("5. 文档概述数字 vs 实际结果文件（报告 §5）")
    print("=" * 70)
    legacy = load_json("models/results_esol_train.json")
    print(f"旧文件 models/results_esol_train.json (非 random 命名口径): "
          f"RMSE={legacy['rmse']:.4f} R2={legacy['r2']:.4f}  "
          f"<- 与文档 0.584/0.928 接近，疑似遗留")
    cur = load_json("models/results_esol_random_train.json")
    print(f"当前 models/results_esol_random_train.json: "
          f"RMSE={cur['rmse']:.4f} R2={cur['r2']:.4f}  <- 报告采用")


if __name__ == "__main__":
    main()
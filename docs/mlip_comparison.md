# MLIP Comparison: Pure MLIP vs MENDEL + MLIP

## 設定

- **Reference data**: rMD17 ethanol, 100 conformers, revPBE-D3 DFT
- **Models**: MACE-OFF-small (float32, CPU) vs ANI-2x (torchani 2.8.0, CPU)
- **Scripts**:
  - `scripts/run_mlip_reference_benchmark.py` — 跑 single-point E+F
  - `scripts/compare_mace_ani2x.py` — 兩模型整體比較圖
  - `scripts/compare_pure_vs_mendel_mlip.py` — pure MLIP vs MENDEL 分解圖

安裝：

```bash
pip install -e ".[mlip]"     # MACE
pip install -e ".[ani2x]"    # ANI-2x (torchani>=2.2 + ase + torch>=2.0)
pip install -e ".[mlip-all]" # 兩個一起
```

> MACE 在 Apple Silicon 需要指定 `--device cpu`（MPS 不支援 float64）。

---

## 整體結果（純 MLIP）

| 指標 | MACE-OFF-small | ANI-2x |
|------|---------------|--------|
| Force MAE (eV/Å) | 0.374 | 0.258 |
| Force RMSE (eV/Å) | 0.443 | 0.305 |
| Energy MAE (eV) | 11.35 | 7.23 |
| Per-element RMSE: C | 0.479 | 0.341 |
| Per-element RMSE: H | 0.418 | 0.293 |
| Per-element RMSE: O | 0.513 | 0.308 |

ANI-2x 在所有指標上勝出約 31–36%。ANI-2x 以 CCSD(T)/CBS 有機小分子資料訓練，與 rMD17（revPBE-D3）有機小分子 conformer 性質接近；MACE-OFF-small 使用 SPICE（ωB97X-D3），float32 精度也有損失。

圖：`reports/figures/mace_vs_ani2x_ethanol.png`

---

## MENDEL 分解結果

MENDEL 識別 ethanol 的 functional group agents：
- **alcohol**（reactive）：alpha-C + O，在 ionic context 下預測為 nucleophile
- **hydroxyl H**（O–H bond）
- **alpha C–H**（reactive side）
- **methyl C–H**（spectator）
- **methyl C**（spectator）

| Group | MACE RMSE | ANI-2x RMSE | 相對 global |
|-------|-----------|-------------|-------------|
| alcohol C–O（reactive） | **0.954** | **0.601** | MACE **2.15×** global |
| hydroxyl H（O–H） | 0.771 | 0.482 | — |
| alpha C–H | 0.758 | 0.550 | — |
| methyl C–H（spectator） | 0.683 | 0.486 | — |
| methyl C（spectator） | 0.587 | 0.512 | — |

**關鍵發現**：兩個模型的 force 誤差都集中在 MENDEL 識別為 reactive 的 functional group。Reactive site 誤差是 global RMSE 的 ~2×。純 MLIP 的 global 單一數字掩蓋了這個事實。

圖：`reports/figures/pure_vs_mendel_mlip.png`

---

## 結論

| 問題 | 答案 |
|------|------|
| ANI-2x vs MACE-OFF-small，哪個好？ | ANI-2x，差距約 31% |
| 加入 functional group agent 比較好嗎？ | **診斷能力更好**，但目前不改善 force 精度 |
| MENDEL 現在的角色 | 事後分析工具（顯微鏡），非預測改善工具 |

MENDEL 目前讓你問「MLIP 在 nucleophile 上準嗎？」這個純 MLIP 無法回答的問題。但 MLIP 的 force 計算本身沒有因為 MENDEL 而改變。

---

## 下一步：讓 MENDEL 真正改善 MLIP

要讓 functional group as agent **實際改善預測精度**，需要以下其中一條路線：

### 路線 B（最快）：Reactive-site weighted fine-tuning

```
MENDEL 識別 reactive atoms
    → fine-tune MACE/ANI 時 reactive atoms 的 force loss weight × 3
    → 重新跑 benchmark，比較 alcohol C-O RMSE before/after
```

預期：reactive group 誤差下降，spectator group 不變。
這才是「functional group as agent 比較好」的實驗證明。

### 路線 A：Role feature injection

把 MENDEL 的 role 預測（nucleophile / electrophile / spectator）作為 node feature 注入 MLIP 的 message passing layer。

### 路線 C：Reaction-center training set curation

用 MENDEL reaction center 篩選 training set，讓 MLIP 多看 reactive conformer，少看 spectator-only 結構。

---

## 產生的檔案

| 檔案 | 內容 |
|------|------|
| `reports/bench_mace_small_ethanol.json` | MACE-OFF-small benchmark report |
| `reports/bench_ani2x_ethanol.json` | ANI-2x benchmark report |
| `reports/preds_mace_small_ethanol.json` | MACE per-structure predictions |
| `reports/preds_ani2x_ethanol.json` | ANI-2x per-structure predictions |
| `reports/fg_force_mace_ethanol.json` | MACE functional group force analysis |
| `reports/fg_force_ani2x_ethanol.json` | ANI-2x functional group force analysis |
| `reports/figures/mace_vs_ani2x_ethanol.png` | 整體比較圖（4 panels）|
| `reports/figures/pure_vs_mendel_mlip.png` | Pure MLIP vs MENDEL 分解圖 |

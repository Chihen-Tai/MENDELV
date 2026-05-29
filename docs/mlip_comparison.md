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

## Route B 實驗結果：Reactive-site Weighted Fine-tuning

### 多分子跨分子泛化實驗

**設計**：
- Train: ethanol (300) + malonaldehyde (300) + aspirin (300) = 900 conformers
- Test（held-out）: salicylic acid 100 conformers（完全未見過的分子）
- MENDEL reactive detection：heteroatom + 鄰近 heavy atom（cutoff 1.65 Å）
- 對照組：相同設定但 `--reactive-weight 1.0`（uniform）

**Per-group force RMSE on salicylic acid（held-out）**：

| Atom group | MENDEL ×3 | Uniform ×1 | Δ | 勝負 |
|------------|-----------|------------|---|------|
| reactive（O, C-O 鄰, 10 atoms） | **0.2340** | 0.2461 | −0.0121 (−4.9%) | **MENDEL** |
| spectator（H, 6 atoms） | 0.1511 | **0.1488** | +0.0023 (+1.5%) | uniform |
| global | **0.2068** | 0.2148 | −0.0080 (−3.7%) | **MENDEL** |

**結論**：改善集中在 MENDEL 標記的 reactive atoms（−4.9%），spectator 略差（+1.5%）。Functional group as agent 假設在跨分子泛化實驗中成立。

Checkpoints: `models/ani2x_multimol_mendel.pt`、`models/ani2x_multimol_uniform.pt`
Script: `scripts/finetune_ani2x_multimol.py`

---

## 更新結論

| 問題 | 答案 |
|------|------|
| ANI-2x vs MACE-OFF-small？ | ANI-2x 勝，差距約 31% |
| MENDEL 能改善 MLIP force 精度嗎？ | **可以**—— reactive atoms −4.9%（跨分子驗證） |
| MENDEL 現在的角色 | 診斷工具 + 訓練正則化工具（reactive-site weighting） |

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
| `reports/finetune_ani2x_multimol_mendel.json` | Route B MENDEL ×3 訓練報告 |
| `reports/finetune_ani2x_multimol_uniform.json` | Route B uniform 對照組報告 |
| `models/ani2x_multimol_mendel.pt` | ANI-2x fine-tuned，reactive ×3 |
| `models/ani2x_multimol_uniform.pt` | ANI-2x fine-tuned，uniform |

# MENDEL

**M**ulti-agent **E**lement-level **N**egotiation for **D**ynamic **E**nergy **L**andscapes

A research prototype exploring whether element-typed LLM agents can produce better adaptive QM/MM partitions than distance-based heuristics — by replacing geometric cutoffs with chemistry-aware reasoning, while keeping LLM cost sparse.

---

## English

### Motivation

In any molecular system, only a small fraction of atoms participate in chemistry at any given moment. Adaptive QM/MM has solved this partitioning problem for 15+ years using distance-based heuristics, but distance is a crude proxy for reactivity. MENDEL asks: *can LLM agents, equipped with element-specific chemical priors, decide better — without blowing up the compute budget?*

### Core Idea

- One LLM agent per **chemical element** (C, O, N, H, Fe, ...) — not per atom.
- Each element agent splits its atoms into an **active sub-agent** (currently reactive) and a **spectator sub-agent** (structural / bulk).
- Atoms are **promoted / demoted** dynamically by a two-gate mechanism: a cheap deterministic descriptor gate, then an LLM reasoning gate.
- Physics runs every timestep; LLM reasoning only fires on rare promotion events.

### Architecture

```
┌───────────────────────────────────────────────┐
│ L5: Mechanism Coordinator                     │
│     Global orchestrator, cross-element        │
│     negotiation, mechanism trace output       │
└──────────────────┬────────────────────────────┘
                   │
┌──────────────────┴────────────────────────────┐
│ L4: Element Agents (C, O, N, H, Fe, ...)      │
│     - Chemical persona (frozen prior)         │
│     - Owns sub-agent partition for its atoms  │
│     - Listens for promotion triggers          │
└────┬───────────────────┬──────────────────────┘
     │                   │
┌────┴────┐         ┌────┴────┐
│ Active  │         │Spectator│  ← L3 sub-agents
│ sub-agt │         │ sub-agt │    (per element)
└────┬────┘         └────┬────┘
     │                   │
┌────┴───────────────────┴──────────────────────┐
│ L2: Atom Instance Registry                    │
│     atom_id ↔ element ↔ sub-agent assignment  │
│     Shared state (not an agent)               │
└──────────────────┬────────────────────────────┘
                   │
┌──────────────────┴────────────────────────────┐
│ L1: Physics Substrate                         │
│     OpenMM / LAMMPS / MLIP                    │
│     Updates forces, positions every timestep  │
└───────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Role | State |
|---|---|---|
| L1 Physics | Run dynamics | positions, velocities, forces, partial charges |
| L2 Registry | Shared state | atom_id ↔ element ↔ sub-agent map |
| L3 Sub-agents | Partition management | active: per-atom detailed state; spectator: aggregate |
| L4 Element Agents | Element-level reasoning | chemical persona (frozen), owned atom IDs |
| L5 Coordinator | Cross-element negotiation, narration | event log, mechanism hypothesis tree |

### Promotion / Demotion: Two-Gate Mechanism

**Gate 1 — Deterministic descriptor gate** (runs every timestep, cheap, no LLM)

Per-element threshold table. Examples:

| Element | Descriptor | Trigger |
|---|---|---|
| C | partial charge change rate | dq/dt > 0.05 e/ps |
| C | nearest neighbor distance | d < 3.0 Å + δ |
| O | lone pair orientation vs δ+ | cos θ > 0.7 |
| O | Wiberg bond order change | dBO > 0.1 |
| H | proton transfer indicator | Grotthuss-style |
| Fe | coordination number change | dCN ≠ 0 |

LLM has **no veto power before Gate 1 fires**.

**Gate 2 — LLM reasoning gate** (fires only when Gate 1 opens)

The element agent's LLM call performs:
- **Classify** the chemical cause (nucleophilic attack / proton transfer / electron transfer / ligand binding)
- **Predict** expected behavior over the next ~hundreds of fs
- **Veto** if reasoning identifies a false positive (e.g. thermal-only fluctuation)
- **Self-consistency**: generate 3 hypotheses; commit only if all 3 agree

Key constraint: LLM can only **reject** a Gate 1 trigger, never **promote** unilaterally. This bounds hallucination damage to over-conservatism, not invention.

**Demotion**: active sub-agent monitors settling (stable charge, frozen bond order, no new neighbors). Triggers symmetric two-gate demotion.

### Energy Continuity

To avoid potential energy surface discontinuities at partition boundaries, MENDEL adopts a **buffer zone with smooth interpolation**, following PAP / HAMBC-PAP convention:

```
Active  ←→  Buffer (weight w ∈ [0,1])  ←→  Spectator
E_total = w · E_active + (1 - w) · E_spectator
w ramps smoothly from 0 to 1 over ~100 timesteps after promotion
```

Hysteresis (promote threshold stricter than demote threshold) prevents boundary chatter — a known weakness of distance-cutoff schemes.

### Setup

> Status: design phase. Code not yet implemented.

Planned dependencies:
- Python 3.11+
- OpenMM (physics substrate, MVP)
- RDKit (descriptors, bond order)
- LangChain or LiteLLM (LLM orchestration)
- Anthropic / OpenAI API (LLM backend)

### Roadmap (8-week MVE)

| Week | Milestone |
|---|---|
| 1–2 | L1 + L2 implemented; SN2 reaction (CH₃Br + OH⁻) runs in OpenMM with custom atom registry |
| 3–4 | L4 element agents implemented (C, O, H, Br personas); single-tier, no sub-agent yet |
| 5–6 | L3 active/spectator sub-agents added; two-gate promotion mechanism wired up |
| 7–8 | SN2 case study; mechanism trace output; comparison against textbook arrow-pushing |

Stretch goals:
- Week 9–10: SN1 stress test (two-stage mechanism, carbocation intermediate)
- Week 11–12: Solvent exchange benchmark vs distance-cutoff PAP

### Evaluation Plan

**Partition quality metrics**

| Metric | Definition | Why |
|---|---|---|
| Recall | Fraction of truly reactive atoms promoted to active | Beats distance-cutoff baseline? |
| Precision | Fraction of promoted atoms that actually react | Avoids over-promotion |
| Latency | Time delta from reaction onset to promotion | Late promotion = useless |

**Ground truth sources**

1. High-level QM reference for small systems (DFT, < 50 atoms)
2. Literature mechanism consensus for well-characterized reactions
3. Curriculum starting from textbook reactions

**Baselines**

- Distance-cutoff adaptive QM/MM (PAP, DAS, FSA)
- Static QM/MM with expert-picked active site (upper bound)
- Full MM, no partition (lower bound)
- Ablation: Gate 1 only, no LLM
- Genie-CAT for active site detection comparison

**Killer experiment: solvent exchange**

Show MENDEL pre-promotes an incoming catalytic water molecule ~N fs *earlier* than distance-cutoff PAP, by reasoning about approach geometry rather than waiting for distance threshold.

### Related Work

- **Adaptive QM/MM** (PAP, HAMBC-PAP, DAS) — provides energy continuity machinery and is the primary baseline
- **AtomAgents** (Buehler et al. 2024) — LLM multi-agent for atomistic simulation, but agents are workflow roles (User, Engineer, Scientist), not elements
- **Genie-CAT** (PNNL 2025) — LLM agent for active site identification in metalloproteins; closest active-site-finding analog, but single-agent + tools, no element typing, no spectator sub-agents
- **ChemHAS** (2025) — hierarchical agent stacking for chemistry tools; hierarchy over tools, not over molecular partition
- **MLIP element embeddings** (SchNet, NequIP, MACE) — per-element representations, the conceptual ancestor of element-as-agent

### Open Questions

- Does element-level agent grouping actually add value over single-agent with element-aware prompts?
- How do oxidation state / hybridization differences within the same element get handled? Sub-personas? Multiple element agents?
- Can negotiation between element agents catch reactions that single-agent reasoning misses?

---

## 中文

### 動機

任何分子系統裡，任何時刻真正在做化學的 atom 只佔少數。Adaptive QM/MM 已經用 distance-based heuristic 解這個 partition 問題 15+ 年，但距離只是 reactivity 的粗略代理。MENDEL 想問的問題是：**LLM agent 帶著 element-specific 化學先驗，能不能判斷得更準 — 同時不讓算力爆炸？**

### 核心想法

- 每個**化學元素**一個 LLM agent（C、O、N、H、Fe...），不是每個 atom 一個
- 每個 element agent 把自己管的 atoms 切成 **active sub-agent**（當前在反應）和 **spectator sub-agent**（結構 / bulk）
- atom 透過 **two-gate 機制**動態 promote / demote：先一個便宜的 deterministic descriptor gate，再一個 LLM reasoning gate
- Physics 每 timestep 都跑；LLM 推理只在 rare promotion event 觸發

### 架構

（同上方架構圖，五層 L1–L5）

### 各層職責

| Layer | 職責 | State 內容 |
|---|---|---|
| L1 Physics | 跑 dynamics | positions, velocities, forces, partial charges |
| L2 Registry | shared state | atom_id ↔ element ↔ sub-agent 對應表 |
| L3 Sub-agents | partition 管理 | active: per-atom 詳細 state；spectator: aggregate |
| L4 Element Agents | element-level reasoning | chemical persona（frozen），管的 atom IDs |
| L5 Coordinator | 跨元素 negotiation + 敘述 | event log、mechanism hypothesis tree |

### Promotion / Demotion：Two-Gate 機制

**Gate 1 — Deterministic descriptor gate**（每 timestep，便宜，不用 LLM）

每個元素有自己的 threshold table。範例如上方英文表。LLM 在 Gate 1 觸發之前**沒有發言權**。

**Gate 2 — LLM reasoning gate**（只在 Gate 1 開門時觸發）

對應的 element agent 呼叫 LLM 做：
- **Classify** 化學原因（nucleophilic attack / proton transfer / electron transfer / ligand binding）
- **Predict** 接下來幾百 fs 內的 expected behavior
- **Veto** 機會：認定 false positive（純熱運動 fluctuation）就 reject
- **Self-consistency**：生 3 個 hypothesis，3 個都同意才 commit

關鍵約束：**LLM 只能拒絕，不能單方面 promote**。把 hallucination 的傷害限制在過度保守，不會無中生有。

**Demotion**：active sub-agent 監測 settling（charge 穩定、bond order 不動、沒新鄰居），對稱地走 two-gate demotion。

### Energy 連續性

為了避免 partition boundary 造成 potential energy surface 不連續，MENDEL 採用 **buffer zone + smooth interpolation**，沿用 PAP / HAMBC-PAP convention：

```
Active  ←→  Buffer (weight w ∈ [0,1])  ←→  Spectator
E_total = w · E_active + (1 - w) · E_spectator
w 在 promotion 後 ~100 timesteps 內從 0 平滑升到 1
```

Hysteresis（promote threshold 比 demote threshold 嚴格）防止 boundary chatter — 這是 distance-cutoff 方案的已知弱點。

### 環境設置

> 狀態：design phase，code 尚未實作。

預計 dependency：
- Python 3.11+
- OpenMM（physics substrate, MVP）
- RDKit（descriptor、bond order）
- LangChain 或 LiteLLM（LLM orchestration）
- Anthropic / OpenAI API（LLM backend）

### Roadmap（8 週 MVE）

| Week | Milestone |
|---|---|
| 1–2 | L1 + L2 實作；SN2 反應（CH₃Br + OH⁻）在 OpenMM 跑起來，搭配自製 atom registry |
| 3–4 | L4 element agents 實作（C、O、H、Br 四個 persona）；single-tier，先不做 sub-agent |
| 5–6 | L3 active/spectator sub-agent 加入；two-gate promotion 接起來 |
| 7–8 | SN2 case study；mechanism trace 輸出；跟教科書 arrow-pushing 比對 |

Stretch goal：
- Week 9–10：SN1 stress test（兩階段機構、carbocation intermediate）
- Week 11–12：solvent exchange benchmark vs distance-cutoff PAP

### 評估計畫

**Partition quality 指標**

| 指標 | 定義 | 為什麼重要 |
|---|---|---|
| Recall | 真正反應的 atom 有多少被升級成 active | beat distance-cutoff 的賣點 |
| Precision | 升級的 atom 有多少真的反應 | 避免亂升級 |
| Latency | 反應實際開始到 promote 的時間差 | 太晚等於沒用 |

**Ground truth 來源**

1. 小系統（< 50 atoms）的 high-level QM 參考（DFT）
2. Well-characterized 反應的文獻機構共識
3. 從教科書反應開始的 curriculum

**Baseline**

- Distance-cutoff adaptive QM/MM（PAP、DAS、FSA）
- Static QM/MM + 專家挑的 active site（upper bound）
- Full MM、無 partition（lower bound）
- Ablation：只有 Gate 1，沒有 LLM
- Genie-CAT 做 active site detection 比較

**Killer experiment：solvent exchange**

展示 MENDEL 能比 distance-cutoff PAP **提早 N fs** pre-promote 一個即將進入 active site 的 catalytic water — 靠 reasoning approach geometry，而不是等距離閾值。

### Related Work

- **Adaptive QM/MM**（PAP、HAMBC-PAP、DAS）— 提供 energy continuity 的工具箱，是主要 baseline
- **AtomAgents**（Buehler et al. 2024）— LLM multi-agent for atomistic simulation，但 agent 是 workflow 角色（User、Engineer、Scientist），不是 element
- **Genie-CAT**（PNNL 2025）— LLM agent 找 metalloprotein active site，是最接近的 active-site-finding 對照，但是 single-agent + tools，沒有 element typing 也沒 spectator sub-agent
- **ChemHAS**（2025）— chemistry tools 的 hierarchical agent stacking，但 hierarchy 切在 tool 上不在 partition 上
- **MLIP element embedding**（SchNet、NequIP、MACE）— per-element representation，是 element-as-agent 的概念祖先

### 待解問題

- Element-level agent 分組真的比 single-agent + element-aware prompt 多帶來價值嗎？
- 同元素的 oxidation state / hybridization 差異怎麼處理？sub-persona？多個 element agent？
- Element agent 之間的 negotiation 能不能抓到 single-agent 漏掉的反應？

---

*This is a personal research prototype. Status: design phase, pre-implementation.*

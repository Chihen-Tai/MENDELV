# Phase 6: Negotiation Layer

`mendel/negotiator.py` — Phase 6

Coordinates raw per-group role predictions (Phase 5) into a globally consistent
reaction-level interpretation.  Fully deterministic, no ML, no MLIP.

---

## Purpose

Phase 5 gives each functional-group *agent* a local role prediction in isolation.
Two alpha-carbon groups in the same acetone molecule both receive `reactive_nucleophile`
from Phase 5 — chemically redundant for a single aldol step.  Phase 6 is the global
consistency layer that resolves such conflicts and assigns a single coherent
mechanism-level interpretation.

The negotiation layer does **not** change the fundamental agent model.  It reads Phase 5
predictions, applies chemistry-aware heuristic rules, and produces one final role per group.

---

## Difference Between Raw Predictions and Negotiated Assignments

| Property | Phase 5 `RolePrediction` | Phase 6 `NegotiatedRoleAssignment` |
|----------|--------------------------|-------------------------------------|
| Role | `predicted_role` — local, independent | `final_role` — globally consistent |
| Confidence | `confidence` — from descriptor scores | `final_confidence` — may change |
| Scope | Per group, ignores peers | Reaction-level, aware of all peers |
| Subrole | — | Optional fine-grained label |
| Reaction center | — | `is_reaction_center` flag |
| Reason | Threshold rule that fired | Updated to include negotiation rationale |

Raw values are preserved in `raw_role` and `raw_confidence`.  Input `RolePrediction`
and `FunctionalGroup` objects are never mutated.

---

## Mechanism Hints

`infer_mechanism_hint` returns one of six string labels, evaluated in priority order:

| Priority | Hint | Conditions |
|----------|------|------------|
| 1 | `radical_bromination_like` | `context == radical` |
| 2 | `diels_alder_like` | `context == pericyclic` |
| 3 | `aldol_like` | ionic AND carbonyl AND alpha_carbon groups present |
| 4 | `sn2_or_e2_like` | ionic AND halide or leaving_group prediction present |
| 5 | `ionic_addition_like` | ionic AND nucleophile AND electrophile both predicted |
| 6 | `unknown` | none of the above |

---

## Mechanism-Specific Negotiation Rules

### A. `sn2_or_e2_like`

- Halide groups confirmed or converted to `leaving_group`.
  If Phase 5 predicted `reactive_electrophile` but `leaving_group_score ≥ 0.50`,
  the final role is converted.
- Halide group marked as reaction center; subrole `"leaving_group_site"`.
- `missing_nucleophile` warning if no `reactive_nucleophile` is detected.
- `coarse_group_granularity` info warning: the C–X bond is one group in v0.1.

### B. `aldol_like`

- Highest-confidence alpha-carbon nucleophile selected as **primary donor**.
- Highest-confidence carbonyl electrophile selected as **primary acceptor**.
- Donor: subrole `"aldol_donor_alpha_carbon"`, `is_reaction_center = True`.
- Acceptor: subrole `"aldol_acceptor_carbonyl"`, `is_reaction_center = True`.
- Remaining alpha-carbon nucleophiles downgraded to `spectator` with subrole
  `"secondary_alpha_candidate"` (when `allow_role_downgrade_to_spectator` is True).
- `heuristic_donor_acceptor_assignment` info warning always added.

### C. `diels_alder_like`

Phase 5 assigns all pi-system groups `reactive_nucleophile` in pericyclic context,
which is chemically incoherent.  Phase 6 fixes this:

- Pi groups split by molecule.  Molecule with **fewer** pi groups = dienophile candidate.
- Dienophile representative reassigned to `reactive_electrophile`; subrole `"dienophile_like"`.
- Diene-side groups retain `reactive_nucleophile`; subrole `"diene_like"`.
- Both sets marked `is_reaction_center = True`.
- `missing_pericyclic_partner` warning if fewer than two pi groups are detected.

Flat roles (`reactive_nucleophile` / `reactive_electrophile`) are used — no new `pericyclic_partner`
Role enum value is introduced.

### D. `radical_bromination_like`

- Benzylic-site groups promoted to `reactive_radical` if not already.
- Benzylic sites marked `is_reaction_center = True`.
- `unsupported_radical_source` info warning: Br₂/AIBN not representable in v0.1.

### E. `ionic_addition_like`

- Highest-confidence nucleophile → reaction center; subrole `"ionic_nucleophile_candidate"`.
- Highest-confidence electrophile → reaction center; subrole `"ionic_electrophile_candidate"`.
- `missing_nucleophile` / `missing_electrophile` warnings if either is absent.

### F. `unknown`

- Raw roles preserved unchanged.
- `unknown_mechanism` info warning added.
- Groups with high confidence non-spectator roles flagged as possible reaction centers.

---

## Reaction Center Inference

`infer_reaction_center_atoms` collects `AtomRef` objects from every group whose
`is_reaction_center` flag is True.  Atom refs are deduplicated by
`(molecule_index, atom_index)` in stable input order.

---

## Phase 8.9: MLP-Aware Negotiation

Phase 8.9 adds an explicit experimental mode:

```python
from mendel.negotiator import NegotiatorConfig, negotiate_predictions

result = negotiate_predictions(
    parsed_reaction,
    groups,
    mlp_role_predictions,
    config=NegotiatorConfig(mode="mlp_aware"),
)
```

The default remains `mode="rule_based"`, so existing rule-based behavior and
tests are unchanged.

MLP-aware negotiation preserves the existing MLP-negotiated role adjustments,
then applies confidence- and mechanism-aware reaction-center selection. This is
needed because group-level role accuracy and reaction-center F1 measure different
things: a functional-group agent can have the correct role while its atom-level
center is incomplete or over-broad.

Key Phase 8.9 rules:

- High-confidence spectators are excluded from reaction centers.
- Control/no-reaction mechanisms suppress centers when spectator confidence dominates.
- SN2/E2 centers focus on the leaving-group site and represented attached carbon.
- Carbonyl addition centers focus on carbonyl atoms, not spectator alpha carbons.
- Diels-Alder centers include alkene pi partners and exclude EWG substituents.
- Nitroalkane deprotonation marks the alpha carbon as the nitronate-like center.
- Assignment metadata preserves predictor provenance and `center_selection_reason`.

`rule_based_negotiated` remains the conservative default unless benchmark evidence
shows the MLP-aware mode also matches or beats reaction-center F1.

---

## How Aldol Disambiguation Works

Given two acetone molecules `CC(=O)C.CC(=O)C`:

1. Phase 5 produces four alpha-carbon predictions (`reactive_nucleophile`, acidity `0.55`)
   and two carbonyl predictions (`reactive_electrophile`, electrophilicity `0.70`).
2. Phase 6 selects one alpha-carbon as primary donor and one carbonyl as primary acceptor.
   Ties in confidence are broken by list order (deterministic).
3. Remaining alpha-carbons downgraded to `spectator`.
4. `heuristic_donor_acceptor_assignment` warning documents the limitation.

Without atom-mapped reaction SMILES the choice between the two acetone molecules is
arbitrary; it becomes non-heuristic in a later phase when atom mapping is available.

---

## How Diels-Alder Flat Roles Are Handled

Phase 5 assigns all alkene groups `reactive_nucleophile` in pericyclic context because
`nucleophilicity_score (0.35) ≥ electrophilicity_score (0.25)` for alkenes.  Butadiene
and ethylene both get the same role — chemically wrong.

Phase 6 groups pi groups by molecule and designates the molecule with the fewest pi groups
as the dienophile.  The representative group is flipped to `reactive_electrophile`.  This
gives a chemically sensible diene/dienophile split without learned negotiation.

---

## How SN2/E2 Limitations Are Handled

In v0.1 the alkyl halide C–X bond is one functional group.  The electrophilic carbon and
the leaving halide share one `group_id`.  Phase 6:

- Marks the group as `leaving_group`.
- Adds subrole `"leaving_group_site"`.
- Adds `coarse_group_granularity` info warning.
- Adds `missing_nucleophile` warning if the nucleophile (e.g. `[OH-]`) was not detected
  (common in v0.1: no C–O bond means alcohol SMARTS does not match `[OH-]`).

---

## Public API

```python
from mendel.negotiator import (
    NegotiationWarning,
    NegotiatedRoleAssignment,
    NegotiationResult,
    NegotiatorConfig,
    RuleBasedNegotiator,
    negotiate_predictions,
    run_full_rule_pipeline,
    summarize_negotiation_result,
    get_final_role_counts,
    get_reaction_center_group_ids,
)
```

### One-call entry point

```python
from mendel.negotiator import run_full_rule_pipeline
from mendel.types import ReactionContext

result = run_full_rule_pipeline(
    "CC(=O)C.CC(=O)C>>CC(=O)CC(O)(C)C",
    context=ReactionContext.ionic,
)
print(result.mechanism_hint)  # "aldol_like"
for a in result.assignments:
    print(a.group_id, a.final_role.value, a.subrole)
```

### Warning codes

| Code | Severity | Meaning |
|------|----------|---------|
| `missing_nucleophile` | warning | No nucleophile detected |
| `missing_electrophile` | warning | No electrophile detected |
| `missing_leaving_group` | warning | No leaving group detected |
| `missing_radical_center` | warning | No radical center candidate found |
| `missing_pericyclic_partner` | warning | Fewer than 2 pi partners |
| `heuristic_donor_acceptor_assignment` | info | Aldol assignment is heuristic |
| `unsupported_radical_source` | info | Br₂/AIBN not representable in v0.1 |
| `coarse_group_granularity` | info | C–X bond is one group |
| `unknown_mechanism` | info | No mechanism rule matched |

---

## Known Limitations

**One role per group** — Each `FunctionalGroup` receives exactly one final role.
The electrophilic carbon and leaving halide of an alkyl halide share one `group_id`.

**No atom-mapped reaction-difference inference** — Aldol donor/acceptor and
Diels-Alder diene/dienophile assignments are heuristic.  Exact identification requires
atom-mapped SMILES (later phase).

**No learned negotiation** — All rules are hand-coded heuristics.

**No TS/barrier/PES prediction** — Role assignment only.  Activation energies,
transition states, and 3D geometry belong to the MLIP phase.

**No MLIP usage** — No MACE, ASE, or DFT data is used.

**Radical sources not representable** — Br₂, AIBN, and similar initiators have no
SMARTS match in Phase 2.

---

## Freeze Status

The negotiation layer is implemented and included in the Phase 0-6 pre-training freeze.
It is the final rule-based stage before optional Phase 7 MLP role predictor training.

For freeze validation:

- `import mendel` must work without `torch`.
- `run_full_rule_pipeline(smiles, context)` must remain usable without `torch`.
- Phase 0-6 tests should pass.
- No training scripts should run.

## What Phase 7 Should Implement Later

- **MLP role predictor** (`mendel/mlp.py`): train a model on labeled reaction data.
- **Atom-mapped disambiguation**: use reaction atom mapping to pinpoint the exact
  alpha-carbon donor in an aldol reaction.
- **Learned negotiation**: train a negotiator on (raw_predictions, true_roles) pairs.
- **Benchmarking module**: systematic evaluation against the five benchmark reactions
  in `data/reactions.json`.

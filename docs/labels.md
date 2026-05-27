# Module 4: Labeled Data Schema

`mendel/labels.py` — Phase 4

Defines the labeled reaction dataset format used for training and evaluation.
Provides load/save/validate/summarise utilities and a training-row flattener.

---

## Data Classes

### `LabeledGroupRole`

Ground-truth role label for one functional group.

| Field | Type | Description |
|-------|------|-------------|
| `group_id` | `str` | Matches the ID produced by `identifier.py` (e.g. `mol0_halide_0`) |
| `molecule_index` | `int` | 0-based reactant index |
| `group_type` | `FunctionalGroupType` | Functional group category enum |
| `atom_indices` | `list[int]` | 0-based atom indices within the molecule |
| `role` | `Role` | Ground-truth role |
| `confidence` | `str` | `'manual'` for human labels; float string otherwise |
| `notes` | `str \| None` | Optional annotation |

### `LabeledReaction`

A fully labeled reaction record.

| Field | Type | Description |
|-------|------|-------------|
| `reaction_id` | `str` | Stable human-readable ID |
| `reaction_smiles` | `str` | Atom-mapped reaction SMILES (`reactants>>products`) |
| `context` | `ReactionContext` | Broad mechanistic category |
| `mechanism_type` | `str` | Fine-grained label (e.g. `'SN2'`, `'Diels-Alder'`) |
| `split` | `str` | `'train'`, `'val'`, or `'test'` |
| `group_roles` | `list[LabeledGroupRole]` | Role labels for all relevant groups |
| `reaction_center_atoms` | `list[int]` | Atom-map numbers of bond-changing atoms |
| `metadata` | `dict` | Arbitrary key/value pairs |

---

## Public API

```python
from mendel.labels import (
    LabelValidationError,
    LabeledGroupRole,
    LabeledReaction,
    load_labeled_reactions,    # (path) → list[LabeledReaction]
    save_labeled_reactions,    # (list[LabeledReaction], path) → None
    validate_labeled_reaction, # (LabeledReaction) → bool | raises LabelValidationError
    validate_labeled_dataset,  # (list[LabeledReaction]) → bool | raises LabelValidationError
    summarize_labeled_dataset, # (list[LabeledReaction]) → dict
    labels_to_training_rows,   # (list[LabeledReaction]) → list[dict]
)
```

---

## JSON File Format

All data files use a top-level `reactions` wrapper:

```json
{
  "reactions": [
    {
      "reaction_id": "sn2_methyl_bromide_oh",
      "reaction_smiles": "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
      "context": "ionic",
      "mechanism_type": "SN2",
      "split": "train",
      "group_roles": [
        {
          "group_id": "mol0_halide_0",
          "molecule_index": 0,
          "group_type": "halide",
          "atom_indices": [0, 1],
          "role": "leaving_group",
          "confidence": "manual",
          "notes": null
        }
      ],
      "reaction_center_atoms": [1, 2, 3],
      "metadata": {}
    }
  ]
}
```

See `data/label_schema.example.json` for the full field reference.

---

## Data Files

| File | Contents |
|------|----------|
| `data/reactions.json` | 5 benchmark reactions + 3 extended examples (train/val/test) |
| `data/reactions.minimal.json` | 2 simple reactions for fast unit tests |
| `data/label_schema.example.json` | Annotated schema reference |

---

## Known Limitations

- `[OH-]`, `[C-]#N`, and Br• radical carriers are not detected as functional groups by `identifier.py`; their roles are recorded in `metadata` notes rather than `group_roles`.
- `radical_bromination_methane` has zero `group_roles` entries because neither CH4 nor Br2 matches any SMARTS pattern.
- Atom-map numbers in `reaction_center_atoms` must be manually verified against the reaction SMILES.
- Labels are heuristic starting points, not curated from experimental databases.

---

## Freeze Status

The labeled data schema is part of the Phase 0-6 pre-training freeze. The existing labels
support rule-based smoke tests and provide the starting point for optional Phase 7 MLP role
predictor training.

Do not run `scripts/train_mlp.py` or `tests/test_mlp.py` as part of Phase 0-6 validation.

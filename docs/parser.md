# Module 0: Reaction SMILES Parser

`mendel/parser.py` — Phase 1

Converts a reaction SMILES string into structured Python objects.
**Functional group identification is out of scope here** — that happens in Phase 2 (`identifier.py`).

---

## Input Format

```
reactants>>products
```

- `>>` separates reactants from products.
- `.` separates individual molecules within each side.
- Atom-map numbers (`:N` notation) are extracted when present but not required.

---

## Examples

**SN2 — ionic, charged species:**

```python
from mendel.parser import parse_reaction_smiles, get_reaction_summary
from mendel.types import ReactionContext

rxn = parse_reaction_smiles("CBr.[OH-]>>CO.[Br-]", context=ReactionContext.ionic)
print(rxn.total_charge_reactants)   # -1
print(get_reaction_summary(rxn))
# {'n_reactants': 2, 'n_products': 2, 'total_charge_reactants': -1,
#  'total_charge_products': -1, 'has_atom_mapping': False, 'has_radicals': False}
```

**Diels-Alder — pericyclic:**

```python
rxn = parse_reaction_smiles("C=CC=C.C=C>>C1CCC=CC1", context="pericyclic")
print(len(rxn.reactants))   # 2  (butadiene + ethylene)
print(len(rxn.products))    # 1  (cyclohexene)
```

**Atom-mapped SN2:**

```python
rxn = parse_reaction_smiles(
    "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]",
    context="ionic",
)
print(rxn.has_atom_mapping)            # True
print(rxn.reactants[0].atom_map_nums) # {0: 1, 1: 2}
```

---

## Public API

| Symbol | Kind | Description |
|--------|------|-------------|
| `ParsedMolecule` | dataclass | One molecule: charge, radical flag, atom-map dict |
| `ParsedReaction` | dataclass | Full reaction with both sides and charge totals |
| `ReactionParseError` | exception | Raised on any invalid input |
| `parse_reaction_smiles()` | function | Main entry point |
| `parse_reaction_record()` | function | Wraps a `ReactionRecord` from `mendel.types` |
| `validate_reaction_smiles()` | function | Non-raising boolean check |
| `get_reaction_summary()` | function | Compact dict for debugging |

---

## Limitations

- Splitting by `.` treats disconnected-fragment SMILES (e.g. `[Na+].[Cl-]`) as two molecules. This is intentional for reaction SMILES.
- The middle reagents field (`reactants>reagents>products`) is not supported. Pass reagents as reactants if needed.
- `.smiles` on each `ParsedMolecule` is RDKit-canonicalised and may differ from the input string.

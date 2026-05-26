# Module 1: Functional Group Identifier

`mendel/identifier.py` — Phase 2

Identifies functional groups in molecules using RDKit SMARTS substructure matching.
Returns `FunctionalGroup` objects (from `mendel.types`) with atom-level references.

---

## Primary vs Contextual Groups

**Primary groups** (13 types) are detected by SMARTS in a single pass:
carboxylic_acid, ester, amide, carbonyl, phenol, alcohol, ether, nitro, nitrile, halide, amine, alkene, alkyne.

Aromatic rings are detected separately via `mol.GetRingInfo().AtomRings()`.

**Contextual groups** (2 types) are detected in a second pass, after all primary groups:
- `alpha_carbon` — sp3 C–H adjacent to an EWG (C=O, C≡N, NO₂)
- `benzylic_site` — sp3 C–H adjacent to an aromatic ring

Disable with `include_contextual=False`.

---

## Priority Ordering and Duplicate Handling

SMARTS patterns are applied in priority order (lower number = higher priority):

| Priority | Group |
|----------|-------|
| 0 | aromatic |
| 1 | carboxylic_acid |
| 2 | ester |
| 3 | amide |
| 4 | carbonyl |
| 5 | phenol |
| 6 | alcohol |
| 7 | ether |
| 8–13 | nitro, nitrile, halide, amine, alkene, alkyne |
| 14–15 | alpha_carbon, benzylic_site |

Mutual exclusivity is achieved by two mechanisms:
1. Restrictive SMARTS (e.g., `carbonyl` uses `!$(C(=O)[OX2])` to exclude acids and esters).
2. Anchor-based deduplication: once an anchor atom is claimed by a higher-priority group, lower-priority patterns cannot claim the same anchor.

---

## Public API

```python
identify_functional_groups_in_mol(
    mol: Chem.Mol,
    molecule_index: int,
    include_contextual: bool = True,
) -> list[FunctionalGroup]

identify_functional_groups(
    parsed_reaction: ParsedReaction,
    include_products: bool = False,
    include_contextual: bool = True,
) -> list[FunctionalGroup]

get_group_summary(groups: list[FunctionalGroup]) -> dict[str, int]
has_group_type(groups, group_type: FunctionalGroupType | str) -> bool
validate_smarts_patterns() -> dict[str, bool]
```

---

## Examples

```python
from mendel.parser import parse_reaction_smiles
from mendel.identifier import identify_functional_groups, get_group_summary
from mendel.types import ReactionContext

rxn = parse_reaction_smiles("CC(=O)C.CBr>>CC(=O)CO", context=ReactionContext.ionic)
groups = identify_functional_groups(rxn)
print(get_group_summary(groups))
# {'carbonyl': 1, 'alpha_carbon': 2, 'halide': 1}

# Include products
groups_all = identify_functional_groups(rxn, include_products=True)

# Single molecule
from rdkit import Chem
mol = Chem.MolFromSmiles("CC(=O)O")
from mendel.identifier import identify_functional_groups_in_mol
groups = identify_functional_groups_in_mol(mol, molecule_index=0)
# → carboxylic_acid + alpha_carbon
```

---

## Metadata Fields

Each `FunctionalGroup.metadata` includes:

| Key | Values |
|-----|--------|
| `source` | `"smarts"`, `"ring_detection"`, `"contextual"` |
| `molecule_role` | `"reactant"`, `"product"` (when called via `identify_functional_groups`) |
| `atom_indices` | list of matched atom indices |
| `priority` | integer (see priority table above) |

---

## Known Limitations

- **Aryl halides** are out of scope (halide pattern requires sp3 C).
- **Ionic nucleophiles** like `[OH-]` match `alcohol` because RDKit perceives the O–H bond; this is intentional for SN2 reactant classification.
- **alpha_carbon** does not distinguish primary/secondary/tertiary.
- **Aromatic detection** covers rings where every atom is marked aromatic by RDKit; some heterocycles may need review.
- **acid_chloride, epoxide, anhydride, allylic_site** are not yet implemented.

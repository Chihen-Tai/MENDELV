"""Phase 2: Functional group identifier using RDKit SMARTS matching."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdchem

from mendel.parser import ParsedReaction
from mendel.types import AtomRef, FunctionalGroup, FunctionalGroupType

FUNCTIONAL_GROUP_SMARTS: dict[FunctionalGroupType, str] = {
    # Carbonyl family — specific before general
    FunctionalGroupType.carboxylic_acid: "[CX3](=O)[OX2H]",
    FunctionalGroupType.ester: "[CX3](=O)[OX2][CX4]",
    FunctionalGroupType.amide: "[CX3](=O)[NX3]",
    # Residual carbonyl: ketone/aldehyde only
    # Note: restrictive pattern prevents double-counting with acid/ester/amide
    FunctionalGroupType.carbonyl: "[CX3;!$(C(=O)[OX2]);!$(C(=O)[NX3])]=[OX1]",
    # Oxygen groups
    FunctionalGroupType.phenol: "c[OX2H]",
    FunctionalGroupType.alcohol: "[CX4][OX2H]",
    FunctionalGroupType.ether: "[CX4][OX2][CX4]",
    # Nitrogen / heteroatom
    FunctionalGroupType.nitro: "[$([NX3](=O)=O),$([N+](=O)[O-])]",
    FunctionalGroupType.nitrile: "[CX2]#[NX1]",
    FunctionalGroupType.halide: "[CX4][F,Cl,Br,I]",
    FunctionalGroupType.amine: "[NX3;!$(NC=O);!$([NX3+])]",
    # Phase 12 heteroatom groups (distinct anchors → no clash with nitrile/amine).
    # Isocyanide anchors on the terminal carbanion-like carbon (R-N#C:).
    FunctionalGroupType.isocyanide: "[CX1-]#[NX2+]",
    # Imine / Schiff base C=N (anchors on the electrophilic carbon).
    FunctionalGroupType.imine: "[CX3]=[NX2]",
    # Organic azide R-N3 (anchors on the N bonded to the rest of the molecule).
    FunctionalGroupType.azide: "[NX2]=[NX2+]=[NX1-]",
    # Unsaturated carbon
    FunctionalGroupType.alkene: "[CX3]=[CX3]",
    FunctionalGroupType.alkyne: "[CX2]#[CX2]",
    # Contextual groups (second pass)
    FunctionalGroupType.alpha_carbon: (
        "[CX4;H1,H2,H3]"
        "[$([CX3]=O),$([CX2]#[NX1]),$([NX3](=O)=O),$([N+](=O)[O-])]"
    ),
    FunctionalGroupType.benzylic_site: "[CX4;H1,H2,H3][c]",
}

# Priority: lower integer = higher priority = matched first
_GROUP_PRIORITY: dict[FunctionalGroupType, int] = {
    FunctionalGroupType.aromatic: 0,
    FunctionalGroupType.carboxylic_acid: 1,
    FunctionalGroupType.ester: 2,
    FunctionalGroupType.amide: 3,
    FunctionalGroupType.carbonyl: 4,
    FunctionalGroupType.phenol: 5,
    FunctionalGroupType.alcohol: 6,
    FunctionalGroupType.ether: 7,
    FunctionalGroupType.nitro: 8,
    FunctionalGroupType.nitrile: 9,
    FunctionalGroupType.halide: 10,
    FunctionalGroupType.amine: 11,
    FunctionalGroupType.isocyanide: 16,
    FunctionalGroupType.imine: 17,
    FunctionalGroupType.azide: 18,
    FunctionalGroupType.alkene: 12,
    FunctionalGroupType.alkyne: 13,
    FunctionalGroupType.alpha_carbon: 14,
    FunctionalGroupType.benzylic_site: 15,
}

_PRIMARY_ORDER: list[FunctionalGroupType] = [
    FunctionalGroupType.carboxylic_acid,
    FunctionalGroupType.ester,
    FunctionalGroupType.amide,
    FunctionalGroupType.carbonyl,
    FunctionalGroupType.phenol,
    FunctionalGroupType.alcohol,
    FunctionalGroupType.ether,
    FunctionalGroupType.azide,
    FunctionalGroupType.nitro,
    FunctionalGroupType.isocyanide,
    FunctionalGroupType.nitrile,
    FunctionalGroupType.halide,
    FunctionalGroupType.imine,
    FunctionalGroupType.amine,
    FunctionalGroupType.alkene,
    FunctionalGroupType.alkyne,
]

_CONTEXTUAL_ORDER: list[FunctionalGroupType] = [
    FunctionalGroupType.alpha_carbon,
    FunctionalGroupType.benzylic_site,
]

_COMPILED: dict[FunctionalGroupType, rdchem.Mol] = {}

# Water is not caught by [CX4][OX2H] (requires carbon). Detect it separately
# and represent as alcohol — semantically approximate but gives the partner
# context features a non-zero nucleophilicity signal for hydrolysis reactions.
_WATER_SMARTS: str = "[OH2]"
_WATER_QUERY: rdchem.Mol | None = None


def _get_water_query() -> rdchem.Mol:
    global _WATER_QUERY
    if _WATER_QUERY is None:
        _WATER_QUERY = Chem.MolFromSmarts(_WATER_SMARTS)
    return _WATER_QUERY


def _detect_water_groups(
    mol: rdchem.Mol,
    molecule_index: int,
    seen_anchors: set[tuple[FunctionalGroupType, int]],
    counter: dict[FunctionalGroupType, int],
    molecule_role: str | None,
) -> list[FunctionalGroup]:
    groups: list[FunctionalGroup] = []
    for match in mol.GetSubstructMatches(_get_water_query()):
        anchor = match[0]
        key = (FunctionalGroupType.alcohol, anchor)
        if key in seen_anchors:
            continue
        seen_anchors.add(key)
        count = counter.get(FunctionalGroupType.alcohol, 0)
        counter[FunctionalGroupType.alcohol] = count + 1
        meta: dict = {
            "source": "water_detection",
            "anchor_atom_index": anchor,
            "atom_indices": list(match),
            "priority": _GROUP_PRIORITY[FunctionalGroupType.alcohol],
        }
        if molecule_role is not None:
            meta["molecule_role"] = molecule_role
        groups.append(
            FunctionalGroup(
                group_id=f"mol{molecule_index}_{FunctionalGroupType.alcohol.value}_{count}",
                group_type=FunctionalGroupType.alcohol,
                atom_refs=[
                    AtomRef(
                        molecule_index=molecule_index,
                        atom_index=anchor,
                        atom_map_num=mol.GetAtomWithIdx(anchor).GetAtomMapNum() or None,
                    )
                ],
                smarts=_WATER_SMARTS,
                metadata=meta,
            )
        )
    return groups


def _get_query(group_type: FunctionalGroupType) -> rdchem.Mol:
    if group_type not in _COMPILED:
        smarts = FUNCTIONAL_GROUP_SMARTS[group_type]
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            raise ValueError(f"Invalid SMARTS for {group_type}: {smarts}")
        _COMPILED[group_type] = q
    return _COMPILED[group_type]


def _detect_aromatic_groups(
    mol: rdchem.Mol,
    molecule_index: int,
    counter: dict[FunctionalGroupType, int],
    molecule_role: str | None,
) -> list[FunctionalGroup]:
    """Detect aromatic rings via RDKit ring info (robust for fused systems)."""
    groups: list[FunctionalGroup] = []
    ring_info = mol.GetRingInfo()
    seen_ring_sets: list[frozenset[int]] = []

    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            ring_set = frozenset(ring)
            if ring_set in seen_ring_sets:
                continue
            seen_ring_sets.append(ring_set)

            count = counter.get(FunctionalGroupType.aromatic, 0)
            counter[FunctionalGroupType.aromatic] = count + 1
            anchor = min(ring)
            atom_indices = sorted(ring)

            # Flag pyridine-like rings so the negotiator can recognise Minisci-type
            # radical additions to heteroaromatic nitrogen systems.
            heteroaromatic_n = any(
                mol.GetAtomWithIdx(idx).GetSymbol() == "N"
                and mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            )

            meta: dict = {
                "source": "ring_detection",
                "ring_size": len(ring),
                "anchor_atom_index": anchor,
                "atom_indices": atom_indices,
                "priority": _GROUP_PRIORITY[FunctionalGroupType.aromatic],
                "heteroaromatic_n": heteroaromatic_n,
            }
            if molecule_role is not None:
                meta["molecule_role"] = molecule_role

            groups.append(
                FunctionalGroup(
                    group_id=f"mol{molecule_index}_{FunctionalGroupType.aromatic.value}_{count}",
                    group_type=FunctionalGroupType.aromatic,
                    atom_refs=[
                        AtomRef(
                            molecule_index=molecule_index,
                            atom_index=idx,
                            atom_map_num=mol.GetAtomWithIdx(idx).GetAtomMapNum() or None,
                        )
                        for idx in atom_indices
                    ],
                    smarts=None,
                    metadata=meta,
                )
            )
    return groups


def _match_smarts_groups(
    mol: rdchem.Mol,
    molecule_index: int,
    group_types: list[FunctionalGroupType],
    seen_anchors: set[tuple[FunctionalGroupType, int]],
    counter: dict[FunctionalGroupType, int],
    source: str,
    molecule_role: str | None,
) -> list[FunctionalGroup]:
    groups: list[FunctionalGroup] = []

    for group_type in group_types:
        query = _get_query(group_type)
        smarts = FUNCTIONAL_GROUP_SMARTS[group_type]

        for match in mol.GetSubstructMatches(query):
            anchor = match[0]
            key = (group_type, anchor)
            if key in seen_anchors:
                continue
            seen_anchors.add(key)

            count = counter.get(group_type, 0)
            counter[group_type] = count + 1
            atom_indices = list(match)

            meta: dict = {
                "source": source,
                "anchor_atom_index": anchor,
                "atom_indices": atom_indices,
                "priority": _GROUP_PRIORITY[group_type],
            }
            if molecule_role is not None:
                meta["molecule_role"] = molecule_role

            groups.append(
                FunctionalGroup(
                    group_id=f"mol{molecule_index}_{group_type.value}_{count}",
                    group_type=group_type,
                    atom_refs=[
                        AtomRef(
                            molecule_index=molecule_index,
                            atom_index=idx,
                            atom_map_num=mol.GetAtomWithIdx(idx).GetAtomMapNum() or None,
                        )
                        for idx in match
                    ],
                    smarts=smarts,
                    metadata=meta,
                )
            )
    return groups


# Michael-acceptor (conjugate-addition) systems. For each pattern the first two
# atoms are the alkene: match[0] = beta-carbon (electrophilic 1,4-addition site),
# match[1] = alpha-carbon (bonded to the activating EWG). Ester is checked before
# the generic carbonyl pattern so acrylates report activating_group="ester".
_MICHAEL_ACCEPTOR_SMARTS: list[tuple[str, str]] = [
    ("ester", "[CX3]=[CX3][CX3](=[OX1])[OX2]"),
    ("carbonyl", "[CX3]=[CX3][CX3]=[OX1]"),
    ("nitrile", "[CX3]=[CX3][CX2]#[NX1]"),
    ("nitro", "[CX3]=[CX3][$([NX3](=O)=O),$([N+](=O)[O-])]"),
]
_MICHAEL_QUERIES: list[tuple[str, rdchem.Mol]] = [
    (name, q)
    for name, smarts in _MICHAEL_ACCEPTOR_SMARTS
    if (q := Chem.MolFromSmarts(smarts)) is not None
]


def _annotate_michael_acceptors(
    mol: rdchem.Mol, groups: list[FunctionalGroup]
) -> None:
    """Tag alkene groups that are Michael acceptors with metadata (in place).

    Sets on the matching alkene FunctionalGroup.metadata:
      is_michael_acceptor=True, activating_group (carbonyl/ester/nitrile/nitro),
      beta_carbon_atom_index, and beta_carbon_atom_map_num when an atom map exists.
    Does not change the descriptor schema — these are metadata signals consumed by
    the negotiator, not new feature columns.
    """
    alkene_groups = [g for g in groups if g.group_type == FunctionalGroupType.alkene]
    if not alkene_groups:
        return
    for activating, query in _MICHAEL_QUERIES:
        for match in mol.GetSubstructMatches(query):
            beta_idx, alpha_idx = match[0], match[1]
            for group in alkene_groups:
                idxs = {ref.atom_index for ref in group.atom_refs}
                if beta_idx in idxs and alpha_idx in idxs:
                    if group.metadata.get("is_michael_acceptor"):
                        continue  # first (most specific) activating group wins
                    group.metadata["is_michael_acceptor"] = True
                    group.metadata["activating_group"] = activating
                    group.metadata["beta_carbon_atom_index"] = beta_idx
                    amap = mol.GetAtomWithIdx(beta_idx).GetAtomMapNum()
                    if amap:
                        group.metadata["beta_carbon_atom_map_num"] = amap


def identify_functional_groups_in_mol(
    mol: rdchem.Mol,
    molecule_index: int,
    include_contextual: bool = True,
    _molecule_role: str | None = None,
) -> list[FunctionalGroup]:
    """Return all functional groups found in a single RDKit molecule.

    Args:
        mol: RDKit Mol object.
        molecule_index: Index used to build deterministic group_ids.
        include_contextual: If True, also detect alpha_carbon and benzylic_site.
        _molecule_role: Internal — "reactant" or "product" stored in metadata.
    """
    if mol is None:
        raise ValueError("mol must be a valid RDKit Mol object")

    groups: list[FunctionalGroup] = []
    seen_anchors: set[tuple[FunctionalGroupType, int]] = set()
    counter: dict[FunctionalGroupType, int] = {}

    groups.extend(_detect_aromatic_groups(mol, molecule_index, counter, _molecule_role))
    groups.extend(
        _match_smarts_groups(
            mol, molecule_index, _PRIMARY_ORDER, seen_anchors, counter, "smarts", _molecule_role
        )
    )
    groups.extend(_detect_water_groups(mol, molecule_index, seen_anchors, counter, _molecule_role))
    if include_contextual:
        groups.extend(
            _match_smarts_groups(
                mol,
                molecule_index,
                _CONTEXTUAL_ORDER,
                seen_anchors,
                counter,
                "contextual",
                _molecule_role,
            )
        )
    _annotate_michael_acceptors(mol, groups)
    return groups


def identify_functional_groups(
    parsed_reaction: ParsedReaction,
    include_products: bool = False,
    include_contextual: bool = True,
) -> list[FunctionalGroup]:
    """Return all functional groups found in a parsed reaction.

    Args:
        parsed_reaction: Output of parse_reaction_smiles / parse_reaction_record.
        include_products: If True, also identify groups in product molecules.
        include_contextual: If True, detect alpha_carbon and benzylic_site.
    """
    all_groups: list[FunctionalGroup] = []

    for pm in parsed_reaction.reactants:
        mol = Chem.MolFromSmiles(pm.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for reactant {pm.molecule_index}: {pm.smiles!r}")
        all_groups.extend(
            identify_functional_groups_in_mol(
                mol, pm.molecule_index, include_contextual, _molecule_role="reactant"
            )
        )

    if include_products:
        for pm in parsed_reaction.products:
            mol = Chem.MolFromSmiles(pm.smiles)
            if mol is None:
                raise ValueError(
                    f"Invalid SMILES for product {pm.molecule_index}: {pm.smiles!r}"
                )
            all_groups.extend(
                identify_functional_groups_in_mol(
                    mol, pm.molecule_index, include_contextual, _molecule_role="product"
                )
            )

    return all_groups


def get_group_summary(groups: list[FunctionalGroup]) -> dict[str, int]:
    """Return a count of each group type found."""
    summary: dict[str, int] = {}
    for g in groups:
        key = g.group_type.value
        summary[key] = summary.get(key, 0) + 1
    return summary


def has_group_type(
    groups: list[FunctionalGroup],
    group_type: FunctionalGroupType | str,
) -> bool:
    """Return True if any group in the list has the given type."""
    if isinstance(group_type, str):
        group_type = FunctionalGroupType(group_type)
    return any(g.group_type == group_type for g in groups)


def validate_smarts_patterns() -> dict[str, bool]:
    """Return a dict mapping each group type name to whether its SMARTS is valid."""
    return {
        group_type.value: Chem.MolFromSmarts(smarts) is not None
        for group_type, smarts in FUNCTIONAL_GROUP_SMARTS.items()
    }

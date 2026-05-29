"""Phase 3: Descriptor builder for functional-group agents.

Each FunctionalGroup is converted into a deterministic 55-dimensional feature vector
covering identity, electronic, local environment, mechanistic heuristic scores, and
reaction context. Heuristic scores are chemistry priors — NOT role predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import AllChem

from mendel.parser import ParsedReaction
from mendel.types import FunctionalGroup, FunctionalGroupType, ReactionContext

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
}

EWG_GROUPS: frozenset[FunctionalGroupType] = frozenset({
    FunctionalGroupType.carbonyl,
    FunctionalGroupType.carboxylic_acid,
    FunctionalGroupType.ester,
    FunctionalGroupType.amide,
    FunctionalGroupType.nitrile,
    FunctionalGroupType.nitro,
})

LEAVING_GROUP_ATOMS: frozenset[str] = frozenset({"F", "Cl", "Br", "I"})

FEATURE_SCHEMA_VERSION: str = "phase6_6_v1"

# ---------------------------------------------------------------------------
# Feature schema — order fixed at module load time
# ---------------------------------------------------------------------------

_ONE_HOT_NAMES: list[str] = [f"is_{gt.value}" for gt in FunctionalGroupType]

_IDENTITY_NAMES: list[str] = _ONE_HOT_NAMES + [
    "atom_count",
    "heteroatom_count",
    "aromatic_atom_fraction",
    "ring_atom_fraction",
]

_ELECTRONIC_NAMES: list[str] = [
    "total_formal_charge",
    "mean_formal_charge",
    "mean_gasteiger_charge",
    "min_gasteiger_charge",
    "max_gasteiger_charge",
    "mean_electronegativity",
    "max_electronegativity",
    "has_electronegative_atom",
    "has_pi_bond",
    "is_aromatic_group",
]

_LOCAL_ENV_NAMES: list[str] = [
    "neighbor_heteroatom_count",
    "distance_to_nearest_heteroatom",
    "distance_to_nearest_leaving_group_atom",
    "neighbor_ewg_count",
    "env_alpha_carbon",
    "env_benzylic_site",
    "is_allylic_like",
    "in_reactant",
    "in_product",
]

_MECHANISTIC_NAMES: list[str] = [
    "nucleophilicity_score",
    "electrophilicity_score",
    "leaving_group_score",
    "acidity_score",
    "radical_stability_score",
]

_CONTEXT_NAMES: list[str] = [
    "context_ionic",
    "context_radical",
    "context_pericyclic",
    "context_unknown",
    "condition_acidic",
    "condition_basic",
    "condition_neutral",
    "condition_thermal",
    "condition_photochemical",
    "solvent_polarity_score",
]

_PARTNER_NAMES: list[str] = [
    "partner_has_carbonyl",
    "partner_has_alpha_carbon",
    "partner_has_alkene",
    "partner_has_halide",
    "partner_max_nuc_score",
    "partner_max_elec_score",
    "rel_nuc_score",
    "rel_elec_score",
    "n_reactant_mols",
    "same_mol_has_alpha_carbon",
]

_FEATURE_NAMES: list[str] = (
    _IDENTITY_NAMES
    + _ELECTRONIC_NAMES
    + _LOCAL_ENV_NAMES
    + _MECHANISTIC_NAMES
    + _CONTEXT_NAMES
    + _PARTNER_NAMES
)

# ---------------------------------------------------------------------------
# Heuristic score tables
# ---------------------------------------------------------------------------

_NUCLEOPHILICITY_BASE: dict[FunctionalGroupType, float] = {
    FunctionalGroupType.alkene: 0.35,
    FunctionalGroupType.alkyne: 0.30,
    FunctionalGroupType.aromatic: 0.30,
    FunctionalGroupType.alcohol: 0.50,
    FunctionalGroupType.phenol: 0.45,
    FunctionalGroupType.ether: 0.20,
    FunctionalGroupType.carbonyl: 0.15,
    FunctionalGroupType.carboxylic_acid: 0.35,
    FunctionalGroupType.ester: 0.20,
    FunctionalGroupType.amine: 0.65,
    FunctionalGroupType.amide: 0.30,
    FunctionalGroupType.halide: 0.15,
    FunctionalGroupType.nitrile: 0.10,
    FunctionalGroupType.nitro: 0.10,
    FunctionalGroupType.alpha_carbon: 0.50,
    FunctionalGroupType.benzylic_site: 0.35,
    FunctionalGroupType.unknown: 0.20,
}

_ELECTROPHILICITY_BASE: dict[FunctionalGroupType, float] = {
    FunctionalGroupType.alkene: 0.25,
    FunctionalGroupType.alkyne: 0.20,
    FunctionalGroupType.aromatic: 0.20,
    FunctionalGroupType.alcohol: 0.10,
    FunctionalGroupType.phenol: 0.15,
    FunctionalGroupType.ether: 0.10,
    FunctionalGroupType.carbonyl: 0.70,
    FunctionalGroupType.carboxylic_acid: 0.60,
    FunctionalGroupType.ester: 0.65,
    FunctionalGroupType.amine: 0.10,
    FunctionalGroupType.amide: 0.50,
    FunctionalGroupType.halide: 0.60,
    FunctionalGroupType.nitrile: 0.50,
    FunctionalGroupType.nitro: 0.45,
    FunctionalGroupType.alpha_carbon: 0.20,
    FunctionalGroupType.benzylic_site: 0.25,
    FunctionalGroupType.unknown: 0.20,
}

_ACIDITY_BASE: dict[FunctionalGroupType, float] = {
    FunctionalGroupType.alkene: 0.08,
    FunctionalGroupType.alkyne: 0.30,
    FunctionalGroupType.aromatic: 0.08,
    FunctionalGroupType.alcohol: 0.30,
    FunctionalGroupType.phenol: 0.65,
    FunctionalGroupType.ether: 0.05,
    FunctionalGroupType.carbonyl: 0.20,
    FunctionalGroupType.carboxylic_acid: 0.90,
    FunctionalGroupType.ester: 0.20,
    FunctionalGroupType.amine: 0.15,
    FunctionalGroupType.amide: 0.30,
    FunctionalGroupType.halide: 0.10,
    FunctionalGroupType.nitrile: 0.20,
    FunctionalGroupType.nitro: 0.20,
    FunctionalGroupType.alpha_carbon: 0.55,
    FunctionalGroupType.benzylic_site: 0.45,
    FunctionalGroupType.unknown: 0.10,
}

_RADICAL_STABILITY_BASE: dict[FunctionalGroupType, float] = {
    FunctionalGroupType.alkene: 0.35,
    FunctionalGroupType.alkyne: 0.25,
    FunctionalGroupType.aromatic: 0.50,
    FunctionalGroupType.alcohol: 0.20,
    FunctionalGroupType.phenol: 0.40,
    FunctionalGroupType.ether: 0.25,
    FunctionalGroupType.carbonyl: 0.20,
    FunctionalGroupType.carboxylic_acid: 0.15,
    FunctionalGroupType.ester: 0.20,
    FunctionalGroupType.amine: 0.35,
    FunctionalGroupType.amide: 0.20,
    FunctionalGroupType.halide: 0.10,
    FunctionalGroupType.nitrile: 0.15,
    FunctionalGroupType.nitro: 0.10,
    FunctionalGroupType.alpha_carbon: 0.40,
    FunctionalGroupType.benzylic_site: 0.80,
    FunctionalGroupType.unknown: 0.15,
}

_ALLYLIC_QUERY: Chem.Mol | None = Chem.MolFromSmarts("[CX4;H1,H2,H3][CX3]=[CX3]")
_EWG_QUERIES: list[Chem.Mol] = [
    q for q in [
        Chem.MolFromSmarts("[CX3]=[OX1]"),
        Chem.MolFromSmarts("[CX2]#[NX1]"),
        Chem.MolFromSmarts("[$([NX3](=O)=O),$([N+](=O)[O-])]"),
    ] if q is not None
]

# ---------------------------------------------------------------------------
# GroupDescriptor
# ---------------------------------------------------------------------------


@dataclass
class GroupDescriptor:
    """Descriptor vector for one functional-group agent.

    feature_names and values are parallel lists of equal length.
    All values are numeric floats — no NaN, no None.
    """

    group_id: str
    group_type: FunctionalGroupType
    feature_names: list[str]
    values: list[float]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return {group_id, group_type, features: {name: value}, metadata}."""
        return {
            "group_id": self.group_id,
            "group_type": self.group_type.value,
            "features": dict(zip(self.feature_names, self.values, strict=True)),
            "metadata": self.metadata,
        }

    def as_vector(self) -> list[float]:
        """Return a copy of the descriptor as a plain list of floats."""
        return list(self.values)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe(value: object, default: float = 0.0) -> float:
    """Cast to float; replace NaN/inf/None with default."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _find_mol(
    parsed_reaction: ParsedReaction, group: FunctionalGroup
) -> tuple[Chem.Mol | None, str]:
    """Return (RDKit Mol, molecule_role) for the molecule that contains the group."""
    mol_idx = group.atom_refs[0].molecule_index if group.atom_refs else -1
    role = str(group.metadata.get("molecule_role", "reactant"))

    pool = parsed_reaction.products if role == "product" else parsed_reaction.reactants
    for pm in pool:
        if pm.molecule_index == mol_idx:
            return Chem.MolFromSmiles(pm.smiles), role

    for pm in parsed_reaction.reactants:
        if pm.molecule_index == mol_idx:
            return Chem.MolFromSmiles(pm.smiles), "reactant"
    for pm in parsed_reaction.products:
        if pm.molecule_index == mol_idx:
            return Chem.MolFromSmiles(pm.smiles), "product"

    return None, role


def _gasteiger(mol: Chem.Mol, indices: set[int]) -> tuple[float, float, float]:
    """Return (mean, min, max) Gasteiger charges; returns (0,0,0) on failure."""
    try:
        mol_copy = Chem.RWMol(mol)
        AllChem.ComputeGasteigerCharges(mol_copy)
        charges = [
            _safe(mol_copy.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
            for i in indices
        ]
        if not charges:
            return 0.0, 0.0, 0.0
        return sum(charges) / len(charges), min(charges), max(charges)
    except Exception:
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Feature category builders (one function per category)
# ---------------------------------------------------------------------------


def _build_identity(group: FunctionalGroup, mol: Chem.Mol) -> list[float]:
    one_hot = [1.0 if gt == group.group_type else 0.0 for gt in FunctionalGroupType]
    indices = {ref.atom_index for ref in group.atom_refs}
    n = len(indices)
    heteroatom_count = sum(
        1 for i in indices if mol.GetAtomWithIdx(i).GetSymbol() not in ("C", "H")
    )
    aromatic_count = sum(1 for i in indices if mol.GetAtomWithIdx(i).GetIsAromatic())
    ring_atoms: set[int] = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
    ring_count = sum(1 for i in indices if i in ring_atoms)
    return one_hot + [
        float(n),
        float(heteroatom_count),
        aromatic_count / n if n else 0.0,
        ring_count / n if n else 0.0,
    ]


def _build_electronic(group: FunctionalGroup, mol: Chem.Mol) -> list[float]:
    indices = {ref.atom_index for ref in group.atom_refs}
    formal_charges = [mol.GetAtomWithIdx(i).GetFormalCharge() for i in indices]
    total_fc = float(sum(formal_charges))
    mean_fc = _safe(total_fc / len(formal_charges)) if formal_charges else 0.0
    mean_g, min_g, max_g = _gasteiger(mol, indices)
    en_vals = [ELECTRONEGATIVITY.get(mol.GetAtomWithIdx(i).GetSymbol(), 2.0) for i in indices]
    mean_en = _safe(sum(en_vals) / len(en_vals)) if en_vals else 0.0
    max_en = max(en_vals) if en_vals else 0.0
    has_en_atom = 1.0 if any(e > 2.9 for e in en_vals) else 0.0
    has_pi = 0.0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in indices and j in indices:
            if bond.GetBondTypeAsDouble() >= 2.0 or bond.GetIsAromatic():
                has_pi = 1.0
                break
    if not has_pi and any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in indices):
        has_pi = 1.0
    is_arom = 1.0 if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in indices) else 0.0
    return [total_fc, mean_fc, mean_g, min_g, max_g, mean_en, max_en, has_en_atom, has_pi, is_arom]


def _build_local_env(group: FunctionalGroup, mol: Chem.Mol, molecule_role: str) -> list[float]:
    indices = {ref.atom_index for ref in group.atom_refs}
    n_atoms = mol.GetNumAtoms()
    external_nbrs: set[int] = set()
    for idx in indices:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in indices:
                external_nbrs.add(nidx)
    neighbor_heteroatom_count = sum(
        1 for i in external_nbrs if mol.GetAtomWithIdx(i).GetSymbol() not in ("C", "H")
    )
    dist_to_heteroatom = 99.0
    dist_to_leaving = 99.0
    if n_atoms > 1:
        dm = Chem.GetDistanceMatrix(mol)
        for aidx in range(n_atoms):
            if aidx in indices:
                continue
            sym = mol.GetAtomWithIdx(aidx).GetSymbol()
            for gidx in indices:
                d = float(dm[gidx][aidx])
                if sym not in ("C", "H") and d < dist_to_heteroatom:
                    dist_to_heteroatom = d
                if sym in LEAVING_GROUP_ATOMS and d < dist_to_leaving:
                    dist_to_leaving = d
    ewg_atom_set: set[int] = set()
    for q in _EWG_QUERIES:
        for match in mol.GetSubstructMatches(q):
            ewg_atom_set.update(match)
    neighbor_ewg_count = sum(1 for i in external_nbrs if i in ewg_atom_set)
    is_alpha = 1.0 if group.group_type == FunctionalGroupType.alpha_carbon else 0.0
    is_benz = 1.0 if group.group_type == FunctionalGroupType.benzylic_site else 0.0
    is_allylic = 0.0
    if _ALLYLIC_QUERY is not None:
        for match in mol.GetSubstructMatches(_ALLYLIC_QUERY):
            if match[0] in indices:
                is_allylic = 1.0
                break
    in_reactant = 1.0 if molecule_role == "reactant" else 0.0
    in_product = 1.0 if molecule_role == "product" else 0.0
    return [
        float(neighbor_heteroatom_count),
        _safe(dist_to_heteroatom),
        _safe(dist_to_leaving),
        float(neighbor_ewg_count),
        is_alpha, is_benz, is_allylic,
        in_reactant, in_product,
    ]


def _build_mechanistic(group: FunctionalGroup, mol: Chem.Mol) -> list[float]:
    gt = group.group_type
    indices = {ref.atom_index for ref in group.atom_refs}
    total_fc = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in indices)
    nuc = min(1.0, _NUCLEOPHILICITY_BASE.get(gt, 0.20) + max(0.0, -total_fc * 0.15))
    elec = max(0.0, min(1.0, _ELECTROPHILICITY_BASE.get(gt, 0.20) + max(0.0, total_fc * 0.05)))
    if gt == FunctionalGroupType.halide and len(group.atom_refs) >= 2:
        sym = mol.GetAtomWithIdx(group.atom_refs[1].atom_index).GetSymbol()
        lg = {"F": 0.50, "Cl": 0.75, "Br": 0.85, "I": 0.95}.get(sym, 0.70)
    elif gt == FunctionalGroupType.carboxylic_acid:
        lg = 0.30
    else:
        lg = 0.05
    return [
        _safe(nuc),
        _safe(elec),
        _safe(lg),
        _safe(_ACIDITY_BASE.get(gt, 0.10)),
        _safe(_RADICAL_STABILITY_BASE.get(gt, 0.15)),
    ]


def _build_context(parsed_reaction: ParsedReaction) -> list[float]:
    ctx = parsed_reaction.context
    meta = parsed_reaction.metadata

    def _m(key: str) -> float:
        return _safe(meta.get(key, 0.0))

    return [
        1.0 if ctx == ReactionContext.ionic else 0.0,
        1.0 if ctx == ReactionContext.radical else 0.0,
        1.0 if ctx == ReactionContext.pericyclic else 0.0,
        1.0 if ctx == ReactionContext.unknown else 0.0,
        _m("condition_acidic"),
        _m("condition_basic"),
        _m("condition_neutral"),
        _m("condition_thermal"),
        _m("condition_photochemical"),
        _m("solvent_polarity_score"),
    ]


def _build_partner_context(
    parsed_reaction: ParsedReaction,
    group: FunctionalGroup,
    all_groups: list[FunctionalGroup],
) -> list[float]:
    this_mol_idx = group.atom_refs[0].molecule_index if group.atom_refs else -1

    def _mol_idx(g: FunctionalGroup) -> int:
        return g.atom_refs[0].molecule_index if g.atom_refs else -1

    reactant_mol_indices = {pm.molecule_index for pm in parsed_reaction.reactants}
    n_reactant_molecules = len(reactant_mol_indices)

    partner_groups = [
        g for g in all_groups
        if _mol_idx(g) != this_mol_idx and _mol_idx(g) in reactant_mol_indices
    ]
    same_mol_groups = [
        g for g in all_groups
        if _mol_idx(g) == this_mol_idx and g.group_id != group.group_id
    ]

    carbonyl_types = {FunctionalGroupType.carbonyl, FunctionalGroupType.carboxylic_acid,
                      FunctionalGroupType.ester, FunctionalGroupType.amide}

    partner_has_carbonyl = float(any(g.group_type in carbonyl_types for g in partner_groups))
    partner_has_alpha_carbon = float(any(g.group_type == FunctionalGroupType.alpha_carbon for g in partner_groups))
    partner_has_alkene = float(any(g.group_type == FunctionalGroupType.alkene for g in partner_groups))
    partner_has_halide = float(any(g.group_type == FunctionalGroupType.halide for g in partner_groups))

    partner_nuc_scores = [_NUCLEOPHILICITY_BASE.get(g.group_type, 0.20) for g in partner_groups]
    partner_elec_scores = [_ELECTROPHILICITY_BASE.get(g.group_type, 0.20) for g in partner_groups]
    partner_max_nuc = max(partner_nuc_scores) if partner_nuc_scores else 0.0
    partner_max_elec = max(partner_elec_scores) if partner_elec_scores else 0.0

    this_nuc = _NUCLEOPHILICITY_BASE.get(group.group_type, 0.20)
    this_elec = _ELECTROPHILICITY_BASE.get(group.group_type, 0.20)
    rel_nuc = _safe(this_nuc - partner_max_elec)
    rel_elec = _safe(this_elec - partner_max_nuc)

    n_reactant_mols = n_reactant_molecules / 4.0

    same_mol_has_alpha = float(
        any(g.group_type == FunctionalGroupType.alpha_carbon for g in same_mol_groups)
    )

    return [
        partner_has_carbonyl,
        partner_has_alpha_carbon,
        partner_has_alkene,
        partner_has_halide,
        _safe(partner_max_nuc),
        _safe(partner_max_elec),
        rel_nuc,
        rel_elec,
        _safe(n_reactant_mols),
        same_mol_has_alpha,
    ]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_feature_names() -> list[str]:
    """Return the global feature schema in fixed order. Deterministic across calls."""
    return list(_FEATURE_NAMES)


def build_group_descriptor(
    parsed_reaction: ParsedReaction,
    group: FunctionalGroup,
    all_groups: list[FunctionalGroup] | None = None,
) -> GroupDescriptor:
    """Build one descriptor vector for one functional-group agent.

    Feature schema (feature_names order) is identical for every descriptor
    produced by this module version.
    """
    mol, molecule_role = _find_mol(parsed_reaction, group)

    if mol is None:
        return GroupDescriptor(
            group_id=group.group_id,
            group_type=group.group_type,
            feature_names=get_feature_names(),
            values=[0.0] * len(_FEATURE_NAMES),
            metadata={"error": "molecule_not_found", "schema_version": FEATURE_SCHEMA_VERSION},
        )

    _all_groups = all_groups if all_groups is not None else [group]
    raw = (
        _build_identity(group, mol)
        + _build_electronic(group, mol)
        + _build_local_env(group, mol, molecule_role)
        + _build_mechanistic(group, mol)
        + _build_context(parsed_reaction)
        + _build_partner_context(parsed_reaction, group, _all_groups)
    )
    return GroupDescriptor(
        group_id=group.group_id,
        group_type=group.group_type,
        feature_names=get_feature_names(),
        values=[_safe(v) for v in raw],
        metadata={
            "schema_version": FEATURE_SCHEMA_VERSION,
            "molecule_role": molecule_role,
            "n_atoms_in_group": len({ref.atom_index for ref in group.atom_refs}),
        },
    )


def build_descriptors(
    parsed_reaction: ParsedReaction,
    groups: list[FunctionalGroup],
) -> list[GroupDescriptor]:
    """Build descriptors for all groups, preserving input order."""
    return [build_group_descriptor(parsed_reaction, g, all_groups=groups) for g in groups]


def descriptor_matrix(
    descriptors: list[GroupDescriptor],
) -> tuple[list[str], list[list[float]]]:
    """Return (feature_names, matrix) where matrix is n_groups × n_features."""
    if not descriptors:
        return get_feature_names(), []
    return descriptors[0].feature_names, [d.as_vector() for d in descriptors]


def summarize_descriptors(descriptors: list[GroupDescriptor]) -> dict[str, object]:
    """Return a debug summary: counts, group ids/types, schema version."""
    return {
        "n_descriptors": len(descriptors),
        "n_features": len(descriptors[0].feature_names) if descriptors else 0,
        "group_ids": [d.group_id for d in descriptors],
        "group_types": [d.group_type.value for d in descriptors],
        "schema_version": FEATURE_SCHEMA_VERSION,
    }


def validate_descriptor_schema(descriptors: list[GroupDescriptor]) -> bool:
    """Return True only if all descriptors share the schema and contain valid floats."""
    if not descriptors:
        return True
    expected = get_feature_names()
    for d in descriptors:
        if d.feature_names != expected:
            return False
        if len(d.feature_names) != len(d.values):
            return False
        if any(math.isnan(v) or math.isinf(v) for v in d.values):
            return False
    return True

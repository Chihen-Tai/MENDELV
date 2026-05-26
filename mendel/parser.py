"""Module 0: Reaction SMILES parser for MENDEL.

Converts a reaction SMILES string into structured ParsedReaction objects.
Molecule-level properties (charge, radical presence, atom mapping) are
extracted here. Functional group identification happens in Phase 2.

Accepted input format:
    reactants>>products
    Molecules within each side are separated by ".".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from mendel.types import ReactionContext, ReactionRecord


class ReactionParseError(Exception):
    """Raised when a reaction SMILES string cannot be parsed."""


@dataclass
class ParsedMolecule:
    """Parsed representation of one molecule within a reaction.

    molecule_index: 0-based position within the reactants or products list.
    smiles: canonical SMILES produced by RDKit.
    role: "reactant" or "product".
    num_atoms: heavy-atom count.
    formal_charge: sum of formal charges across all atoms.
    has_radical: True if any atom carries radical electrons.
    atom_map_nums: {atom_index: atom_map_number} for mapped atoms only.
    metadata: extensible key/value store for downstream modules.
    """

    molecule_index: int
    smiles: str
    role: str
    num_atoms: int
    formal_charge: int
    has_radical: bool
    atom_map_nums: dict[int, int]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class ParsedReaction:
    """Structured representation of a fully parsed reaction.

    reaction_smiles: the original input string, preserved verbatim.
    reactants: ParsedMolecule list for the left side of ">>".
    products: ParsedMolecule list for the right side of ">>".
    context: broad mechanistic category (ionic / radical / pericyclic / unknown).
    has_atom_mapping: True if at least one atom carries a map number.
    total_charge_reactants: algebraic sum of charges across all reactants.
    total_charge_products: algebraic sum of charges across all products.
    metadata: extensible key/value store.
    """

    reaction_smiles: str
    reactants: list[ParsedMolecule]
    products: list[ParsedMolecule]
    context: ReactionContext
    has_atom_mapping: bool
    total_charge_reactants: int
    total_charge_products: int
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_one_molecule(smiles: str, index: int, role: str) -> ParsedMolecule:
    """Parse a single molecule SMILES with RDKit.

    Raises ReactionParseError if RDKit returns None (invalid SMILES).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ReactionParseError(
            f"RDKit could not parse molecule SMILES {smiles!r} "
            f"(position {index} in {role}s)"
        )

    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    has_radical = any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
    atom_map_nums: dict[int, int] = {
        atom.GetIdx(): atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }
    canonical = Chem.MolToSmiles(mol)

    return ParsedMolecule(
        molecule_index=index,
        smiles=canonical,
        role=role,
        num_atoms=mol.GetNumAtoms(),
        formal_charge=formal_charge,
        has_radical=has_radical,
        atom_map_nums=atom_map_nums,
    )


def _resolve_context(context: ReactionContext | str) -> ReactionContext:
    """Coerce a string into a ReactionContext; falls back to unknown."""
    if isinstance(context, ReactionContext):
        return context
    try:
        return ReactionContext(context)
    except ValueError:
        return ReactionContext.unknown


def _split_smiles_side(side: str) -> list[str]:
    """Split one side of a reaction SMILES into individual molecule strings."""
    return [s for s in side.split(".") if s]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_reaction_smiles(
    reaction_smiles: str,
    context: ReactionContext | str = ReactionContext.unknown,
) -> ParsedReaction:
    """Parse a reaction SMILES string into a structured ParsedReaction.

    Args:
        reaction_smiles: String of the form "reactants>>products".
                         Multiple molecules on each side are separated by ".".
        context: Mechanistic category. Accepts a ReactionContext enum value
                 or a plain string ("ionic", "radical", "pericyclic", "unknown").
                 Unrecognised strings map to ReactionContext.unknown.

    Returns:
        A populated ParsedReaction dataclass.

    Raises:
        ReactionParseError: If the string is missing ">>", has empty sides,
                            or contains any molecule SMILES that RDKit rejects.
    """
    if ">>" not in reaction_smiles:
        raise ReactionParseError(
            f"Reaction SMILES must contain '>>'. Got: {reaction_smiles!r}"
        )

    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        raise ReactionParseError(
            f"Reaction SMILES must contain exactly one '>>'. Got: {reaction_smiles!r}"
        )

    reactants_str, products_str = parts

    if not reactants_str.strip():
        raise ReactionParseError("Reactants side of '>>' is empty.")
    if not products_str.strip():
        raise ReactionParseError("Products side of '>>' is empty.")

    reactant_tokens = _split_smiles_side(reactants_str)
    product_tokens = _split_smiles_side(products_str)

    if not reactant_tokens:
        raise ReactionParseError("No reactant SMILES found after splitting by '.'.")
    if not product_tokens:
        raise ReactionParseError("No product SMILES found after splitting by '.'.")

    reactants = [_parse_one_molecule(s, i, "reactant") for i, s in enumerate(reactant_tokens)]
    products = [_parse_one_molecule(s, i, "product") for i, s in enumerate(product_tokens)]

    has_atom_mapping = any(mol.atom_map_nums for mol in reactants + products)
    total_charge_reactants = sum(mol.formal_charge for mol in reactants)
    total_charge_products = sum(mol.formal_charge for mol in products)

    return ParsedReaction(
        reaction_smiles=reaction_smiles,
        reactants=reactants,
        products=products,
        context=_resolve_context(context),
        has_atom_mapping=has_atom_mapping,
        total_charge_reactants=total_charge_reactants,
        total_charge_products=total_charge_products,
    )


def parse_reaction_record(record: ReactionRecord) -> ParsedReaction:
    """Parse a Phase 0 ReactionRecord into a ParsedReaction.

    Convenience wrapper that reads reaction_smiles and context from the record.
    """
    return parse_reaction_smiles(
        reaction_smiles=record.reaction_smiles,
        context=record.context,
    )


def validate_reaction_smiles(reaction_smiles: str) -> bool:
    """Return True if the reaction SMILES is parseable, False otherwise.

    Never raises. Use parse_reaction_smiles directly for error details.
    """
    try:
        parse_reaction_smiles(reaction_smiles)
        return True
    except ReactionParseError:
        return False


def get_reaction_summary(parsed: ParsedReaction) -> dict[str, Any]:
    """Return a compact summary dict useful for tests and debugging.

    Keys: n_reactants, n_products, total_charge_reactants,
          total_charge_products, has_atom_mapping, has_radicals.
    """
    return {
        "n_reactants": len(parsed.reactants),
        "n_products": len(parsed.products),
        "total_charge_reactants": parsed.total_charge_reactants,
        "total_charge_products": parsed.total_charge_products,
        "has_atom_mapping": parsed.has_atom_mapping,
        "has_radicals": any(m.has_radical for m in parsed.reactants + parsed.products),
    }

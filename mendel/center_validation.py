"""Phase 8.11 leakage-resistant center validation utilities.

This module audits atom-level reaction-center labels and builds deterministic
split assignments that reduce template leakage. It does not train MLIP, MACE,
energies, forces, transition states, or barriers.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem

from mendel.center_head import build_atom_center_examples, summarize_atom_center_examples
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledReaction, load_labeled_reactions
from mendel.parser import parse_reaction_smiles
from mendel.types import FunctionalGroupType, Role

Scalar = str | int | float | bool

_CONTROL_MECHANISMS = frozenset({"control", "ester_control", "nitrile_control", "no_reaction"})
_VALID_STRATEGIES = frozenset(
    {
        "template",
        "mechanism",
        "source",
        "reaction_id_prefix",
        "mechanism_balanced_template",
        "val_test_balanced_template",
    }
)


@dataclass
class CenterLabelIssue:
    reaction_id: str
    mechanism_type: str
    issue_code: str
    severity: str
    message: str
    reaction_center_atoms: list[int]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "issue_code": self.issue_code,
            "severity": self.severity,
            "message": self.message,
            "reaction_center_atoms": list(self.reaction_center_atoms),
            "metadata": dict(self.metadata),
        }


@dataclass
class CenterSplitRecord:
    reaction_id: str
    mechanism_type: str
    original_split: str
    leakage_group: str
    leakage_group_type: str
    assigned_split: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "mechanism_type": self.mechanism_type,
            "original_split": self.original_split,
            "leakage_group": self.leakage_group,
            "leakage_group_type": self.leakage_group_type,
            "assigned_split": self.assigned_split,
            "metadata": dict(self.metadata),
        }


@dataclass
class LeakageValidationReport:
    dataset_path: str
    split_strategy: str
    n_reactions: int
    n_atom_examples: int
    n_positive_center_atoms: int
    n_negative_atoms: int
    original_split_distribution: dict[str, int]
    new_split_distribution: dict[str, int]
    mechanism_distribution_by_split: dict[str, dict[str, int]]
    leakage_groups_by_split: dict[str, list[str]]
    center_label_issues: list[CenterLabelIssue]
    split_records: list[CenterSplitRecord]
    metrics: dict[str, object]
    recommendations: list[str]
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "split_strategy": self.split_strategy,
            "n_reactions": self.n_reactions,
            "n_atom_examples": self.n_atom_examples,
            "n_positive_center_atoms": self.n_positive_center_atoms,
            "n_negative_atoms": self.n_negative_atoms,
            "original_split_distribution": dict(self.original_split_distribution),
            "new_split_distribution": dict(self.new_split_distribution),
            "mechanism_distribution_by_split": {
                split: dict(counts)
                for split, counts in self.mechanism_distribution_by_split.items()
            },
            "leakage_groups_by_split": {
                split: list(groups) for split, groups in self.leakage_groups_by_split.items()
            },
            "center_label_issues": [issue.to_dict() for issue in self.center_label_issues],
            "split_records": [record.to_dict() for record in self.split_records],
            "metrics": dict(self.metrics),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


def _canonical_mechanism(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _reaction_id_prefix(reaction_id: str) -> str:
    parts = reaction_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else reaction_id


def infer_template_key(reaction: LabeledReaction) -> str:
    """Infer a deterministic leakage group key for template-aware splitting."""
    metadata = reaction.metadata
    for key in ("template_name", "generation_template", "template"):
        value = metadata.get(key)
        if value:
            return str(value)
    pieces = [
        str(metadata.get("generation_method", "")),
        str(metadata.get("source_type", "")),
        _canonical_mechanism(reaction.mechanism_type),
        _reaction_id_prefix(reaction.reaction_id),
    ]
    compact = [piece for piece in pieces if piece]
    if compact:
        return "::".join(compact)
    reactants, _, products = reaction.reaction_smiles.partition(">>")
    return f"{_canonical_mechanism(reaction.mechanism_type)}::{reactants}>>{products}"


def infer_mechanism_key(reaction: LabeledReaction) -> str:
    return _canonical_mechanism(reaction.mechanism_type)


def _group_key(reaction: LabeledReaction, strategy: str) -> str:
    if strategy in {"template", "mechanism_balanced_template", "val_test_balanced_template"}:
        return infer_template_key(reaction)
    if strategy == "mechanism":
        return infer_mechanism_key(reaction)
    if strategy == "source":
        return str(
            reaction.metadata.get("source_type")
            or reaction.metadata.get("generation_method")
            or "unknown_source"
        )
    if strategy == "reaction_id_prefix":
        return _reaction_id_prefix(reaction.reaction_id)
    raise ValueError(f"Unsupported leakage split strategy: {strategy}")


def _target_split_for_group(
    group_index: int,
    n_groups: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    if n_groups == 1:
        return "train"
    train_cut = max(int(round(n_groups * train_fraction)), 1)
    val_cut = max(int(round(n_groups * (train_fraction + val_fraction))), train_cut + 1)
    if group_index < min(train_cut, n_groups):
        return "train"
    if group_index < min(val_cut, n_groups):
        return "val"
    return "test"


def _balanced_group_assignments(
    groups: dict[str, list[LabeledReaction]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[dict[str, str], str | None]:
    group_keys = sorted(groups)
    by_mechanism: dict[str, list[str]] = {}
    for group in group_keys:
        mechanism = groups[group][0].mechanism_type
        by_mechanism.setdefault(mechanism, []).append(group)
    rng = random.Random(seed)
    for keys in by_mechanism.values():
        rng.shuffle(keys)

    n_groups = len(group_keys)
    n_train = max(int(round(n_groups * train_fraction)), 1) if n_groups else 0
    n_val = max(int(round(n_groups * val_fraction)), 1) if n_groups >= 3 else 0
    n_test = max(n_groups - n_train - n_val, 0)
    if n_groups >= 3 and n_test == 0:
        n_test = 1
        n_train = max(n_train - 1, 1)

    assignments: dict[str, str] = {}
    mechanisms = sorted(by_mechanism)
    for idx, mechanism in enumerate(mechanisms):
        keys = by_mechanism[mechanism]
        if not keys:
            continue
        if idx % 3 == 0 and len(assignments) < n_groups:
            assignments[keys.pop()] = "test"
        elif idx % 3 == 1:
            assignments[keys.pop()] = "val"
        else:
            assignments[keys.pop()] = "train"

    remaining = [group for group in group_keys if group not in assignments]
    rng.shuffle(remaining)
    counts = Counter(assignments.values())
    for group in remaining:
        if counts.get("train", 0) < n_train:
            split = "train"
        elif counts.get("val", 0) < n_val:
            split = "val"
        elif counts.get("test", 0) < n_test:
            split = "test"
        else:
            split = min(("train", "val", "test"), key=lambda s: counts.get(s, 0))
        assignments[group] = split
        counts[split] += 1

    test_mechanisms = {
        groups[group][0].mechanism_type for group, split in assignments.items() if split == "test"
    }
    warning = None
    if n_groups < 3 or len(test_mechanisms) < min(5, len(mechanisms)):
        warning = "mechanism_balanced_template could not place five mechanisms in test"
    return assignments, warning


def _val_test_balanced_group_assignments(
    groups: dict[str, list[LabeledReaction]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[dict[str, str], str | None]:
    """Assign template groups while broadening both validation and test coverage."""
    group_keys = sorted(groups)
    if not group_keys:
        return {}, "no leakage groups available"
    by_mechanism: dict[str, list[str]] = {}
    for group in group_keys:
        mechanism = groups[group][0].mechanism_type
        by_mechanism.setdefault(mechanism, []).append(group)
    rng = random.Random(seed)
    for mechanism, keys in by_mechanism.items():
        keys.sort(key=lambda key: (len(groups[key]), key))
        rng.shuffle(keys)
        by_mechanism[mechanism] = keys

    total_reactions = sum(len(rxns) for rxns in groups.values())
    target_val_reactions = min(max(15, int(round(total_reactions * val_fraction))), total_reactions)
    target_test_fraction = max(0.0, 1.0 - train_fraction - val_fraction)
    target_test_reactions = min(
        max(15, int(round(total_reactions * target_test_fraction))),
        total_reactions,
    )

    assignments: dict[str, str] = {}
    counts: Counter[str] = Counter()
    mechanism_sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    def assign(group: str, split: str) -> None:
        assignments[group] = split
        counts[split] += len(groups[group])
        mechanism_sets[split].add(groups[group][0].mechanism_type)

    mechanisms = sorted(by_mechanism)
    for idx, mechanism in enumerate(mechanisms):
        keys = [key for key in by_mechanism[mechanism] if key not in assignments]
        if not keys:
            continue
        if len(keys) >= 2:
            first, second = (("val", "test") if idx % 2 == 0 else ("test", "val"))
            assign(keys.pop(), first)
            assign(keys.pop(), second)
        else:
            split = "val" if len(mechanism_sets["val"]) <= len(mechanism_sets["test"]) else "test"
            assign(keys.pop(), split)

    remaining = [group for group in group_keys if group not in assignments]
    remaining.sort(key=lambda group: (groups[group][0].mechanism_type, group))
    for group in remaining:
        val_needs_mechanisms = len(mechanism_sets["val"]) < min(5, len(mechanisms))
        test_needs_mechanisms = len(mechanism_sets["test"]) < min(5, len(mechanisms))
        if counts["val"] < target_val_reactions or val_needs_mechanisms:
            split = "val"
        elif counts["test"] < target_test_reactions or test_needs_mechanisms:
            split = "test"
        else:
            split = "train"
        assign(group, split)

    # If train became empty on tiny datasets, move the largest held-out group back to train.
    if counts["train"] == 0 and len(assignments) > 2:
        candidates = sorted(
            assignments,
            key=lambda group: (len(groups[group]), group),
            reverse=True,
        )
        for group in candidates:
            old_split = assignments[group]
            if old_split in {"val", "test"}:
                mechanism = groups[group][0].mechanism_type
                assignments[group] = "train"
                counts[old_split] -= len(groups[group])
                counts["train"] += len(groups[group])
                if not any(
                    assignments[other] == old_split
                    and groups[other][0].mechanism_type == mechanism
                    for other in assignments
                    if other != group
                ):
                    mechanism_sets[old_split].discard(mechanism)
                mechanism_sets["train"].add(mechanism)
                break

    warnings: list[str] = []
    for split, target in (("val", target_val_reactions), ("test", target_test_reactions)):
        if counts[split] < min(15, total_reactions):
            warnings.append(f"{split} split has fewer than 15 reactions")
        if len(mechanism_sets[split]) < min(5, len(mechanisms)):
            warnings.append(f"{split} split has fewer than five mechanisms")
        if counts[split] < target:
            warnings.append(f"{split} split could not reach target reaction count")
    return assignments, "; ".join(warnings) if warnings else None


def assign_leakage_resistant_splits(
    reactions: list[LabeledReaction],
    strategy: str = "template",
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[LabeledReaction], list[CenterSplitRecord]]:
    """Assign splits while keeping leakage groups intact."""
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_VALID_STRATEGIES)}")
    if train_fraction + val_fraction + test_fraction <= 0:
        raise ValueError("split fractions must sum to a positive value")

    groups: dict[str, list[LabeledReaction]] = {}
    for rxn in reactions:
        groups.setdefault(_group_key(rxn, strategy), []).append(rxn)
    if strategy == "mechanism_balanced_template":
        assigned_by_group, split_warning = _balanced_group_assignments(
            groups, seed, train_fraction, val_fraction
        )
    elif strategy == "val_test_balanced_template":
        assigned_by_group, split_warning = _val_test_balanced_group_assignments(
            groups, seed, train_fraction, val_fraction
        )
    else:
        group_keys = sorted(groups)
        rng = random.Random(seed)
        rng.shuffle(group_keys)
        assigned_by_group = {
            group: _target_split_for_group(idx, len(group_keys), train_fraction, val_fraction)
            for idx, group in enumerate(group_keys)
        }
        split_warning = None

    updated: list[LabeledReaction] = []
    records: list[CenterSplitRecord] = []
    for rxn in reactions:
        group = _group_key(rxn, strategy)
        assigned = assigned_by_group[group]
        metadata = copy.deepcopy(rxn.metadata)
        metadata["original_split"] = rxn.split
        metadata["leakage_split_strategy"] = strategy
        metadata["leakage_group"] = group
        if split_warning:
            metadata["split_warning"] = split_warning
        new_rxn = replace(rxn, split=assigned, metadata=metadata)
        updated.append(new_rxn)
        records.append(
            CenterSplitRecord(
                reaction_id=rxn.reaction_id,
                mechanism_type=rxn.mechanism_type,
                original_split=rxn.split,
                leakage_group=group,
                leakage_group_type=strategy,
                assigned_split=assigned,
                metadata={"n_reactions_in_group": len(groups[group])},
            )
        )
    return updated, records


def _valid_center_values(reaction: LabeledReaction) -> tuple[set[int], set[int]]:
    parsed = parse_reaction_smiles(reaction.reaction_smiles, context=reaction.context)
    atom_maps: set[int] = set()
    atom_indices: set[int] = set()
    for mol in parsed.reactants:
        rd_mol = Chem.MolFromSmiles(mol.smiles)
        if rd_mol is None:
            continue
        for atom in rd_mol.GetAtoms():
            atom_indices.add(atom.GetIdx())
            if atom.GetAtomMapNum():
                atom_maps.add(atom.GetAtomMapNum())
    return atom_maps, atom_indices


def _issue(
    reaction: LabeledReaction,
    issue_code: str,
    severity: str,
    message: str,
    metadata: dict[str, Scalar] | None = None,
) -> CenterLabelIssue:
    return CenterLabelIssue(
        reaction_id=reaction.reaction_id,
        mechanism_type=reaction.mechanism_type,
        issue_code=issue_code,
        severity=severity,
        message=message,
        reaction_center_atoms=list(reaction.reaction_center_atoms),
        metadata=metadata or {},
    )


def audit_center_labels(reactions: list[LabeledReaction]) -> list[CenterLabelIssue]:
    """Audit reaction_center_atoms consistency and mechanism-specific policies."""
    issues: list[CenterLabelIssue] = []
    for rxn in reactions:
        mechanism = _canonical_mechanism(rxn.mechanism_type)
        center = list(rxn.reaction_center_atoms)
        try:
            valid_maps, valid_indices = _valid_center_values(rxn)
            parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
            groups = identify_functional_groups(parsed)
        except Exception as exc:
            issues.append(_issue(rxn, "parse_failed", "error", str(exc)))
            continue
        valid_values = valid_maps if valid_maps else valid_indices
        invalid = [atom for atom in center if atom not in valid_values]
        if invalid:
            issues.append(
                _issue(
                    rxn,
                    "invalid_center_atom",
                    "error",
                    "reaction_center_atoms contains values not present in reactants",
                    {"invalid_atoms": ",".join(str(atom) for atom in invalid)},
                )
            )
        if len(center) != len(set(center)):
            issues.append(
                _issue(rxn, "duplicate_center_atoms", "warning", "duplicate center atoms")
            )
        if mechanism in _CONTROL_MECHANISMS and center:
            issues.append(
                _issue(
                    rxn,
                    "control_has_center_atoms",
                    "error",
                    "control reaction has center atoms",
                )
            )
        if mechanism not in _CONTROL_MECHANISMS and not center:
            issues.append(
                _issue(
                    rxn,
                    "reactive_empty_center",
                    "error",
                    "reactive mechanism has empty reaction_center_atoms",
                )
            )
        if len(center) > 8:
            issues.append(
                _issue(
                    rxn,
                    "center_atom_count_large",
                    "warning",
                    "reaction center contains unusually many atoms",
                    {"n_center_atoms": len(center)},
                )
            )
        if center and all(label.role is Role.spectator for label in rxn.group_roles):
            issues.append(
                _issue(
                    rxn,
                    "spectator_only_with_center",
                    "warning",
                    "all labeled groups are spectators but center atoms are non-empty",
                )
            )
        group_values_by_type: dict[str, set[int]] = {}
        all_group_values: set[int] = set()
        for group in groups:
            values = {
                ref.atom_map_num if ref.atom_map_num is not None else ref.atom_index
                for ref in group.atom_refs
            }
            group_values_by_type.setdefault(group.group_type.value, set()).update(values)
            all_group_values.update(values)
        if center and all_group_values and any(atom not in all_group_values for atom in center):
            issues.append(
                _issue(
                    rxn,
                    "center_atom_outside_detected_groups",
                    "info",
                    "some center atoms are outside detected functional groups",
                )
            )
        if mechanism == "carbonyl_addition":
            alpha = group_values_by_type.get(FunctionalGroupType.alpha_carbon.value, set())
            carbonyl = group_values_by_type.get(FunctionalGroupType.carbonyl.value, set())
            if set(center) & alpha and not (set(center) & carbonyl):
                issues.append(
                    _issue(
                        rxn,
                        "carbonyl_addition_alpha_without_carbonyl",
                        "warning",
                        "carbonyl addition center includes alpha carbon but not carbonyl",
                    )
                )
        if mechanism in {"sn2", "e2"}:
            halide = group_values_by_type.get(FunctionalGroupType.halide.value, set())
            if halide and not (set(center) & halide):
                issues.append(
                    _issue(
                        rxn,
                        "sn2_e2_missing_halide_center",
                        "warning",
                        "SN2/E2 center does not include detected halide group",
                    )
                )
        if mechanism in {"radical_bromination", "benzylic_radical_bromination"}:
            aromatic = group_values_by_type.get(FunctionalGroupType.aromatic.value, set())
            if aromatic and len(set(center) & aromatic) >= 5:
                issues.append(
                    _issue(
                        rxn,
                        "radical_center_includes_aromatic_ring",
                        "warning",
                        "radical bromination center appears to include most of an aromatic ring",
                    )
                )
        if mechanism == "diels_alder":
            substituent_types = [
                FunctionalGroupType.nitrile.value,
                FunctionalGroupType.ester.value,
                FunctionalGroupType.carbonyl.value,
            ]
            substituent_atoms = set().union(
                *(group_values_by_type.get(group_type, set()) for group_type in substituent_types)
            )
            if set(center) & substituent_atoms:
                issues.append(
                    _issue(
                        rxn,
                        "diels_alder_substituent_center_atoms",
                        "warning",
                        "Diels-Alder center includes EWG substituent atoms",
                    )
                )
    return issues


def summarize_center_label_issues(
    issues: list[CenterLabelIssue],
) -> dict[str, object]:
    by_severity: dict[str, int] = {}
    by_issue_code: dict[str, int] = {}
    by_mechanism_type: dict[str, int] = {}
    for issue in issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        by_issue_code[issue.issue_code] = by_issue_code.get(issue.issue_code, 0) + 1
        by_mechanism_type[issue.mechanism_type] = by_mechanism_type.get(issue.mechanism_type, 0) + 1
    return {
        "n_issues": len(issues),
        "by_severity": by_severity,
        "by_issue_code": by_issue_code,
        "by_mechanism_type": by_mechanism_type,
    }


def save_labeled_reactions_json(
    reactions: list[LabeledReaction],
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"reactions": [rxn.to_dict() for rxn in reactions]}, indent=2),
        encoding="utf-8",
    )


def _distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mechanisms_by_split(reactions: list[LabeledReaction]) -> dict[str, dict[str, int]]:
    result = {"train": {}, "val": {}, "test": {}}
    for rxn in reactions:
        split_counts = result.setdefault(rxn.split, {})
        split_counts[rxn.mechanism_type] = split_counts.get(rxn.mechanism_type, 0) + 1
    return {split: dict(sorted(counts.items())) for split, counts in result.items()}


def _leakage_groups_by_split(records: list[CenterSplitRecord]) -> dict[str, list[str]]:
    result = {"train": [], "val": [], "test": []}
    for record in records:
        groups = result.setdefault(record.assigned_split, [])
        if record.leakage_group not in groups:
            groups.append(record.leakage_group)
    return {split: sorted(groups) for split, groups in result.items()}


def _split_diagnostics(
    reactions: list[LabeledReaction],
    records: list[CenterSplitRecord],
) -> dict[str, object]:
    examples = build_atom_center_examples(reactions)
    positive_counts = {"train": 0, "val": 0, "test": 0}
    negative_counts = {"train": 0, "val": 0, "test": 0}
    for example in examples:
        if example.is_labeled_center:
            positive_counts[example.split] = positive_counts.get(example.split, 0) + 1
        else:
            negative_counts[example.split] = negative_counts.get(example.split, 0) + 1

    center_count_distribution: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for reaction in reactions:
        split_counts = center_count_distribution.setdefault(reaction.split, {})
        key = str(len(reaction.reaction_center_atoms))
        split_counts[key] = split_counts.get(key, 0) + 1

    mechanism_distribution = _mechanisms_by_split(reactions)
    leakage_groups = _leakage_groups_by_split(records)
    required = {
        "sn2",
        "e2",
        "diels_alder",
        "carbonyl_addition",
        "benzylic_radical_bromination",
        "radical_bromination",
        "control",
        "ester_control",
        "nitrile_control",
        "aldol",
        "cross_aldol",
    }
    target_coverage: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        mechanisms = set(mechanism_distribution.get(split, {}))
        missing_target_mechanisms = sorted(required - mechanisms)
        target_coverage[split] = {
            "n_reactions": sum(mechanism_distribution.get(split, {}).values()),
            "n_mechanisms": len(mechanisms),
            "mechanism_target_met": len(mechanisms) >= 5 if split in {"val", "test"} else True,
            "reaction_count_target_met": sum(mechanism_distribution.get(split, {}).values()) >= 15
            if split in {"val", "test"}
            else True,
            "missing_target_mechanisms": missing_target_mechanisms,
        }

    return {
        "mechanism_distribution_by_split": mechanism_distribution,
        "positive_atom_counts_by_split": positive_counts,
        "negative_atom_counts_by_split": negative_counts,
        "center_atom_count_distribution_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in center_count_distribution.items()
        },
        "leakage_groups_by_split": leakage_groups,
        "target_coverage": target_coverage,
    }


def build_leakage_validation_report(
    dataset_path: str | Path,
    split_strategy: str,
    output_dataset_path: str | Path | None = None,
) -> LeakageValidationReport:
    reactions = load_labeled_reactions(dataset_path)
    split_reactions, split_records = assign_leakage_resistant_splits(
        reactions,
        strategy=split_strategy,
    )
    if output_dataset_path is not None:
        save_labeled_reactions_json(split_reactions, output_dataset_path)
    issues = audit_center_labels(split_reactions)
    examples = build_atom_center_examples(split_reactions)
    summary = summarize_atom_center_examples(examples)
    issue_summary = summarize_center_label_issues(issues)
    report = LeakageValidationReport(
        dataset_path=str(dataset_path),
        split_strategy=split_strategy,
        n_reactions=len(split_reactions),
        n_atom_examples=int(summary["n_examples"]),
        n_positive_center_atoms=int(summary["n_positive"]),
        n_negative_atoms=int(summary["n_negative"]),
        original_split_distribution=_distribution([rxn.split for rxn in reactions]),
        new_split_distribution=_distribution([rxn.split for rxn in split_reactions]),
        mechanism_distribution_by_split=_mechanisms_by_split(split_reactions),
        leakage_groups_by_split=_leakage_groups_by_split(split_records),
        center_label_issues=issues,
        split_records=split_records,
        metrics={
            "atom_example_summary": summary,
            "center_label_issue_summary": issue_summary,
            "split_diagnostics": _split_diagnostics(split_reactions, split_records),
        },
        recommendations=[],
        metadata={
            "output_dataset_path": str(output_dataset_path or ""),
            "scope_note": "Leakage validation only; no MLIP or energy/force modeling.",
        },
    )
    report.recommendations = generate_center_validation_recommendations(report)
    return report


def generate_center_validation_recommendations(
    report: LeakageValidationReport,
) -> list[str]:
    summary = report.metrics.get("center_label_issue_summary", {})
    by_code = summary.get("by_issue_code", {}) if isinstance(summary, dict) else {}
    recommendations: list[str] = []
    if by_code.get("control_has_center_atoms", 0):
        recommendations.append("Clean control/no-reaction records with non-empty centers.")
    if by_code.get("reactive_empty_center", 0):
        recommendations.append("Manually label missing centers for reactive mechanisms.")
    if report.new_split_distribution.get("test", 0) == 0:
        recommendations.append("Strict split has no test reactions; add more leakage groups.")
    if not report.leakage_groups_by_split.get("test"):
        recommendations.append("No leakage groups assigned to test; benchmark is unreliable.")
    test_mechanisms = report.mechanism_distribution_by_split.get("test", {})
    if len(test_mechanisms) <= 1:
        recommendations.append("Test split has narrow mechanism coverage; expand promoted data.")
    if not recommendations:
        recommendations.append("Proceed with strict-split center-head benchmark before MLIP work.")
    return recommendations


def analyze_split_performance_gap(
    benchmark_report: dict,
    validation_report: dict,
) -> dict[str, object]:
    """Explain validation/test reaction-center F1 gaps from split composition."""

    def _metric(split: str) -> float | None:
        value = benchmark_report.get(split, {})
        if isinstance(value, dict):
            raw = value.get("reaction_center_f1")
            return float(raw) if raw is not None else None
        return None

    val_f1 = _metric("val")
    test_f1 = _metric("test")
    f1_gap = None if val_f1 is None or test_f1 is None else round(abs(test_f1 - val_f1), 4)
    mechanisms = validation_report.get("mechanism_distribution_by_split", {})
    metrics = validation_report.get("metrics", {})
    split_diagnostics = metrics.get("split_diagnostics", {}) if isinstance(metrics, dict) else {}
    positives = split_diagnostics.get("positive_atom_counts_by_split", {})
    val_mechanisms = set(mechanisms.get("val", {})) if isinstance(mechanisms, dict) else set()
    test_mechanisms = set(mechanisms.get("test", {})) if isinstance(mechanisms, dict) else set()
    recommendations: list[str] = []
    possible_causes: list[str] = []
    if f1_gap is not None and f1_gap > 0.15:
        possible_causes.append("large val/test reaction-center F1 gap")
        recommendations.append(
            "Use val/test balanced template splits and inspect per-mechanism errors."
        )
    if val_mechanisms != test_mechanisms:
        possible_causes.append("different mechanism coverage between val and test")
        recommendations.append("Add or promote reactions so val and test share more mechanisms.")
    if isinstance(positives, dict) and positives.get("val", 0) != positives.get("test", 0):
        possible_causes.append("different positive atom counts between val and test")
        recommendations.append("Balance positive center atom counts across held-out splits.")
    if not recommendations:
        recommendations.append(
            "No obvious split-composition gap detected; inspect atom features and labels."
        )
    return {
        "val_reaction_center_f1": val_f1,
        "test_reaction_center_f1": test_f1,
        "f1_gap": f1_gap,
        "val_mechanisms": sorted(val_mechanisms),
        "test_mechanisms": sorted(test_mechanisms),
        "val_positive_atoms": positives.get("val") if isinstance(positives, dict) else None,
        "test_positive_atoms": positives.get("test") if isinstance(positives, dict) else None,
        "possible_causes": possible_causes,
        "recommendations": recommendations,
    }


def save_leakage_validation_report(
    report: LeakageValidationReport,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

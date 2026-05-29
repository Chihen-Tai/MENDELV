"""Add Michael addition training examples to the dataset."""
import json
from pathlib import Path

from mendel.curation import DraftLabelConfig, DraftReactionInput, draft_labeled_reaction
from mendel.identifier import identify_functional_groups
from mendel.labels import LabeledGroupRole
from mendel.parser import parse_reaction_smiles
from mendel.types import ReactionContext, Role

CORRECTIONS: dict[str, dict[str, Role]] = {
    "michael_malonate_mvk": {
        "mol0_ester_0":        Role.spectator,
        "mol0_ester_1":        Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_acetylacetone_acrylonitrile": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_carbonyl_1":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol0_alpha_carbon_2": Role.spectator,
    },
    "michael_cyclohexanone_mvk": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_nitromethane_mvk": {
        "mol0_nitro_0":        Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_diethylmalonate_acrylate": {
        "mol0_ester_0":  Role.spectator,
        "mol0_ester_1":  Role.spectator,
        "mol1_ester_0":  Role.spectator,
    },
    "michael_ethyl_acetoacetate_acrylonitrile": {
        "mol0_ester_0":        Role.spectator,
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
    },
    "michael_malononitrile_mvk": {
        "mol0_nitrile_0":      Role.spectator,
        "mol0_nitrile_1":      Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_malononitrile_acrylonitrile": {
        "mol0_nitrile_0":      Role.spectator,
        "mol0_nitrile_1":      Role.spectator,
        "mol1_nitrile_0":      Role.spectator,
    },
    "michael_acetylacetone_mvk": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_carbonyl_1":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol0_alpha_carbon_2": Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_nitromethane_acrylate": {
        "mol0_nitro_0":        Role.spectator,
        "mol1_ester_0":        Role.spectator,
    },
    "michael_dimethylmalonate_mvk": {
        "mol0_ester_0":        Role.spectator,
        "mol0_ester_1":        Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
    "michael_ethyl_acetoacetate_mvk": {
        "mol0_ester_0":        Role.spectator,
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol1_carbonyl_0":     Role.spectator,
    },
}

INPUTS = [
    DraftReactionInput(
        reaction_id="michael_malonate_mvk",
        reaction_smiles="CCOC(=O)CC(=O)OCC.C=CC(C)=O>>CCOC(=O)C(CCC(C)=O)C(=O)OCC",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_acetylacetone_acrylonitrile",
        reaction_smiles="CC(=O)CC(=O)C.C=CC#N>>CC(=O)C(CCC#N)C(=O)C",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_cyclohexanone_mvk",
        reaction_smiles="O=C1CCCCC1.C=CC(C)=O>>CC(=O)CCC1CCCCC1=O",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_nitromethane_mvk",
        reaction_smiles="[O-][N+](=O)C.C=CC(C)=O>>CC(=O)CCC[N+](=O)[O-]",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_diethylmalonate_acrylate",
        reaction_smiles="CCOC(=O)CC(=O)OCC.C=CC(=O)OCC>>CCOC(=O)C(CCC(=O)OCC)C(=O)OCC",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_ethyl_acetoacetate_acrylonitrile",
        reaction_smiles="CC(=O)CC(=O)OCC.C=CC#N>>CC(=O)C(CCC#N)C(=O)OCC",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_malononitrile_mvk",
        reaction_smiles="N#CCC#N.C=CC(C)=O>>CC(=O)CCC(C#N)C#N",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_malononitrile_acrylonitrile",
        reaction_smiles="N#CCC#N.C=CC#N>>N#CC(CCC#N)C#N",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_acetylacetone_mvk",
        reaction_smiles="CC(=O)CC(=O)C.C=CC(C)=O>>CC(=O)C(CCC(C)=O)C(=O)C",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_nitromethane_acrylate",
        reaction_smiles="[O-][N+](=O)C.C=CC(=O)OCC>>CCOC(=O)CCC[N+](=O)[O-]",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_dimethylmalonate_mvk",
        reaction_smiles="COC(=O)CC(=O)OC.C=CC(C)=O>>CC(=O)CCC(C(=O)OC)C(=O)OC",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
    DraftReactionInput(
        reaction_id="michael_ethyl_acetoacetate_mvk",
        reaction_smiles="CC(=O)CC(=O)OCC.C=CC(C)=O>>CC(=O)C(CCC(C)=O)C(=O)OCC",
        context=ReactionContext.ionic,
        mechanism_type="michael_addition",
    ),
]

DATA_PATH = Path("data/reactions.center_balanced.cleaned.json")

cfg = DraftLabelConfig(include_spectators=False)
new_records = []
for inp in INPUTS:
    r = draft_labeled_reaction(inp, cfg)
    corr = CORRECTIONS.get(r.reaction_id, {})

    for g in r.group_roles:
        if g.group_id in corr:
            g.role = corr[g.group_id]
        g.confidence = "manual"
        g.notes = "manually labeled"

    # Inject mol1_alkene_0 as reactive_electrophile — the rule-based predictor
    # assigns alkene spectator in ionic context so it gets filtered before correction.
    if "mol1_alkene_0" not in {g.group_id for g in r.group_roles}:
        rxn = parse_reaction_smiles(inp.reaction_smiles, context=inp.context)
        groups = identify_functional_groups(rxn)
        alkene = next((g for g in groups if g.group_id == "mol1_alkene_0"), None)
        if alkene is not None:
            r.group_roles.append(
                LabeledGroupRole(
                    group_id=alkene.group_id,
                    molecule_index=alkene.atom_refs[0].molecule_index,
                    group_type=alkene.group_type,
                    atom_indices=[ref.atom_index for ref in alkene.atom_refs],
                    role=Role.reactive_electrophile,
                    confidence="manual",
                    notes="Michael acceptor alkene (beta-carbon of conjugated system)",
                )
            )

    r.metadata["needs_manual_review"] = False
    r.metadata["mechanism"] = inp.mechanism_type
    new_records.append(r.to_dict())
    print(f"{r.reaction_id}")
    for g in r.group_roles:
        print(f"  {g.group_id:35s} {g.role.value}")

with open(DATA_PATH) as f:
    data = json.load(f)

existing_ids = {rec["reaction_id"] for rec in data["reactions"]}
added = 0
for rec in new_records:
    if rec["reaction_id"] not in existing_ids:
        data["reactions"].append(rec)
        added += 1

with open(DATA_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nAdded {added} reactions. Total: {len(data['reactions'])}")

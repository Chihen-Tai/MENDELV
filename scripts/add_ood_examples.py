"""Add reductive amination and hetero-DA training examples to the dataset."""
import json
from pathlib import Path

from mendel.curation import DraftLabelConfig, DraftReactionInput, draft_labeled_reaction
from mendel.types import ReactionContext, Role

CORRECTIONS: dict[str, dict[str, Role]] = {
    "reductive_amination_butanal_propylamine":       {"mol0_alpha_carbon_0": Role.spectator},
    "reductive_amination_benzaldehyde_ethylamine":   {},
    "reductive_amination_acetone_propylamine":       {"mol0_alpha_carbon_0": Role.spectator},
    "reductive_amination_cyclohexanone_methylamine": {"mol0_alpha_carbon_0": Role.spectator},
    "hetero_da_vinylether_acrolein": {
        "mol0_alkene_0": Role.reactive_nucleophile,
        "mol1_alkene_0": Role.reactive_electrophile,
        "mol1_carbonyl_0": Role.spectator,
    },
    "hetero_da_ethylvinylether_acrolein": {
        "mol0_alkene_0": Role.reactive_nucleophile,
        "mol1_alkene_0": Role.reactive_electrophile,
        "mol1_carbonyl_0": Role.spectator,
    },
    "hetero_da_vinylether_ethylacrylate": {
        "mol0_alkene_0": Role.reactive_nucleophile,
        "mol1_alkene_0": Role.reactive_electrophile,
        "mol1_ester_0":  Role.spectator,
    },
    "hetero_da_vinylether_acrylonitrile": {
        "mol0_alkene_0": Role.reactive_nucleophile,
        "mol1_alkene_0": Role.reactive_electrophile,
        "mol1_nitrile_0": Role.spectator,
    },
}

INPUTS = [
    DraftReactionInput(reaction_id="reductive_amination_butanal_propylamine",
        reaction_smiles="O=CCCC.NCCC>>C(NCCC)CCC", context=ReactionContext.ionic,
        mechanism_type="reductive_amination"),
    DraftReactionInput(reaction_id="reductive_amination_benzaldehyde_ethylamine",
        reaction_smiles="O=Cc1ccccc1.NCC>>C(NCC)c1ccccc1", context=ReactionContext.ionic,
        mechanism_type="reductive_amination"),
    DraftReactionInput(reaction_id="reductive_amination_acetone_propylamine",
        reaction_smiles="CC(=O)C.NCCC>>CC(NCCC)C", context=ReactionContext.ionic,
        mechanism_type="reductive_amination"),
    DraftReactionInput(reaction_id="reductive_amination_cyclohexanone_methylamine",
        reaction_smiles="O=C1CCCCC1.NC>>C1(NC)CCCCC1", context=ReactionContext.ionic,
        mechanism_type="reductive_amination"),
    DraftReactionInput(reaction_id="hetero_da_vinylether_acrolein",
        reaction_smiles="C=CO.C=CC=O>>C1CC=COC1", context=ReactionContext.pericyclic,
        mechanism_type="hetero_diels_alder"),
    DraftReactionInput(reaction_id="hetero_da_ethylvinylether_acrolein",
        reaction_smiles="CCOC=C.C=CC=O>>C1(CC)OCC=C1", context=ReactionContext.pericyclic,
        mechanism_type="hetero_diels_alder"),
    DraftReactionInput(reaction_id="hetero_da_vinylether_ethylacrylate",
        reaction_smiles="C=CO.C=CC(=O)OCC>>C1CC=COC1", context=ReactionContext.pericyclic,
        mechanism_type="hetero_diels_alder"),
    DraftReactionInput(reaction_id="hetero_da_vinylether_acrylonitrile",
        reaction_smiles="C=CO.C=CC#N>>C1CC=COC1", context=ReactionContext.pericyclic,
        mechanism_type="hetero_diels_alder"),
]

DATA_PATH = Path("data/reactions.center_balanced.cleaned.json")

cfg = DraftLabelConfig()
new_records = []
for inp in INPUTS:
    r = draft_labeled_reaction(inp, cfg)
    corr = CORRECTIONS.get(r.reaction_id, {})
    for g in r.group_roles:
        if g.group_id in corr:
            g.role = corr[g.group_id]
        g.confidence = "manual"
        g.notes = "manually labeled"
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

"""Add clean aldol training examples to the dataset.

Label convention (correct 2-molecule aldol):
  mol0_carbonyl_0      = spectator  (donor carbonyl activates alpha-H, not electrophile)
  mol0_alpha_carbon_0  = reactive_nucleophile
  mol1_carbonyl_0      = reactive_electrophile  (acceptor)
  other alpha_carbons  = spectator
"""
import json
from pathlib import Path

from mendel.curation import DraftLabelConfig, DraftReactionInput, draft_labeled_reaction
from mendel.types import ReactionContext, Role

CORRECTIONS: dict[str, dict[str, Role]] = {
    "aldol_acetaldehyde_formaldehyde": {
        "mol0_carbonyl_0":     Role.spectator,
    },
    "aldol_acetone_acetaldehyde_cross": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol1_alpha_carbon_0": Role.spectator,
    },
    "aldol_acetone_propanal_cross": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
        "mol1_alpha_carbon_0": Role.spectator,
    },
    "aldol_cyclohexanone_formaldehyde": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_alpha_carbon_1": Role.spectator,
    },
    "aldol_acetophenone_benzaldehyde": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol0_aromatic_0":     Role.spectator,
        "mol1_aromatic_0":     Role.spectator,
    },
    "aldol_butanal_self": {
        "mol0_carbonyl_0":     Role.spectator,
        "mol1_alpha_carbon_0": Role.spectator,
    },
}

INPUTS = [
    DraftReactionInput(
        reaction_id="aldol_acetaldehyde_formaldehyde",
        reaction_smiles="CC=O.C=O>>OCCC=O",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
    ),
    DraftReactionInput(
        reaction_id="aldol_acetone_acetaldehyde_cross",
        reaction_smiles="CC(=O)C.CC=O>>CC(=O)CC(O)C",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
    ),
    DraftReactionInput(
        reaction_id="aldol_acetone_propanal_cross",
        reaction_smiles="CC(=O)C.CCC=O>>CC(=O)CC(O)CC",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
    ),
    DraftReactionInput(
        reaction_id="aldol_cyclohexanone_formaldehyde",
        reaction_smiles="O=C1CCCCC1.C=O>>OCC1CCCCC1=O",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
    ),
    DraftReactionInput(
        reaction_id="aldol_acetophenone_benzaldehyde",
        reaction_smiles="CC(=O)c1ccccc1.O=Cc1ccccc1>>O=C(CC(O)c1ccccc1)c1ccccc1",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
    ),
    DraftReactionInput(
        reaction_id="aldol_butanal_self",
        reaction_smiles="CCCC=O.CCCC=O>>O=CC(CC)C(O)CCC",
        context=ReactionContext.ionic,
        mechanism_type="aldol",
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

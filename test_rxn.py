from mendel.negotiator import run_full_rule_pipeline
from mendel.parser import parse_reaction_smiles
from mendel.identifier import identify_functional_groups

tests = {
    "diels_alder_cyclopentadiene_ethylene": {
        "rxn": "[CH2:1]=[CH:2][CH:3]=[CH:4][CH2:5].[CH2:6]=[CH2:7]>>[CH2:1]1[CH:2]=[CH:3][CH:4][CH2:5][CH2:6][CH2:7]1",
        "context": "pericyclic",
    },
    "hetero_diels_alder_formaldehyde_butadiene": {
        "rxn": "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[O:6]>>[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][O:6]1",
        "context": "pericyclic",
    },
    "michael_addition_nitromethane_methyl_vinyl_ketone": {
        "rxn": "[CH3:1][N+:2](=O)[O-:3].[CH2:4]=[CH:5][C:6](=[O:7])[CH3:8]>>[CH2:1]([N+:2](=O)[O-:3])[CH2:4][CH2:5][C:6](=[O:7])[CH3:8]",
        "context": "ionic",
    },
    "minisci_like_pyridine_tertbutyl_radical": {
        "rxn": "[nH+:1]1[CH:2][CH:3][CH:4][CH:5][CH:6]1.[C:7]([CH3:8])([CH3:9])[CH3:10]>>[nH+:1]1[C:2]([C:7]([CH3:8])([CH3:9])[CH3:10])[CH:3][CH:4][CH:5][CH:6]1",
        "context": "radical",
    },
    "ugi_like_isocyanide_imine_addition": {
        "rxn": "[CH3:1][N:2]=[CH:3][CH3:4].[CH3:5][N+:6]#[C-:7]>>[CH3:1][N:2]([C:7]#[N+:6][CH3:5])[CH:3][CH3:4]",
        "context": "ionic",
    },
    "azide_alkyne_click_core": {
        "rxn": "[CH3:1][C:2]#[CH:3].[N-:4]=[N+:5]=[N:6][CH3:7]>>[CH3:1][C:2]1=[CH:3][N:4]=[N:5][N:6]1[CH3:7]",
        "context": "pericyclic",
    },
}

for name, item in tests.items():
    print("=" * 80)
    print(name)
    print("=" * 80)

    rxn = item["rxn"]
    context = item["context"]

    print("Detected functional groups:")
    parsed = parse_reaction_smiles(rxn)
    groups = identify_functional_groups(parsed)
    for g in groups:
        print(" ", g.group_id, g.group_type, [
              a.to_dict() for a in g.atom_refs], g.smarts)

    print()
    result = run_full_rule_pipeline(rxn, context=context)
    print("mechanism_hint:", result.mechanism_hint)

    for ra in result.assignments:
        print(" ", ra.group_id)
        print("   final_role:", getattr(ra, "final_role", None))
        print("   is_reaction_center:", getattr(
            ra, "is_reaction_center", None))
        print("   reason:", getattr(ra, "reason", None))
    print()

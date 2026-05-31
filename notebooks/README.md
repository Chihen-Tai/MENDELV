# MENDELV Notebooks

## `mendel_reaction_tester.ipynb`

Interactive tester for reaction SMILES through the MENDELV pipeline:
functional-group detection → rule-based role assignments → mechanism hint →
reaction center, plus optional MLP + negotiator predictions and lightweight
failure diagnosis.

> MENDELV predicts **functional-group reaction roles**, a mechanism hint, and a
> reaction center — it is **not** a full reaction-mechanism generator.

### Launch

```bash
conda activate mendel
pip install -e ".[dev]"
jupyter notebook notebooks/mendel_reaction_tester.ipynb
```

### To test the MLP pipeline

```bash
pip install -e ".[ml]"
```

Also confirm the checkpoint exists at `models/role_mlp.pt`. The MLP sections
skip gracefully when the model or the `[ml]` extra is missing — the rest of the
notebook still runs.

### Notes

- Run cells top to bottom. All errors are caught per reaction, so one bad SMILES
  never halts the notebook.
- Outputs are pandas DataFrames where possible; without pandas they fall back to
  plain lists of dicts.
- No network access, no model downloading, no model training, no repo files
  modified.
- Edit the **Interactive manual test** cell (`rxn` / `context` / `use_mlp`) to
  try your own reactions.

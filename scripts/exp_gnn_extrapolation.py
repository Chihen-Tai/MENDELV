"""EXPERIMENT (not part of the committed pipeline): does learned message
passing over the functional-group graph close the leave-one-mechanism-out
(LOMO) extrapolation gap that the static-feature MLP cannot?

Clean ablation: node features are the SAME 65-dim descriptor the MLP uses.
The only change is architecture — independent per-node MLP  ->  a small
relational GNN where each functional group exchanges messages with its
same-molecule and cross-molecule neighbours before predicting its role.

LOMO protocol mirrors scripts/.../lomo.py exactly:
  for each mechanism: train on the other 13, score role accuracy on the
  held-out (never-seen) mechanism's *labeled* groups.

Baseline MLP LOMO numbers are read from reports/lomo_extrapolation.json
(same data, same config) so we compare like-for-like without re-running it.

Decision rule (fixed in advance):
  GNN LOMO >> MLP LOMO on the collapse classes (diels_alder, michael,
  radical) -> architecture is the lever, promote it.
  GNN LOMO ~= MLP LOMO -> data is the binding constraint, not architecture.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from mendel.descriptor import build_descriptors
from mendel.identifier import identify_functional_groups
from mendel.labels import load_labeled_reactions
from mendel.mlp import ROLE_TO_INDEX, set_random_seed
from mendel.parser import parse_reaction_smiles

ROOT = Path("/Applications/codes/mendel")
DATA = ROOT / "data" / "reactions.center_balanced.cleaned.json"
MLP_LOMO = ROOT / "reports" / "lomo_extrapolation.json"

N_ROLES = 5
HIDDEN = 64
ROUNDS = 2
EPOCHS = 150
LR = 1e-3
DROPOUT = 0.10
SEED = 42

_MOL_RE = re.compile(r"^mol(\d+)_")


def _mol_index(group_id: str) -> int:
    m = _MOL_RE.match(group_id)
    return int(m.group(1)) if m else -1


class Graph:
    """One reaction as a functional-group graph (tensors precomputed)."""

    def __init__(self, x, a_intra, a_inter, label_idx, y, mechanism):
        self.x = x                # [N, 65]
        self.a_intra = a_intra    # [N, N] row-normalised mean adjacency (same mol)
        self.a_inter = a_inter    # [N, N] row-normalised mean adjacency (cross mol)
        self.label_idx = label_idx  # LongTensor of node rows that carry a role label
        self.y = y                # LongTensor[len(label_idx)] role indices
        self.mechanism = mechanism


def build_graphs() -> list[Graph]:
    reactions = load_labeled_reactions(DATA)
    graphs: list[Graph] = []
    for rxn in reactions:
        if rxn.metadata.get("needs_manual_review"):
            continue
        try:
            parsed = parse_reaction_smiles(rxn.reaction_smiles, context=rxn.context)
            groups = identify_functional_groups(parsed)
            descriptors = build_descriptors(parsed, groups)
        except Exception:
            continue
        if not descriptors:
            continue

        gid_order = [d.group_id for d in descriptors]
        row_of = {gid: i for i, gid in enumerate(gid_order)}
        mol_of = [_mol_index(gid) for gid in gid_order]
        n = len(gid_order)

        x = torch.tensor([d.as_vector() for d in descriptors], dtype=torch.float32)

        intra = torch.zeros((n, n), dtype=torch.float32)
        inter = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if mol_of[i] == mol_of[j]:
                    intra[i, j] = 1.0
                else:
                    inter[i, j] = 1.0
        # row-normalise to a mean aggregator (rows with no neighbour stay 0)
        intra = _row_normalise(intra)
        inter = _row_normalise(inter)

        label_rows: list[int] = []
        y_vals: list[int] = []
        for lgr in rxn.group_roles:
            if lgr.confidence == "draft":
                continue
            row = row_of.get(lgr.group_id)
            if row is None:
                continue
            label_rows.append(row)
            y_vals.append(ROLE_TO_INDEX[lgr.role])
        if not label_rows:
            continue

        graphs.append(
            Graph(
                x=x,
                a_intra=intra,
                a_inter=inter,
                label_idx=torch.tensor(label_rows, dtype=torch.long),
                y=torch.tensor(y_vals, dtype=torch.long),
                mechanism=rxn.mechanism_type,
            )
        )
    return graphs


def _row_normalise(a: torch.Tensor) -> torch.Tensor:
    deg = a.sum(dim=1, keepdim=True)
    deg = torch.where(deg > 0, deg, torch.ones_like(deg))
    return a / deg


class RelationalGNN(nn.Module):
    """Message passing with separate same-molecule / cross-molecule channels."""

    def __init__(self, in_dim: int, hidden: int, rounds: int, dropout: float):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.updates = nn.ModuleList(
            nn.Linear(hidden * 3, hidden) for _ in range(rounds)
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden, N_ROLES)

    def forward(self, x, a_intra, a_inter):
        h = torch.relu(self.embed(x))
        for upd in self.updates:
            intra_msg = a_intra @ h
            inter_msg = a_inter @ h
            m = torch.cat([h, intra_msg, inter_msg], dim=1)
            h = h + self.dropout(torch.relu(upd(m)))
        return self.readout(h)


def _class_weights(graphs: list[Graph]) -> torch.Tensor:
    counts = torch.zeros(N_ROLES)
    for g in graphs:
        for c in g.y.tolist():
            counts[c] += 1
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    w = counts.sum() / (N_ROLES * counts)
    return w


def train_eval(train_graphs: list[Graph], eval_graphs: list[Graph]) -> float:
    set_random_seed(SEED)
    model = RelationalGNN(in_dim=train_graphs[0].x.shape[1], hidden=HIDDEN,
                          rounds=ROUNDS, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train_graphs))

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits_all, y_all = [], []
        for g in train_graphs:
            logits = model(g.x, g.a_intra, g.a_inter)
            logits_all.append(logits[g.label_idx])
            y_all.append(g.y)
        loss = loss_fn(torch.cat(logits_all), torch.cat(y_all))
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for g in eval_graphs:
            logits = model(g.x, g.a_intra, g.a_inter)
            pred = logits[g.label_idx].argmax(dim=1)
            correct += int((pred == g.y).sum())
            total += int(g.y.numel())
    return correct / total if total else 0.0


def main() -> None:
    graphs = build_graphs()
    by_mech: dict[str, list[Graph]] = defaultdict(list)
    for g in graphs:
        by_mech[g.mechanism].append(g)

    mlp_lomo = json.loads(MLP_LOMO.read_text()) if MLP_LOMO.exists() else {}

    rows = []
    for mech in sorted(by_mech):
        held = by_mech[mech]
        train = [g for g in graphs if g.mechanism != mech]
        n_grp = sum(int(g.y.numel()) for g in held)
        gnn_acc = train_eval(train, held)
        mlp_acc = mlp_lomo.get(mech, {}).get("lomo_accuracy")
        rows.append((mech, n_grp, mlp_acc, gnn_acc))

    print(f"\n{'mechanism':30s} {'n_grp':>5s} {'MLP_LOMO':>9s} "
          f"{'GNN_LOMO':>9s} {'delta':>7s}")
    print("-" * 64)
    gnn_vals, mlp_vals = [], []
    for mech, n_grp, mlp_acc, gnn_acc in rows:
        gnn_vals.append(gnn_acc)
        d = ""
        if mlp_acc is not None:
            mlp_vals.append(mlp_acc)
            diff = gnn_acc - mlp_acc
            d = f"{diff:+.3f}"
            mlp_s = f"{mlp_acc:.3f}"
        else:
            mlp_s = "  -  "
        print(f"{mech:30s} {n_grp:5d} {mlp_s:>9s} {gnn_acc:9.3f} {d:>7s}")
    print("-" * 64)
    print(f"GNN LOMO mean: {sum(gnn_vals) / len(gnn_vals):.3f}")
    if mlp_vals:
        print(f"MLP LOMO mean (committed): {sum(mlp_vals) / len(mlp_vals):.3f}")

    out = ROOT / "reports" / "gnn_lomo_extrapolation.json"
    out.write_text(json.dumps(
        {m: {"n_grp": n, "mlp_lomo": a, "gnn_lomo": g} for m, n, a, g in rows},
        indent=2,
    ))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

# Functional Group SMARTS Patterns

Priority order: lower number = matched first. Specific patterns come before general ones.

| Priority | Group Type | SMARTS | Notes |
|----------|------------|--------|-------|
| 0 | `aromatic` | *(ring_detection)* | Detected via `GetRingInfo()`, not SMARTS |
| 1 | `carboxylic_acid` | `[CX3](=O)[OX2H]` | Matched before `carbonyl` |
| 2 | `ester` | `[CX3](=O)[OX2][CX4]` | Matched before `carbonyl` and `ether` |
| 3 | `amide` | `[CX3](=O)[NX3]` | Matched before `carbonyl` and `amine` |
| 4 | `carbonyl` | `[CX3;!$(C(=O)[OX2]);!$(C(=O)[NX3])]=[OX1]` | Residual: ketone/aldehyde only |
| 5 | `phenol` | `c[OX2H]` | Aromatic C–OH; matched before `alcohol` |
| 6 | `alcohol` | `[CX4][OX2H]` | Aliphatic OH only |
| 7 | `ether` | `[CX4][OX2][CX4]` | Both flanking C must be sp3 |
| 8 | `nitro` | `[$([NX3](=O)=O),$([N+](=O)[O-])]` | Supports both neutral and charge-separated forms |
| 9 | `nitrile` | `[CX2]#[NX1]` | C≡N triple bond |
| 10 | `halide` | `[CX4][F,Cl,Br,I]` | sp3 C–X only; aryl halides are out of scope |
| 11 | `amine` | `[NX3;!$(NC=O);!$([NX3+])]` | Excludes amide N and quaternary N |
| 12 | `alkene` | `[CX3]=[CX3]` | C=C double bond |
| 13 | `alkyne` | `[CX2]#[CX2]` | C≡C triple bond |
| 14 | `alpha_carbon` | `[CX4;H1,H2,H3][$([CX3]=O),$([CX2]#[NX1]),$([NX3](=O)=O),$([N+](=O)[O-])]` | **Contextual — second pass** |
| 15 | `benzylic_site` | `[CX4;H1,H2,H3][c]` | **Contextual — second pass** |

## Notes

- **nitro** supports both neutral `[NX3](=O)=O` and charge-separated `[N+](=O)[O-]` forms via SMARTS alternation.
- **aryl halides** are out of scope. The halide pattern requires an sp3 carbon (`[CX4]`), so Ar–X bonds are intentionally excluded.
- **alpha_carbon** and **benzylic_site** are contextual groups detected in a second pass after all primary groups. They can be disabled with `include_contextual=False`.
- **carbonyl** uses restrictive SMARTS (`!$(C(=O)[OX2])`, `!$(C(=O)[NX3])`) to prevent double-counting with carboxylic_acid, ester, and amide.
- **phenol** uses lowercase `c` (aromatic carbon) to prevent overlap with `alcohol` (which requires sp3 `[CX4]`).

## Out of Scope (Phase 2)

`acid_chloride`, `epoxide`, `anhydride`, `allylic_site`, `aryl_halide` — not implemented.

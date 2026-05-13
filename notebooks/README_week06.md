# Week 6 — Local Atomic Environments (setup)

Files in this folder:

| File                              | Purpose                                                  |
|-----------------------------------|----------------------------------------------------------|
| `week06_local_atomic_envs.ipynb`  | The 4-part exercise notebook (Magpie → struct → SOAP → audit). |
| `spinels_mp.csv`                  | Pre-cached Materials Project subset, 220 AB$_2$O$_4$ spinels. |
| `README_week06.md`                | This file.                                               |

## What `spinels_mp.csv` contains

220 entries filtered from `matminer.datasets.matbench_mp_gap` (~106k MP
structures) to **AB$_2$O$_4$ stoichiometry** (any two metals + oxygen, 1:2:4
ratio after composition reduction), then each structure reduced to its
primitive cell (`SpacegroupAnalyzer`, `symprec=0.1`) and capped at
≤ 30 sites for SOAP-runtime sanity. The split design rewards the
polymorph-leakage audit in Part 4:

* 220 rows
* 115 distinct reduced formulas
* **105 formulas appear with two polymorphs** (different space groups,
  different DFT band gaps)
* Top space groups: 227 (cubic Fd-3m spinel), 62, 74, 63, 141

Columns:

```
material_id, pretty_formula, structure (JSON-dict of pymatgen Structure),
band_gap (PBE, eV), formation_energy_per_atom (NaN — not in matbench_mp_gap),
theoretical, spacegroup
```

Target for the exercise is `band_gap`. (Formation energy is left as a
column placeholder for a forward link to Unit 8.)

## Install

The notebook needs `numpy`, `pandas`, `matplotlib`, `scikit-learn`,
`pymatgen`, `matminer`, and `dscribe`. With **uv**:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy pandas matplotlib scikit-learn pymatgen matminer dscribe jupyter
```

Or with **conda**:

```bash
conda create -n mg-w6 python=3.12 -y
conda activate mg-w6
pip install matminer dscribe pymatgen jupyter scikit-learn
```

Verify in one command:

```bash
python -c "import matminer, dscribe, pymatgen; print('ok')"
```

Note: dscribe wheels ship with their own C/C++ extension and currently
build cleanly on Linux/macOS with Python 3.10–3.12. On Windows, prefer
WSL2.

## Running

```bash
cd /path/to/MaterialsGenomics/notebooks
jupyter lab week06_local_atomic_envs.ipynb
```

Or headless end-to-end:

```bash
jupyter nbconvert --to notebook --execute week06_local_atomic_envs.ipynb \
  --output week06_local_atomic_envs.ipynb \
  --ExecutePreprocessor.timeout=600
```

Total runtime on a 2022-class laptop CPU: ≈ 2–3 minutes. Most of the
budget is in Part 2 (`SiteStatsFingerprint` + `BondFractions` on
220 structures, ~80 s combined). SOAP itself is ~1 s thanks to the small
220-row dataset and `average="inner"`.

## Pedagogical notes (for the TA)

* The exercise deliberately makes the leakage point land: with 105
  polymorph-formulas, the random-split MAE on Tier 1 is materially
  better than the grouped-split MAE — a clean teaching moment for
  Part 4.
* If Tier 2 / Tier 3 do **not** beat Tier 1 on this dataset, that is
  *not* a bug. Spinels are composition-dominated (narrow chemistry),
  and the lecture explicitly anticipates this (slide 10 anti-hype
  frame).
* The `material_id` field is a stable index into matbench, not a real
  MP-API `mp-XXXXX` id. We did not require an MP-API key for build.

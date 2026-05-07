# OpenMM MD framework

Small workflow for running OpenMM simulations starting from Amber `prmtop` and `inpcrd/rst7` files.

The cleaned workflow has three scripts:

| Script | Purpose |
|---|---|
| `openmm_prep.py` | Builds `system.xml`, optionally adds Amber-mask positional restraints, minimizes, and writes `sys_min.xml`. |
| `openmm_md.py` | Runs one MD stage from `system.xml` plus an input state, writing a new restart/state XML, DCD, log, and checkpoint. |
| `openmm_generate_script.py` | Generates a resumable `run_simulation.sh` for minimization, NVT, NPT, and segmented production MD. |

## Requirements

Install in a clean conda/mamba environment:

```bash
mamba create -n openmm-md -c conda-forge python=3.11 openmm parmed mdtraj ambertools
mamba activate openmm-md
```

`parmed` is required only if you use Amber-mask restraints such as `!:WAT&!@H=`.

## Quick start

From a directory containing `complex_solvated.prmtop`, `complex_solvated.inpcrd`, and a reference PDB with matching atom order:

```bash
python openmm_generate_script.py \
  -i complex_solvated.inpcrd \
  -t complex_solvated.prmtop \
  --reference prot_amber.pdb \
  --restrain-mask '!:WAT&!@H=' \
  minim nvt npt md

bash run_simulation.sh
```

Default times are:

- NVT: 1 ns
- NPT: 2 ns
- Production: 10 ns split into 5 segments
- timestep: 2 fs
- temperature: 298.15 K
- Langevin friction: 1 ps⁻¹

## Longer production example

```bash
python openmm_generate_script.py \
  -i complex_solvated.inpcrd \
  -t complex_solvated.prmtop \
  --reference prot_amber.pdb \
  --restrain-mask '!:WAT&!@H=' \
  minim nvt npt md \
  --nvt-time 2 \
  --npt-time 5 \
  --md-time 100 \
  --prod-segments 20 \
  --dt 2.0 \
  --temp 310 \
  --platform CUDA --cuda 0

bash run_simulation.sh
```

## Running stages manually

### 1. Prepare and minimize

```bash
python openmm_prep.py \
  -t complex_solvated.prmtop \
  -i complex_solvated.inpcrd \
  --system-xml system.xml \
  --min-state sys_min.xml \
  --longrange PME \
  --cutoff 10.0 \
  --shake \
  --restrain-mask '!:WAT&!@H=' \
  --reference prot_amber.pdb \
  -k 10.0
```

### 2. NVT

```bash
python openmm_md.py \
  --xml system.xml \
  -t complex_solvated.prmtop \
  -i complex_solvated.inpcrd \
  -s sys_min.xml \
  --restart sys_NVT.xml \
  -x sys_NVT.dcd \
  -r sys_NVT.log \
  --chk sys_NVT.chk \
  --temp 298.15 \
  --gamma-ln 1.0 \
  --dt 2.0 \
  -n 500000
```

### 3. NPT

```bash
python openmm_md.py \
  --xml system.xml \
  -t complex_solvated.prmtop \
  -s sys_NVT.xml \
  --restart sys_NPT.xml \
  -x sys_NPT.dcd \
  -r sys_NPT.log \
  --chk sys_NPT.chk \
  --npt --pressure 1.0
```

### 4. Production segment

```bash
python openmm_md.py \
  --xml system.xml \
  -t complex_solvated.prmtop \
  -s sys_NPT.xml \
  --restart sys_md_1.xml \
  -x sys_md_1.dcd \
  -r sys_md_1.log \
  --chk sys_md_1.chk \
  --npt \
  -n 1000000
```

## Notes and conventions

- `system.xml` stores the force-field system definition. It is reused for all stages.
- `sys_min.xml`, `sys_NVT.xml`, `sys_NPT.xml`, and `sys_md_*.xml` are serialized OpenMM `State` files and are used as restart files.
- The generated `run_simulation.sh` is resumable: a stage is skipped if its target restart XML already exists.
- Positional restraints are currently added in `openmm_prep_clean.py`, so they become part of `system.xml`. This means the same restrained system is used in later stages. For unrestrained production, generate a second unrestrained `system.xml` or extend the scripts to scale/remove the restraint force after equilibration.
- The reference file used for restraints must have the same atom order as the Amber topology.



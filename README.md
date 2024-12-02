# openmm_md_framework
 Framework to automate openmm runs starting from amber parameters and coordinates: prmtop and inpcrd
 
 Requirements:

* openmm
* ambertools
* parmed
* mdtraj


python openmm_generate_script.py -i complex_solvated.inpcrd -t complex_solvated.prmtop minim nvt npt md

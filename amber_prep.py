#!/usr/bin/env python3

# Example: python amber_prep.py --protein protein.pdb --hits hits_folder --n_cat 2 --n_ani 2 \
# --buffer 12.0 --forcefield leaprc.protein.ff19SB

import os
import glob
import shutil
import subprocess
import argparse


def convert_to_mol2(input_file):
    mol2_file = os.path.splitext(input_file)[0] + ".mol2"
    if input_file.endswith(".mol2"):
        return input_file
    print(f"Converting {input_file} to {mol2_file} using Open Babel...")
    try:
        subprocess.run(
            f"obabel -i {input_file.split('.')[-1]} {input_file} -o mol2 -O {mol2_file} --gen3d",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting {input_file} to MOL2: {e}")
    return mol2_file


def process_hits_or_single_ligand(
    hits_folder, single_lig_file, receptor_file, n_cat, n_ani, buffer, forcefield
):
    """Process a folder of hits or a single ligand."""
    if hits_folder:
        if not os.path.isdir(hits_folder):
            raise FileNotFoundError(f"Hits folder '{hits_folder}' does not exist.")
        all_ligands = glob.glob(
            os.path.join(hits_folder, "*.[mp][od][b2]")
        ) + glob.glob(os.path.join(hits_folder, "*.sdf"))
        if not all_ligands:
            raise FileNotFoundError(f"No ligand files found in '{hits_folder}'.")
        print(
            f"Found {len(all_ligands)} ligand files in '{hits_folder}': {all_ligands}"
        )

        for lig in all_ligands:
            mol2_file = convert_to_mol2(lig)
            amber_files = prepare_dock(mol2_file)
            create_complex_and_ligand_files(
                amber_files, receptor_file, n_cat, n_ani, buffer, forcefield, hits=True
            )

    elif single_lig_file:
        if not os.path.isfile(single_lig_file):
            raise FileNotFoundError(f"Ligand file '{single_lig_file}' does not exist.")
        mol2_file = convert_to_mol2(single_lig_file)
        amber_files = prepare_dock(mol2_file)
        create_complex_and_ligand_files(
            amber_files, receptor_file, n_cat, n_ani, buffer, forcefield, hits=True
        )

    else:
        # Receptor only
        create_complex_and_ligand_files(
            None, receptor_file, n_cat, n_ani, buffer, forcefield
        )


def add_tripos_tag(mol2_file):
    """Adds the <@TRIPOS> tag at the end of each MOL2 file."""
    print(f"Adding <@TRIPOS> tag to {mol2_file}")
    with open(mol2_file, "a") as f:
        f.write("<@TRIPOS> TAG\n")


def prepare_dock(mol2_file):
    """Processes the input MOL2 file and prepares it for docking using Amber tools."""
    base_name = os.path.splitext(os.path.basename(mol2_file))[0]
    frcmod_file = f"{base_name}.frcmod"
    amber_file = f"{base_name}.amber.mol2"

    # Add <@TRIPOS> tag
    add_tripos_tag(mol2_file)

    print(f"Running antechamber for {mol2_file}...")
    try:
        subprocess.run(
            f"antechamber -i {mol2_file} -fi mol2 -o {amber_file} -fo mol2 -c bcc -s 2 -at gaff -pf y",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running antechamber: {e}")

    print(f"Running parmchk2 for {amber_file}...")
    try:
        subprocess.run(
            f"parmchk2 -i {amber_file} -f mol2 -o {frcmod_file}",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running parmchk2: {e}")

    return amber_file, frcmod_file


def create_complex_and_ligand_files(
    amber_files, receptor_file, n_cat, n_ani, buffer, forcefield, hits=None
):
    """
    Generates ligand and complex files for Amber.
    If hits are None, prepares only the receptor with solvation.
    """
    if hits:
        amber_file, frcmod_file = amber_files
        base_name = os.path.splitext(os.path.basename(amber_file))[0]
        output_dir = f"{base_name}_prep"
        os.makedirs(output_dir, exist_ok=True)

        # Copy files to the working directory
        shutil.copy(amber_file, os.path.join(output_dir, "ligand.mol2"))
        shutil.copy(receptor_file, os.path.join(output_dir, "receptor.pdb"))
        shutil.copy(frcmod_file, os.path.join(output_dir, frcmod_file))
    else:
        # Protein-only preparation
        base_name = os.path.splitext(os.path.basename(receptor_file))[0]
        output_dir = f"{base_name}_prep"
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(receptor_file, os.path.join(output_dir, "receptor.pdb"))

    # Run tleap to generate Amber files
    os.chdir(output_dir)
    try:
        with open("leap.in", "w") as f:
            f.write(f"source {forcefield}\n")
            f.write(f"source leaprc.gaff\n")
            f.write(f"source leaprc.water.tip3p\n")
            if hits:
                f.write(f"loadamberparams {frcmod_file}\n")
                f.write(f"ligand = loadmol2 ligand.mol2\n")
            f.write(f"receptor = loadpdb receptor.pdb\n")
            if hits:
                f.write(f"complex = combine {{ receptor ligand }}\n")
                f.write(f"check complex")
                f.write(f"savepdb complex complex.prmtop complex.inpcrd\n")
                f.write(f"solvatebox complex TIP3PBOX {{{buffer} {buffer} {buffer}}}\n")
                f.write(f"addionsrand complex Na+ {n_cat}\n")
                f.write(f"addionsrand complex Cl- {n_ani}\n")
            else:
                f.write(f"check receptor\n")
                f.write(
                    f"solvatebox receptor TIP3PBOX {{{buffer} {buffer} {buffer}}}\n"
                )
                f.write(f"addionsrand receptor Na+ {n_cat}\n")
                f.write(f"addionsrand receptor Cl- {n_ani}\n")
            if hits:
                f.write(
                    f"saveamberparm complex complex_solvated.prmtop complex_solvated.inpcrd\n"
                )
                f.write(f"savepdb complex complex_solvated.pdb\n")
            else:
                f.write(
                    f"saveamberparm receptor receptor_solvated.prmtop receptor_solvated.inpcrd\n"
                )
                f.write(f"savepdb receptor receptor_solvated.pdb\n")
            f.write("quit\n")

        print(f"Running tleap for {base_name}...")
        subprocess.run("tleap -f leap.in", shell=True, check=True)

        # Validate output files
        output_files = (
            [
                f"complex_solvated.prmtop",
                f"complex_solvated.inpcrd",
            ]
            if hits
            else [
                f"receptor_solvated.prmtop",
                f"receptor_solvated.inpcrd",
            ]
        )

        for file in output_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Expected output file {file} not found.")
        print(f"Amber files successfully generated for {base_name}.")
    finally:
        os.chdir("..")


def process_hits(hits_folder, receptor_file, n_cat, n_ani, buffer, forcefield):
    """Processes ligands and prepares docking files."""
    if hits_folder is None:
        # Solvate only the receptor
        create_complex_and_ligand_files(
            None, receptor_file, n_cat, n_ani, buffer, forcefield
        )
    else:
        if not os.path.isdir(hits_folder):
            raise FileNotFoundError(f"Hits folder '{hits_folder}' does not exist.")

        mol2_files = glob.glob(os.path.join(hits_folder, "*.mol2"))
        if not mol2_files:
            raise FileNotFoundError(
                f"No .mol2 files found in the hits folder '{hits_folder}'."
            )

        print(f"Found {len(mol2_files)} .mol2 files in '{hits_folder}': {mol2_files}")

        for mol2_file in mol2_files:
            amber_files = prepare_dock(mol2_file)
            create_complex_and_ligand_files(
                amber_files, receptor_file, n_cat, n_ani, buffer, forcefield, hits=True
            )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ligands and receptor for docking with Amber."
    )
    parser.add_argument(
        "-p", "--protein", required=True, help="Path to the receptor PDB file."
    )
    parser.add_argument(
        "-f", "--hits", help="Path to folder containing ligand files (.pdb/.mol2/.sdf)"
    )
    parser.add_argument("-l", "--lig", help="Single ligand file (.pdb/.mol2/.sdf)")
    parser.add_argument(
        "--n_cat", type=int, default=0, help="Number of cations to add (default: 0)."
    )
    parser.add_argument(
        "--n_ani", type=int, default=0, help="Number of anions to add (default: 0)."
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=12.0,
        help="Buffer distance for solvation (default: 12.0).",
    )
    parser.add_argument(
        "--forcefield",
        "-ff",
        default="leaprc.protein.ff14SB",
        help="Amber forcefield to use (default: ff14SB).",
    )

    args = parser.parse_args()

    if args.hits and args.lig:
        parser.error("Please provide either --hits or --lig, not both.")
    process_hits_or_single_ligand(
        args.hits,
        args.lig,
        args.protein,
        args.n_cat,
        args.n_ani,
        args.buffer,
        args.forcefield,
    )


if __name__ == "__main__":
    main()

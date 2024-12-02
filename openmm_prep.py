#!/usr/bin/env python3
# title           :openmm_prep.py
# description     :Prepare and serialize an OpenMM system from Amber topology and coordinates.
# date            :2024-11-20
# python_version  :3.x
# usage           :python openmm_prep.py -h
# ==============================================================================

import os
import math
from argparse import ArgumentParser

import parmed as pmd
from parmed import unit as u
import openmm as mm
import openmm.app as app
from openmm.app.internal.unitcell import computeLengthsAndAngles
from collections import defaultdict


def perform_minimization(system, positions, topology, step_number):
    """
    Perform energy minimization on the given system.
    Args:
        system: OpenMM System object.
        positions: Initial positions of the system.
        topology: Topology of the system.
        step_number: Current minimization cycle number.
    """
    integrator = mm.VerletIntegrator(0.001)  # Dummy integrator for minimization
    platform = mm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)

    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    # Get minimized energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    print(f"Step {step_number}: Minimized energy = {energy:.4f} kcal/mol")

    return state.getPositions()


def main():
    # Command-line arguments
    parser = ArgumentParser(
        description="Prepare and serialize an OpenMM system from Amber files or PDB+forcefield."
    )

    input_group = parser.add_argument_group("Input Files")
    input_group.add_argument(
        "-p",
        "--pdb",
        metavar="<PDB FILE>",
        help="PDB file for the target system (optional if using Amber files).",
    )
    input_group.add_argument(
        "-t",
        "--topology",
        metavar="<TOP FILE>",
        help="Amber topology file (required for Amber inputs).",
    )
    input_group.add_argument(
        "-i",
        "--inpcrd",
        metavar="<CRD FILE>",
        help="Amber coordinate file (required for Amber inputs).",
    )
    input_group.add_argument(
        "-f",
        "--forcefield",
        metavar="<FORCEFIELD FILE>",
        nargs="+",
        default=["amoeba2013"],
        help="Forcefield XML files (required for PDB inputs). Default: amoeba2013.",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-s",
        "--system-xml",
        metavar="OUTPUT FILE",
        default="system.xml",
        help="Output OpenMM system XML file. Default: system.xml.",
    )

    simulation_group = parser.add_argument_group("Simulation Settings")
    simulation_group.add_argument(
        "--cuda",
        metavar="CUDA DEVICE",
        default="0",
        help="CUDA device index to use (default: 0).",
    )
    simulation_group.add_argument(
        "--shake",
        action="store_true",
        default=False,
        help="Apply SHAKE constraints to hydrogen bonds. Default: off.",
    )
    simulation_group.add_argument(
        "-l",
        "--longrange",
        choices=["Ewald", "PME", "NoCutoff"],
        default="PME",
        help="Method for long-range electrostatics (default: PME).",
    )
    simulation_group.add_argument(
        "-c",
        "--cutoff",
        metavar="CUTOFF",
        type=float,
        default=10.0,
        help="Non-bonded interaction cutoff distance in Angstroms (default: 10.0).",
    )
    simulation_group.add_argument(
        "-e",
        "--ewaldTolerance",
        metavar="TOLERANCE",
        type=float,
        default=5e-4,
        help="Ewald error tolerance for PME (default: 5e-4).",
    )
    simulation_group.add_argument(
        "-v",
        "--vdw-cutoff",
        metavar="CUTOFF",
        type=float,
        help="Cutoff for van der Waals interactions (AMOEBA forcefields only).",
    )
    simulation_group.add_argument(
        "--epsilon",
        metavar="EPSILON",
        type=float,
        help="Convergence criteria for polarizable dipoles (AMOEBA only).",
    )

    restraint_group = parser.add_argument_group("Restraints and Repulsion")
    restraint_group.add_argument(
        "--reference",
        metavar="<PDB FILE>",
        help="Reference PDB for positional restraints.",
    )
    restraint_group.add_argument(
        "--restrain-mask",
        metavar="<MASK>",
        help="Amber mask for positional restraints.",
    )
    restraint_group.add_argument(
        "--repulsion-mask",
        metavar="<MASK>",
        help="Amber mask for repulsion restraints.",
    )
    restraint_group.add_argument(
        "-k",
        "--force-constant",
        metavar="FORCE CONSTANT",
        type=float,
        default=10.0,
        help="Force constant for positional restraints (default: 10.0 kcal/mol/A^2).",
    )

    opt = parser.parse_args()

    # Determine non-bonded method
    longrange_method = {"PME": app.PME, "Ewald": app.Ewald, "NoCutoff": app.NoCutoff}[
        opt.longrange
    ]

    constraints = app.HBonds if opt.shake else None

    # Load system
    if opt.topology and opt.inpcrd:
        print(f"Loading Amber files: {opt.topology}, {opt.inpcrd}")
        prmtop = app.AmberPrmtopFile(opt.topology)
        inpcrd = app.AmberInpcrdFile(opt.inpcrd)
        system = prmtop.createSystem(
            nonbondedMethod=longrange_method,
            nonbondedCutoff=opt.cutoff * u.angstroms,
            rigidWater=opt.shake,
            constraints=constraints,
            ewaldErrorTolerance=opt.ewaldTolerance,
        )
        pos = inpcrd.positions
        top = prmtop.topology

    elif opt.pdb and opt.forcefield:
        print(f"Loading PDB file: {opt.pdb}")
        pdb = app.PDBFile(opt.pdb)
        forcefields = [
            f"{ff}.xml" if not ff.endswith(".xml") else ff for ff in opt.forcefield
        ]
        ff = app.ForceField(*forcefields)
        kwargs = {
            "nonbondedMethod": longrange_method,
            "nonbondedCutoff": opt.cutoff * u.angstroms,
            "rigidWater": opt.shake,
            "constraints": constraints,
            "ewaldErrorTolerance": opt.ewaldTolerance,
        }
        if opt.epsilon:
            kwargs.update(
                {"polarization": "mutual", "mutualInducedTargetEpsilon": opt.epsilon}
            )
        system = ff.createSystem(pdb.topology, **kwargs)
        pos = pdb.positions
        top = pdb.topology
    else:
        raise ValueError(
            "Specify either Amber files (-t and -i) or a PDB file (-p) with forcefield (-f)."
        )

    # Apply van der Waals cutoff for AMOEBA
    if opt.vdw_cutoff:
        for force in system.getForces():
            if isinstance(force, mm.AmoebaVdwForce):
                print(f"Setting van der Waals cutoff to {opt.vdw_cutoff} Å.")
                force.setCutoff(opt.vdw_cutoff * u.angstroms)

    # Serialize the system
    print(f"Writing system XML to {opt.system_xml}")
    with open(opt.system_xml, "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    print("System serialized successfully.")

    # Perform three minimization cycles
    print("Starting minimization cycles...")
    for i in range(1, 4):
        pos = perform_minimization(system, pos, top, i)

    # Save the final minimized state as sys_min.xml
    print("Saving the final minimized state to sys_min.xml...")
    integrator = mm.VerletIntegrator(0.001)  # Dummy integrator for state creation
    platform = mm.Platform.getPlatformByName("CPU")  # Platform for state creation
    simulation = app.Simulation(top, system, integrator, platform)
    simulation.context.setPositions(pos)
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getForces=True, getEnergy=True
    )

    with open("sys_min.xml", "w") as f:
        f.write(mm.XmlSerializer.serialize(state))

        print("Final minimized state saved as sys_min.xml.")


if __name__ == "__main__":
    main()

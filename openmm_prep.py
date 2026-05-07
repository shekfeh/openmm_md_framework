#!/usr/bin/env python3
"""Prepare and minimize an OpenMM system from Amber files or PDB+OpenMM force fields.

Typical Amber usage:
    python openmm_prep.py -t complex.prmtop -i complex.inpcrd \
        --system-xml system.xml --min-state sys_min.xml \
        --longrange PME --cutoff 10.0 --shake \
        --restrain-mask '!:WAT&!@H=' --reference prot_amber.pdb -k 10.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import openmm as mm
import openmm.app as app
from openmm import unit as u

try:
    import parmed as pmd
except ImportError:  # pragma: no cover
    pmd = None

LOGGER = logging.getLogger("openmm_prep")

KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2 = 418.4


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an OpenMM System XML and minimized State XML."
    )

    input_group = parser.add_argument_group("Input files")
    input_group.add_argument("-t", "--topology", help="Amber topology/prmtop file.")
    input_group.add_argument("-i", "--inpcrd", help="Amber coordinate/inpcrd file.")
    input_group.add_argument("-p", "--pdb", help="PDB file for PDB+ForceField mode.")
    input_group.add_argument(
        "-f",
        "--forcefield",
        nargs="+",
        default=None,
        help="OpenMM force-field XML files for PDB mode, e.g. amber14-all.xml amber14/tip3pfb.xml.",
    )

    output_group = parser.add_argument_group("Output files")
    output_group.add_argument(
        "--system-xml",
        default="system.xml",
        help="Output serialized OpenMM System XML.",
    )
    output_group.add_argument(
        "--min-state", default="sys_min.xml", help="Output minimized State XML."
    )

    sim_group = parser.add_argument_group("System settings")
    sim_group.add_argument(
        "-l",
        "--longrange",
        choices=["PME", "Ewald", "NoCutoff"],
        default="PME",
        help="Long-range electrostatics method.",
    )
    sim_group.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=10.0,
        help="Nonbonded cutoff in Angstrom.",
    )
    sim_group.add_argument(
        "-e",
        "--ewald-tolerance",
        type=float,
        default=5e-4,
        help="PME/Ewald error tolerance.",
    )
    sim_group.add_argument(
        "--shake",
        action="store_true",
        help="Constrain bonds involving hydrogen and use rigid water.",
    )
    sim_group.add_argument(
        "--min-cycles", type=int, default=3, help="Number of minimization cycles."
    )
    sim_group.add_argument(
        "--min-tolerance",
        type=float,
        default=10.0,
        help="Minimization tolerance in kJ/mol/nm.",
    )
    sim_group.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Max minimizer iterations per cycle; 0 lets OpenMM choose.",
    )
    sim_group.add_argument(
        "--platform",
        choices=["CUDA", "OpenCL", "CPU"],
        default="CPU",
        help="Platform used for minimization.",
    )
    sim_group.add_argument("--cuda", default="0", help="CUDA device index.")
    sim_group.add_argument("--opencl", default="0", help="OpenCL device index.")

    restraint_group = parser.add_argument_group("Optional positional restraints")
    restraint_group.add_argument(
        "--restrain-mask",
        help="Amber mask for restrained atoms, e.g. '!:WAT&!@H='. Requires Amber topology.",
    )
    restraint_group.add_argument(
        "--reference",
        help="Reference PDB or coordinate file with atom order matching the system.",
    )
    restraint_group.add_argument(
        "-k",
        "--force-constant",
        type=float,
        default=10.0,
        help="Positional restraint force constant in kcal/mol/A^2.",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def get_platform(name: str, cuda: str = "0", opencl: str = "0"):
    platform = mm.Platform.getPlatformByName(name)
    if name == "CUDA":
        return platform, {"CudaPrecision": "mixed", "CudaDeviceIndex": cuda}
    if name == "OpenCL":
        return platform, {"OpenCLDeviceIndex": opencl}
    return platform, {}


def load_reference_positions(path: str):
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return app.PDBFile(path).positions
    if suffix in {".rst7", ".inpcrd", ".crd"}:
        return app.AmberInpcrdFile(path).positions
    raise ValueError("Reference must be PDB/ENT or Amber inpcrd/rst7.")


def amber_mask_indices(
    topology_file: str, coordinate_file: str, mask: str
) -> list[int]:
    if pmd is None:
        raise RuntimeError("ParmEd is required for --restrain-mask selection.")
    structure = pmd.load_file(topology_file, coordinate_file)
    selection = structure[mask]
    return [atom.idx for atom in selection.atoms]


def add_positional_restraints(
    system: mm.System,
    indices: list[int],
    reference_positions,
    force_constant_kcal_per_mol_a2: float,
) -> None:
    if not indices:
        raise ValueError("Restraint mask selected zero atoms.")
    if len(reference_positions) < system.getNumParticles():
        raise ValueError("Reference has fewer atoms than the OpenMM system.")

    k = force_constant_kcal_per_mol_a2 * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k * u.kilojoule_per_mole / u.nanometer**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for idx in indices:
        pos = reference_positions[idx].value_in_unit(u.nanometer)
        force.addParticle(idx, [pos.x, pos.y, pos.z])

    system.addForce(force)
    LOGGER.info(
        "Added positional restraints to %d atoms (k=%g kcal/mol/A^2).",
        len(indices),
        force_constant_kcal_per_mol_a2,
    )


def build_system(opt: argparse.Namespace):
    method = {"PME": app.PME, "Ewald": app.Ewald, "NoCutoff": app.NoCutoff}[
        opt.longrange
    ]
    constraints = app.HBonds if opt.shake else None
    cutoff = opt.cutoff * u.angstrom

    if opt.topology and opt.inpcrd:
        LOGGER.info("Loading Amber files: %s, %s", opt.topology, opt.inpcrd)
        prmtop = app.AmberPrmtopFile(opt.topology)
        inpcrd = app.AmberInpcrdFile(opt.inpcrd)
        system = prmtop.createSystem(
            nonbondedMethod=method,
            nonbondedCutoff=cutoff,
            constraints=constraints,
            rigidWater=opt.shake,
            ewaldErrorTolerance=opt.ewald_tolerance,
        )
        if inpcrd.boxVectors is not None:
            system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
        topology = prmtop.topology
        positions = inpcrd.positions
    elif opt.pdb and opt.forcefield:
        LOGGER.info("Loading PDB file: %s", opt.pdb)
        pdb = app.PDBFile(opt.pdb)
        forcefield = app.ForceField(*opt.forcefield)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=method,
            nonbondedCutoff=cutoff,
            constraints=constraints,
            rigidWater=opt.shake,
            ewaldErrorTolerance=opt.ewald_tolerance,
        )
        topology = pdb.topology
        positions = pdb.positions
    else:
        raise ValueError("Use either Amber mode (-t/-i) or PDB mode (-p/-f).")

    if opt.restrain_mask:
        if not (opt.topology and opt.inpcrd):
            raise ValueError("--restrain-mask currently requires Amber mode (-t/-i).")
        if not opt.reference:
            raise ValueError("--reference is required when --restrain-mask is used.")
        indices = amber_mask_indices(opt.topology, opt.inpcrd, opt.restrain_mask)
        reference_positions = load_reference_positions(opt.reference)
        add_positional_restraints(
            system, indices, reference_positions, opt.force_constant
        )

    return system, topology, positions


def minimize(system, topology, positions, opt: argparse.Namespace):
    platform, properties = get_platform(opt.platform, opt.cuda, opt.opencl)
    integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    tolerance = opt.min_tolerance * u.kilojoule_per_mole / u.nanometer
    for cycle in range(1, opt.min_cycles + 1):
        LOGGER.info("Minimization cycle %d/%d", cycle, opt.min_cycles)
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=opt.max_iterations)
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
        LOGGER.info("Potential energy after cycle %d: %.4f kcal/mol", cycle, energy)

    return simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )


def main() -> None:
    opt = parse_args()
    setup_logging(opt.verbose)
    system, topology, positions = build_system(opt)

    LOGGER.info("Writing System XML: %s", opt.system_xml)
    Path(opt.system_xml).write_text(mm.XmlSerializer.serialize(system))

    state = minimize(system, topology, positions, opt)
    LOGGER.info("Writing minimized State XML: %s", opt.min_state)
    Path(opt.min_state).write_text(mm.XmlSerializer.serialize(state))
    LOGGER.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("Failed: %s", exc)
        sys.exit(1)

#!/usr/bin/env python3
"""Run one OpenMM MD stage from a serialized System XML and a State/Amber coordinate source.

Examples:
    python openmm_md.py --xml system.xml -t complex.prmtop -i complex.inpcrd \
        -s sys_min.xml --restart sys_NVT.xml -x sys_NVT.dcd -r sys_NVT.log \
        --temp 298.15 --gamma-ln 1.0 --dt 2.0 -n 500000 --interval 1000

    python openmm_md.py --xml system.xml -t complex.prmtop -s sys_NVT.xml \
        --restart sys_NPT.xml --npt --pressure 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from sys import stdout
from typing import Optional, Tuple

import openmm as mm
import openmm.app as app
from openmm import unit as u

LOGGER = logging.getLogger("openmm_md")


class XMLStateReporter:
    """
    Periodically write a portable OpenMM State XML.

    This is intentionally separate from CheckpointReporter:
    - checkpoint files are platform/precision dependent
    - XML state files are portable and easier to inspect/convert to PDB
    """

    def __init__(
        self, filename: str, report_interval: int, enforce_periodic_box: bool = True
    ):
        self.filename = filename
        self.report_interval = int(report_interval)
        self.enforce_periodic_box = enforce_periodic_box

    def describeNextReport(self, simulation: app.Simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return {
            "steps": steps,
            "periodic": self.enforce_periodic_box,
            "include": ["positions", "velocities", "energy"],
        }

    def report(self, simulation: app.Simulation, state):
        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=self.enforce_periodic_box,
        )
        Path(self.filename).write_text(mm.XmlSerializer.serialize(state))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OpenMM MD stage.")

    # Input/output
    parser.add_argument("--xml", required=True, help="Serialized OpenMM System XML.")
    parser.add_argument(
        "-t",
        "--top",
        required=True,
        help="Amber topology/prmtop file for topology/reporters.",
    )
    parser.add_argument(
        "-i", "--crd", help="Amber coordinates/inpcrd file used if --state is omitted."
    )
    parser.add_argument("-s", "--state", help="Input serialized OpenMM State XML.")
    parser.add_argument(
        "--restart", required=True, help="Output serialized OpenMM State XML."
    )
    parser.add_argument(
        "--write-restart-interval",
        type=int,
        default=None,
        help=(
            "Write --restart periodically every N steps. "
            "Default: same as --interval. Set 0 to disable periodic XML writing."
        ),
    )
    parser.add_argument(
        "--crash-state",
        default="crash_state.xml",
        help="Emergency XML state written if the simulation fails.",
    )

    # MD settings
    parser.add_argument(
        "--dt", "--timestep", type=float, default=2.0, help="Timestep in fs."
    )
    parser.add_argument(
        "--temp", "--temperature", type=float, default=298.15, help="Temperature in K."
    )
    parser.add_argument(
        "--gamma-ln",
        "--gamma_ln",
        dest="gamma_ln",
        type=float,
        default=1.0,
        help="Langevin friction in 1/ps.",
    )
    parser.add_argument(
        "-n", "--steps", type=int, default=500000, help="Number of MD steps."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Reporter/checkpoint interval in steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for integrator/velocities."
    )

    # Ensemble
    parser.add_argument(
        "--npt", action="store_true", help="Add Monte Carlo barostat for NPT."
    )
    parser.add_argument(
        "--pressure", type=float, default=1.0, help="Pressure in atm for NPT."
    )
    # Restraints (currently compatibility/pass-through)
    parser.add_argument(
        "--restrain-mask",
        default=None,
        help="Amber mask for positional restraints.",
    )

    parser.add_argument(
        "-k",
        "--force-constant",
        type=float,
        default=2.5,
        help="Force constant for restraints (kcal/mol/A^2).",
    )

    parser.add_argument(
        "--reference",
        default=None,
        help="Reference structure for restraints.",
    )

    # Velocity handling
    parser.add_argument(
        "--reset-velocities",
        action="store_true",
        help="Ignore velocities from state XML and initialize fresh Maxwell-Boltzmann velocities.",
    )

    # Outputs
    parser.add_argument(
        "-x", "--trajectory", default="trajectory.dcd", help="DCD trajectory output."
    )
    parser.add_argument(
        "-r", "--log", default="state.log", help="StateDataReporter output file."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Legacy option; kept for compatibility, not used.",
    )
    parser.add_argument(
        "--chk", default="simulation.chk", help="Checkpoint output file."
    )

    # Platform
    parser.add_argument(
        "--platform",
        choices=["CUDA", "OpenCL", "CPU"],
        default="CUDA",
        help="Compute platform.",
    )
    parser.add_argument("--cuda", default="0", help="CUDA device index.")
    parser.add_argument("--opencl", default="0", help="OpenCL device index.")
    parser.add_argument(
        "--cuda-precision",
        choices=["single", "mixed", "double"],
        default="mixed",
        help="CUDA precision. Must match when reading binary checkpoints elsewhere.",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def read_xml(path: str):
    with open(path, "r") as handle:
        return mm.XmlSerializer.deserialize(handle.read())


def get_platform(name: str, cuda: str, opencl: str, cuda_precision: str):
    platform = mm.Platform.getPlatformByName(name)
    if name == "CUDA":
        return platform, {"CudaPrecision": cuda_precision, "CudaDeviceIndex": str(cuda)}
    if name == "OpenCL":
        return platform, {"OpenCLDeviceIndex": str(opencl)}
    return platform, {}


def add_barostat_if_needed(
    system: mm.System, temperature: float, pressure: float
) -> None:
    for force in system.getForces():
        if isinstance(force, mm.MonteCarloBarostat):
            LOGGER.warning(
                "System already contains a MonteCarloBarostat; not adding another."
            )
            return

    system.addForce(
        mm.MonteCarloBarostat(pressure * u.atmosphere, temperature * u.kelvin)
    )
    LOGGER.info(
        "Added MonteCarloBarostat at %.3g atm and %.2f K.", pressure, temperature
    )


def load_inputs(opt: argparse.Namespace):
    system = read_xml(opt.xml)
    prmtop = app.AmberPrmtopFile(opt.top)
    topology = prmtop.topology

    positions = None
    velocities = None
    box_vectors = None

    if opt.state:
        LOGGER.info("Loading input State XML: %s", opt.state)
        state = read_xml(opt.state)
        positions = state.getPositions()
        velocities = state.getVelocities()
        box_vectors = state.getPeriodicBoxVectors()

    elif opt.crd:
        LOGGER.info("Loading Amber coordinates: %s", opt.crd)
        inpcrd = app.AmberInpcrdFile(opt.crd)
        positions = inpcrd.positions
        box_vectors = inpcrd.boxVectors

    else:
        raise ValueError("Provide either --state or --crd.")

    if positions is None:
        raise ValueError("No positions were loaded from the input.")

    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    if opt.npt:
        add_barostat_if_needed(system, opt.temp, opt.pressure)

    LOGGER.info("System particles: %d", system.getNumParticles())
    LOGGER.info("Default box vectors: %s", system.getDefaultPeriodicBoxVectors())

    return system, topology, positions, velocities


def has_velocities(velocities) -> bool:
    if velocities is None:
        return False
    try:
        return len(velocities) > 0
    except TypeError:
        return True


def build_simulation(system, topology, positions, velocities, opt: argparse.Namespace):
    integrator = mm.LangevinMiddleIntegrator(
        opt.temp * u.kelvin,
        opt.gamma_ln / u.picosecond,
        opt.dt * u.femtosecond,
    )
    if opt.seed is not None:
        integrator.setRandomNumberSeed(int(opt.seed))

    platform, properties = get_platform(
        opt.platform, opt.cuda, opt.opencl, opt.cuda_precision
    )
    LOGGER.info("Using platform %s with properties %s", opt.platform, properties)

    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    if opt.reset_velocities:
        LOGGER.info("Resetting velocities at %.2f K.", opt.temp)
        simulation.context.setVelocitiesToTemperature(
            opt.temp * u.kelvin, opt.seed or 0
        )

    elif has_velocities(velocities):
        LOGGER.info("Using velocities loaded from input state.")
        simulation.context.setVelocities(velocities)

    else:
        LOGGER.info(
            "No velocities found; assigning Maxwell-Boltzmann velocities at %.2f K.",
            opt.temp,
        )
        simulation.context.setVelocitiesToTemperature(
            opt.temp * u.kelvin, opt.seed or 0
        )

    return simulation


def attach_reporters(simulation: app.Simulation, opt: argparse.Namespace) -> None:
    interval = int(opt.interval)

    if opt.trajectory:
        LOGGER.info("Writing trajectory: %s", opt.trajectory)
        simulation.reporters.append(app.DCDReporter(opt.trajectory, interval))

    if opt.chk:
        LOGGER.info("Writing checkpoint: %s", opt.chk)
        simulation.reporters.append(app.CheckpointReporter(opt.chk, interval))

    restart_interval = opt.write_restart_interval
    if restart_interval is None:
        restart_interval = interval

    if restart_interval and restart_interval > 0:
        LOGGER.info(
            "Writing portable XML restart every %d steps: %s",
            restart_interval,
            opt.restart,
        )
        simulation.reporters.append(XMLStateReporter(opt.restart, restart_interval))

    if opt.log:
        LOGGER.info("Writing state log: %s", opt.log)
        simulation.reporters.append(
            app.StateDataReporter(
                opt.log,
                interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                density=True,
                speed=True,
                progress=True,
                remainingTime=True,
                totalSteps=opt.steps,
                separator="\t",
            )
        )

    simulation.reporters.append(
        app.StateDataReporter(
            stdout,
            interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=opt.steps,
            separator="\t",
        )
    )


def write_state(simulation: app.Simulation, path: str, label: str) -> None:
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    Path(path).write_text(mm.XmlSerializer.serialize(state))
    LOGGER.info("Saved %s State XML: %s", label, path)


def run(opt: argparse.Namespace) -> None:
    system, topology, positions, velocities = load_inputs(opt)
    simulation = build_simulation(system, topology, positions, velocities, opt)
    attach_reporters(simulation, opt)

    LOGGER.info(
        "Running %d steps at %.2f K, dt=%.3g fs, gamma_ln=%.3g 1/ps.",
        opt.steps,
        opt.temp,
        opt.dt,
        opt.gamma_ln,
    )

    try:
        simulation.step(opt.steps)
    except Exception:
        LOGGER.exception("Simulation failed during dynamics.")
        try:
            write_state(simulation, opt.crash_state, "emergency crash")
        except Exception:
            LOGGER.exception("Could not save emergency crash state.")
        raise

    write_state(simulation, opt.restart, "final restart")


def main() -> None:
    opt = parse_args()
    setup_logging(opt.verbose)
    run(opt)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("Failed: %s", exc)
        sys.exit(1)

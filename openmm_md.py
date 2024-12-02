#!/usr/bin/env python
# title           :openmm_md.py
# description     :This will perform MD simulation with OpenMM package
# date            :11-20-2024
# usage           :python openmm_md.py -h
# python_version  :3.x
# ==============================================================================

import argparse
import logging
import sys
from sys import stdout
from openmm import unit as u
from openmm import app as app
from openmm import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *
from mdtraj.reporters import DCDReporter

# Setup logging
logger = logging.getLogger("openmm_md")
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def calculate_temperature(state, system):
    """Calculate temperature from the kinetic energy of the system."""
    kinetic_energy = (
        state.getKineticEnergy()
    )  # Kinetic energy with units of energy (e.g., kilojoules/mole)
    num_particles = system.getNumParticles()
    degrees_of_freedom = (
        3 * num_particles - system.getNumConstraints()
    )  # Degrees of freedom

    # Ensure unit compatibility by converting energy to joules
    kinetic_energy_joules = kinetic_energy.value_in_unit(u.joule)  # Convert to joules
    boltzmann_constant = u.BOLTZMANN_CONSTANT_kB.value_in_unit(
        u.joule / u.kelvin
    )  # Ensure consistent units

    # Calculate temperature
    temperature = (2 * kinetic_energy_joules) / (
        degrees_of_freedom * boltzmann_constant
    )
    return temperature


# Function to log detailed simulation state
def log_simulation_state(simulation, step, log_file, system):
    """Log detailed simulation state."""
    state = simulation.context.getState(
        getEnergy=True, getPositions=False, getVelocities=True
    )

    kinetic_energy = state.getKineticEnergy().value_in_unit(u.kilocalories_per_mole)
    potential_energy = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    total_energy = kinetic_energy + potential_energy

    # Log formatted output
    with open(log_file, "a") as log:
        log.write(f"NSTEP = {step:10d}\n")
        # log.write(f"TEMP(K) = {temperature}\n")
        log.write(
            f"Etot   = {total_energy:.4f}  EKtot   = {kinetic_energy:.4f}  EPtot      = {potential_energy:.4f}\n"
        )
        log.write("-" * 80 + "\n")


def log_stage(stage, status="START"):
    """Log the start or end of a simulation stage."""
    logger.info(f"{'='*40}")
    logger.info(f"{status} STAGE: {stage}")
    logger.info(f"{'='*40}")


def load_system(opt):
    """Load the system from XML, Amber topology, or minimized state."""
    log_stage("SYSTEM SETUP", "START")
    system, topology, positions, velocities = None, None, None, None

    # Parse system XML file
    if opt.xml:
        logger.info(f"Parsing system XML file: {opt.xml}")
        try:
            with open(opt.xml, "r") as f:
                system = mm.XmlSerializer.deserialize(f.read())
            logger.info("System XML successfully parsed.")
        except Exception as e:
            logger.error(f"Failed to parse system XML: {e}")
            sys.exit(1)
    else:
        logger.error("System XML file (--xml) is required.")
        sys.exit(1)

    # Load minimized state if provided
    if opt.state:
        try:
            logger.info(f"Loading minimized state from: {opt.state}")
            with open(opt.state, "r") as f:
                state = mm.XmlSerializer.deserialize(f.read())
            logger.info("Minimized state loaded successfully.")

            # Set positions and box vectors from the minimized state
            positions = state.getPositions()
            velocities = state.getVelocities()
            if velocities is None:
                logger.warning("Velocities not found in state. Initializing to zero.")
                velocities = [mm.Vec3(0, 0, 0) for _ in range(len(positions))]
            system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        except Exception as e:
            logger.error(f"Failed to load minimized state file: {e}")
            sys.exit(1)

    # Load Amber topology and coordinates if provided
    elif opt.crd and opt.top:
        try:
            logger.info(f"Loading Amber topology: {opt.top} and coordinates: {opt.crd}")
            prmtop = AmberPrmtopFile(opt.top)
            inpcrd = AmberInpcrdFile(opt.crd)
            positions = inpcrd.positions
            topology = prmtop.topology
            if inpcrd.boxVectors:
                logger.info("Setting periodic box vectors from Amber inpcrd file.")
                system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
            else:
                logger.warning("No periodic box vectors found in Amber inpcrd file.")
            logger.info("Amber topology and coordinates loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Amber files: {e}")
            sys.exit(1)

    else:
        logger.error("No valid state or Amber inputs provided. Exiting.")
        sys.exit(1)

    # Apply NPT condition and Monte Carlo Barostat if requested
    if opt.npt:
        logger.info("Adding Monte Carlo Barostat for NPT ensemble...")
        try:
            system.addForce(
                mm.MonteCarloBarostat(1.0 * u.atmosphere, opt.temperature * u.kelvin)
            )
            logger.info("Monte Carlo Barostat added successfully.")
        except Exception as e:
            logger.error(f"Failed to add Monte Carlo Barostat: {e}")
            sys.exit(1)

    # Final validation of system setup
    logger.info("Validating system setup...")
    if system is not None:
        logger.info(f"Number of particles: {system.getNumParticles()}")
        logger.info(f"Box vectors: {system.getDefaultPeriodicBoxVectors()}")

    log_stage("SYSTEM SETUP", "END")
    return system, topology, positions, velocities


def run_simulation(simulation, system, opt):
    """Run the simulation with detailed logging."""
    log_stage("SIMULATION", "START")
    logger.info(f"Running simulation for {opt.steps} steps at {opt.temperature} K")

    # Initialize detailed log file
    log_file = opt.log if opt.log else "simulation.log"
    with open(log_file, "w") as log:
        log.write("Detailed Simulation Log\n")
        log.write("=" * 80 + "\n")

    try:
        # Main simulation loop with progress and detailed state logging
        for step in range(0, opt.steps, opt.interval):
            simulation.step(opt.interval)

            # Log simulation progress
            logger.info(f"Progress: {step + opt.interval}/{opt.steps} steps completed.")

            # Log detailed state to file
            log_simulation_state(simulation, step + opt.interval, log_file, system)

        logger.info("Simulation completed successfully!")

        # Save the final state if restart file is specified
        if opt.restart:
            logger.info(f"Saving final state to {opt.restart}...")
            with open(opt.restart, "w") as f:
                f.write(
                    mm.XmlSerializer.serialize(
                        simulation.context.getState(
                            getPositions=True, getVelocities=True
                        )
                    )
                )
            logger.info("Final state saved successfully.")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)

    log_stage("SIMULATION", "END")


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(
        description="Run OpenMM molecular dynamics simulation."
    )
    parser.add_argument("--xml", required=True, help="System XML file.")
    parser.add_argument("-i", "--crd", help="Amber coordinates file.")
    parser.add_argument("-t", "--top", help="Amber topology file.")
    parser.add_argument("-s", "--state", help="Minimized state file (sys_min.xml).")
    parser.add_argument("--restart", help="Restart state file to save (sys_final.xml).")

    # Simulation parameters
    parser.add_argument(
        "--timestep",
        "--dt",
        type=float,
        default=2.0,
        help="Simulation timestep in femtoseconds.",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=298.15,
        help="Simulation temperature in Kelvin.",
    )
    parser.add_argument(
        "--gamma_ln", type=float, default=1.0, help="Langevin collision frequency."
    )
    parser.add_argument(
        "--steps", "-n", type=int, default=500000, help="Number of simulation steps."
    )
    parser.add_argument(
        "--interval", type=int, default=1000, help="Interval for reporting data."
    )

    # Restraints
    parser.add_argument(
        "--restrain-mask", help="Mask for positional restraints (e.g., !:WAT&!@H=)."
    )
    parser.add_argument(
        "-k",
        "--force-constant",
        type=float,
        default=2.5,
        help="Force constant for restraints (kcal/mol/A^2).",
    )
    parser.add_argument("--reference", help="Reference PDB file for restraints.")
    parser.add_argument(
        "--npt",
        action="store_true",
        help="Enable NPT ensemble with a Monte Carlo barostat.",
    )

    # Output files
    parser.add_argument(
        "-x", "--trajectory", default="trajectory.dcd", help="Trajectory output file."
    )
    parser.add_argument(
        "-r", "--log", default="state.log", help="Log file for simulation statistics."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="simulation.out",
        help="Output file for simulation progress.",
    )
    parser.add_argument(
        "--chk",
        default="simulation.chk",
        help="Checkpoint file for the simulation.",
    )

    # Platform options
    parser.add_argument(
        "--platform",
        choices=["CUDA", "OpenCL", "CPU"],
        default="CUDA",
        help="Compute platform.",
    )
    parser.add_argument("--cuda", default="0", help="CUDA device index.")
    parser.add_argument("--opencl", default="0", help="OpenCL device index.")

    opt = parser.parse_args()

    # Load the system
    system, topology, positions, velocities = load_system(opt)
    # system, topology, positions, _ = load_system(opt)

    # Create integrator
    integrator = mm.LangevinIntegrator(
        opt.temperature * u.kelvin,
        1.0 / u.picoseconds,
        opt.timestep * u.femtoseconds,
    )

    # Setup platform
    logger.info(f"Setting platform to {opt.platform}...")
    if opt.platform == "CUDA":
        platform = mm.Platform.getPlatformByName("CUDA")
        platform_properties = {"CudaPrecision": "mixed", "CudaDeviceIndex": opt.cuda}
    elif opt.platform == "OpenCL":
        platform = mm.Platform.getPlatformByName("OpenCL")
        platform_properties = {"OpenCLDeviceIndex": opt.opencl}
    else:
        platform = mm.Platform.getPlatformByName("CPU")
        platform_properties = {}

    # Add NPT barostat if requested
    if opt.npt:
        logger.info("Adding Monte Carlo Barostat for NPT ensemble...")
        try:
            system.addForce(
                mm.MonteCarloBarostat(1.0 * u.atmosphere, opt.temperature * u.kelvin)
            )
            logger.info("Monte Carlo Barostat added successfully.")
        except Exception as e:
            logger.error(f"Failed to add Monte Carlo Barostat: {e}")
            sys.exit(1)

    # Create simulation
    logger.info("Creating simulation object...")

    # Set up the simulation
    simulation = app.Simulation(
        topology, system, integrator, platform, platform_properties
    )
    simulation.context.setPositions(positions)
    if velocities:
        simulation.context.setVelocities(velocities)

    # Attach MDTraj's DCDReporter to the simulation
    if opt.trajectory:
        try:
            logger.info(f"Saving trajectory to: {opt.trajectory}")
            simulation.reporters.append(DCDReporter(opt.trajectory, int(opt.interval)))
        except Exception as e:
            logger.error(f"Failed to set DCDReporter: {e}")
            sys.exit(1)
    else:
        logger.warning("No trajectory file specified. Skipping DCD reporting.")

    simulation.reporters.append(CheckpointReporter(opt.chk, int(opt.interval)))

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            int(opt.interval),
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=opt.steps,
            separator="      ",
        )
    )

    # Optional: Log initial temperature
    state = simulation.context.getState(getEnergy=True, getVelocities=True)
    try:
        temperature = calculate_temperature(state, system)
        logger.info(f"Initial Temperature: {temperature:.2f} K")
    except Exception as e:
        logger.warning(f"Failed to calculate initial temperature: {e}")

    # Run the simulation
    run_simulation(simulation, system, opt)


if __name__ == "__main__":
    main()

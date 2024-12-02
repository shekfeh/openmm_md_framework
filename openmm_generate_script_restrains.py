#!/usr/bin/env python3
# Example: python generate_script.py -i my_system.inpcrd -t my_system.prmtop --prep-script openmm_prep.py \
# --md-script openmm_md.py --restrain-mask "!:WAT&!@H=" --reference my_reference.pdb \
# minim nvt npt md --min-time 0.05 --nvt-time 2 --npt-time 5 --md-time 20 --prod-segments 10 --dt 2.0

# or:  python openmm_generate_script.py -i my_system.inpcrd -t my_system.prmtop  minim nvt npt md

#!/usr/bin/env python3

import argparse


def generate_script(
    prep_script,
    md_script,
    restrain_mask,
    reference,
    inpcrd,
    prmtop,
    run_minim,
    run_nvt,
    run_npt,
    run_md,
    minim_steps,
    nvt_steps,
    npt_steps,
    prod_steps,
    prod_segments,
    dt,
    temp,
    gamma_ln,
):
    """
    Generates a bash script to run OpenMM simulations with specified arguments.
    """
    with open("run_simulation.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Redirect both stdout and stderr to a log file\n")
        f.write('exec > >(tee -a "openmm.log") 2>&1\n')

        # Minimization phase
        if run_minim:
            f.write("# Minimization phase\n")
            f.write("if [ ! -f sys_min.xml ]; then\n")
            f.write(f"python {prep_script} -i {inpcrd} -t {prmtop} \\\n")
            f.write("-l PME -c 10.00 --shake \\\n")
            f.write(
                f"--restrain-mask '{restrain_mask}' -k 10.0 --reference {reference} \\\n"
            )
            f.write("fi\n\n")

        # NVT phase
        if run_nvt:
            f.write("# NVT simulation\n")
            f.write("if [ ! -f sys_NVT.xml ]; then\n")
            f.write(f"python {md_script} -i {inpcrd} -t {prmtop} \\\n")
            f.write("--xml system.xml -s sys_min.xml --restart sys_NVT.xml \\\n")
            f.write(
                "-x sys_NVT.dcd -r sys_NVT.info -o sys_NVT.out --chk sys_NVT.chk\\\n"
            )
            f.write(
                f"--temp {temp} --gamma_ln {gamma_ln} -n {nvt_steps} --interval 1000 --dt {dt} \\\n"
            )
            f.write(
                f"--restrain-mask '{restrain_mask}' -k 2.5 --reference {reference} \\\n"
            )
            f.write("fi\n\n")

        # NPT phase
        if run_npt:
            f.write("# NPT simulation\n")
            f.write("if [ ! -f sys_NPT.xml ]; then\n")
            f.write(f"python {md_script} -i {inpcrd} -t {prmtop} \\\n")
            f.write("--xml system.xml -s sys_NVT.xml --restart sys_NPT.xml \\\n")
            f.write(
                "-x sys_NPT.dcd -r sys_NPT.info -o sys_NPT.out --chk sys_NPT.chk\\\n"
            )
            f.write(
                f"--temp {temp} --gamma_ln {gamma_ln} -n {npt_steps} --interval 10000 --dt {dt} \\\n"
            )
            f.write(
                f"--restrain-mask '{restrain_mask}' -k 2.5 --reference {reference} --npt \\\n"
            )
            f.write("fi\n\n")

        # Production MD phase
        if run_md:
            f.write("# Production MD phase\n")
            for i in range(1, prod_segments + 1):
                f.write(f"if [ ! -f sys_md_{i}.xml ]; then\n")
                f.write(f"python {md_script} -i {inpcrd} -t {prmtop} \\\n")
                f.write(
                    f"--xml system.xml -s sys_md_{i-1 if i > 1 else 'NPT'}.xml --restart sys_md_{i}.xml \\\n"
                )
                f.write(
                    f"-x sys_md_{i}.dcd -r sys_md_{i}.info -o sys_md_{i}.out --chk sys_md_{i}.chk \\\n"
                )
                f.write(
                    f"--temp {temp} --gamma_ln {gamma_ln} -n {prod_steps} --interval 1000 --dt {dt} \\\n"
                )
                f.write(
                    f"--restrain-mask '{restrain_mask}' -k 2.5 --reference {reference} --npt \\\n"
                )
                f.write("fi\n\n")


def calculate_steps(total_time_ns, dt_fs):
    return int((total_time_ns * 1e6) / dt_fs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenMM run script for molecular dynamics simulations."
    )

    # Paths to scripts
    parser.add_argument(
        "--prep-script",
        default="openmm_prep.py",
        help="Path to the preparation script.",
    )
    parser.add_argument(
        "--md-script", default="openmm_md.py", help="Path to the MD simulation script."
    )

    # Amber input files
    parser.add_argument(
        "-i", "--inpcrd", required=True, help="Amber coordinate file (inpcrd)."
    )
    parser.add_argument(
        "-t", "--prmtop", required=True, help="Amber topology file (prmtop)."
    )

    # Restraints
    parser.add_argument(
        "--restrain-mask", default="!:WAT&!@H=", help="Mask for positional restraints."
    )
    parser.add_argument(
        "--reference",
        default="prot_amber.pdb",
        help="Reference PDB file for restraints.",
    )

    # Phases
    parser.add_argument(
        "phases",
        choices=["minim", "nvt", "npt", "md"],
        nargs="+",
        help="Phases to run (choose from: minim, nvt, npt, md).",
    )

    # Time settings
    parser.add_argument(
        "--min-time", type=float, default=0.025, help="Minimization time (ns)."
    )
    parser.add_argument(
        "--nvt-time", type=float, default=1, help="NVT simulation time (ns)."
    )
    parser.add_argument(
        "--npt-time", type=float, default=2, help="NPT simulation time (ns)."
    )
    parser.add_argument(
        "--md-time", type=float, default=10, help="Total production MD time (ns)."
    )
    parser.add_argument(
        "--prod-segments", type=int, default=5, help="Number of production MD segments."
    )

    # MD Parameters
    parser.add_argument("--dt", type=float, default=2.0, help="Timestep (fs).")
    parser.add_argument(
        "--temp", type=float, default=298.15, help="Simulation temperature (K)."
    )
    parser.add_argument(
        "--gamma-ln",
        type=float,
        default=1.0,
        help="Langevin thermostat friction coefficient.",
    )

    args = parser.parse_args()

    run_minim = "minim" in args.phases
    run_nvt = "nvt" in args.phases
    run_npt = "npt" in args.phases
    run_md = "md" in args.phases

    minim_steps = calculate_steps(args.min_time, args.dt)
    nvt_steps = calculate_steps(args.nvt_time, args.dt)
    npt_steps = calculate_steps(args.npt_time, args.dt)
    prod_steps = calculate_steps(args.md_time / args.prod_segments, args.dt)

    generate_script(
        args.prep_script,
        args.md_script,
        args.restrain_mask,
        args.reference,
        args.inpcrd,
        args.prmtop,
        run_minim,
        run_nvt,
        run_npt,
        run_md,
        minim_steps,
        nvt_steps,
        npt_steps,
        prod_steps,
        args.prod_segments,
        args.dt,
        args.temp,
        args.gamma_ln,
    )


if __name__ == "__main__":
    main()

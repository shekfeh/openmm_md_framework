#!/usr/bin/env python3
"""
Generate a resumable bash workflow for OpenMM minimization/equilibration/production.
a safer OpenMM bash workflow with optional gradual heating
Example:
    python openmm_generate_script.py \
      -i complex_solvated.inpcrd -t complex_solvated.prmtop \
      minim nvt npt md \
      --reference prot_amber.pdb --restrain-mask '!:WAT&!@H='
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def q(x) -> str:
    return shlex.quote(str(x))


def ns_to_steps(ns: float, dt_fs: float) -> int:
    if ns < 0:
        raise ValueError("time in ns cannot be negative")
    if dt_fs <= 0:
        raise ValueError("dt must be positive")
    return max(1, int(round(ns * 1_000_000.0 / dt_fs)))


def cmd_lines(parts: list[str], indent: str = "  ") -> str:
    """
    Format shell command nicely:
      --flag value pairs stay together
      standalone flags stay standalone
    """

    if not parts:
        raise ValueError("empty command")

    grouped = []
    i = 0

    while i < len(parts):
        token = parts[i]

        # standalone boolean flag
        if token.startswith("-") and (
            i + 1 >= len(parts) or parts[i + 1].startswith("-")
        ):
            grouped.append(token)
            i += 1

        # option + value pair
        elif token.startswith("-") and i + 1 < len(parts):
            grouped.append(f"{token} {parts[i + 1]}")
            i += 2

        # command/program
        else:
            grouped.append(token)
            i += 1

    if len(grouped) == 1:
        return indent + grouped[0] + "\n"

    lines = [indent + grouped[0] + " \\"]

    for item in grouped[1:-1]:
        lines.append(f"{indent}{item} \\")

    lines.append(f"{indent}{grouped[-1]}")

    return "\n".join(lines) + "\n"


def write_stage(f, title: str, target: str, parts: list[str]) -> None:
    f.write(f"\n# {title}\n")
    f.write(f"if [ ! -f {q(target)} ]; then\n")
    f.write(f'  echo "Running: {title}"\n')
    f.write(cmd_lines(parts))
    f.write("fi\n")


def md_parts(
    args,
    *,
    state_in: str,
    restart: str,
    traj: str,
    log: str,
    chk: str,
    temp: float,
    gamma_ln: float,
    dt: float,
    steps: int,
    interval: int,
    npt: bool = False,
    k: float | None = None,
    reset_velocities: bool = False,
) -> list[str]:
    parts = [
        "python",
        q(args.md_script),
        "--xml",
        "system.xml",
        "-t",
        q(args.prmtop),
        "-s",
        q(state_in),
        "--restart",
        q(restart),
        "-x",
        q(traj),
        "-r",
        q(log),
        "-o",
        q(log.replace(".log", ".out")),
        "--chk",
        q(chk),
        "--temp",
        q(temp),
        "--gamma_ln",
        q(gamma_ln),
        "-n",
        q(steps),
        "--interval",
        q(interval),
        "--dt",
        q(dt),
        "--platform",
        q(args.platform),
    ]

    if args.platform == "CUDA":
        parts += ["--cuda", q(args.cuda), "--cuda-precision", q(args.cuda_precision)]
    elif args.platform == "OpenCL":
        parts += ["--opencl", q(args.opencl)]

    if args.write_restart_interval is not None:
        parts += ["--write-restart-interval", q(args.write_restart_interval)]

    if reset_velocities:
        parts.append("--reset-velocities")

    # Compatibility with openmm_md.py versions that accept restraint args.
    # If your openmm_md.py only accepts but does not apply restraints, these are harmless.
    if args.restrain_mask and args.reference and k is not None and k > 0:
        parts += [
            "--restrain-mask",
            q(args.restrain_mask),
            "-k",
            q(k),
            "--reference",
            q(args.reference),
        ]

    if npt:
        parts.append("--npt")

    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a safer run_simulation.sh for OpenMM."
    )

    p.add_argument("-i", "--inpcrd", required=True, help="Amber inpcrd/rst7 file")
    p.add_argument("-t", "--prmtop", required=True, help="Amber prmtop file")

    p.add_argument(
        "phases",
        choices=["minim", "nvt", "npt", "md"],
        nargs="+",
        help="Stages to include",
    )

    p.add_argument("--prep-script", default="openmm_prep.py")
    p.add_argument("--md-script", default="openmm_md.py")
    p.add_argument("--output", default="run_simulation.sh")

    # Restraints
    p.add_argument("--reference", default="prot_amber.pdb")
    p.add_argument("--restrain-mask", default="!:WAT&!@H=")
    p.add_argument("--min-k", type=float, default=10.0)
    p.add_argument("--heat-k", type=float, default=10.0)
    p.add_argument("--nvt-k", type=float, default=5.0)
    p.add_argument("--npt-k", type=float, default=2.5)
    p.add_argument("--relax1-k", type=float, default=2.5)
    p.add_argument("--relax2-k", type=float, default=1.0)
    p.add_argument("--relax3-k", type=float, default=0.25)
    p.add_argument("--restrain-production", action="store_true")
    p.add_argument("--production-k", type=float, default=0.25)

    # Safer defaults: production default is 1 fs.
    p.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Production timestep in fs. Default: 1.0. Use 2.0 only after stability is verified.",
    )
    p.add_argument(
        "--temp", type=float, default=298.15, help="Final target temperature in K"
    )
    p.add_argument("--gamma-ln", dest="gamma_ln", type=float, default=1.0)
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--npt-interval", type=int, default=500)
    p.add_argument(
        "--write-restart-interval",
        type=int,
        default=None,
        help="Forwarded to openmm_md.py. Default: not explicitly passed.",
    )

    # Platform
    p.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    p.add_argument("--cuda", default="0")
    p.add_argument("--opencl", default="0")
    p.add_argument(
        "--cuda-precision",
        choices=["single", "mixed", "double"],
        default="mixed",
    )

    # Heating
    p.add_argument("--heating", action="store_true", default=True)
    p.add_argument("--no-heating", action="store_false", dest="heating")
    p.add_argument("--heat-temps", default="50,100,150,200,250")
    p.add_argument("--heat-steps", type=int, default=50000)
    p.add_argument("--heat-dt", type=float, default=0.5)
    p.add_argument("--heat-gamma-ln", type=float, default=5.0)

    # Main equilibration
    p.add_argument("--min-time", type=float, default=0.025)
    p.add_argument("--nvt-time", type=float, default=1.0)
    p.add_argument("--nvt-dt", type=float, default=0.5)
    p.add_argument("--nvt-gamma-ln", type=float, default=5.0)

    p.add_argument("--npt-time", type=float, default=2.0)
    p.add_argument("--npt-dt", type=float, default=0.5)
    p.add_argument("--npt-gamma-ln", type=float, default=5.0)

    # Extra relaxation after NPT, before production
    p.add_argument(
        "--md-relax",
        action="store_true",
        default=True,
        help="Add production relaxation stages before production. Default: on.",
    )
    p.add_argument("--no-md-relax", action="store_false", dest="md_relax")
    p.add_argument("--relax1-steps", type=int, default=50000)
    p.add_argument("--relax1-dt", type=float, default=0.5)
    p.add_argument("--relax1-gamma-ln", type=float, default=5.0)

    p.add_argument("--relax2-steps", type=int, default=100000)
    p.add_argument("--relax2-dt", type=float, default=1.0)
    p.add_argument("--relax2-gamma-ln", type=float, default=2.0)

    p.add_argument("--relax3-steps", type=int, default=100000)
    p.add_argument("--relax3-dt", type=float, default=1.0)
    p.add_argument("--relax3-gamma-ln", type=float, default=1.0)

    # Production
    p.add_argument("--md-time", type=float, default=10.0)
    p.add_argument("--prod-segments", type=int, default=5)
    p.add_argument("--skip-bash-check", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    phases = set(args.phases)

    if args.prod_segments < 1:
        raise ValueError("--prod-segments must be >= 1")

    heating_temps: list[float] = []
    if args.heating:
        heating_temps = [
            float(x.strip()) for x in args.heat_temps.split(",") if x.strip()
        ]
        heating_temps = [t for t in heating_temps if t < args.temp]

    nvt_steps = ns_to_steps(args.nvt_time, args.nvt_dt)
    npt_steps = ns_to_steps(args.npt_time, args.npt_dt)
    prod_steps = ns_to_steps(args.md_time / args.prod_segments, args.dt)

    out = Path(args.output)

    with out.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write("exec > >(tee -a openmm.log) 2>&1\n")
        f.write('echo "Started at $(date)"\n')

        if "minim" in phases:
            parts = [
                "python",
                q(args.prep_script),
                "-i",
                q(args.inpcrd),
                "-t",
                q(args.prmtop),
                "-l",
                "PME",
                "-c",
                "10.0",
                "--shake",
                "--restrain-mask",
                q(args.restrain_mask),
                "-k",
                q(args.min_k),
                "--reference",
                q(args.reference),
            ]
            write_stage(f, "Minimization / system preparation", "sys_min.xml", parts)

        current_state = "sys_min.xml"

        if "nvt" in phases:
            if heating_temps:
                for idx, temp in enumerate(heating_temps, start=1):
                    out_state = f"sys_heat_{idx:02d}_{int(round(temp))}K.xml"
                    parts = md_parts(
                        args,
                        state_in=current_state,
                        restart=out_state,
                        traj=f"sys_heat_{idx:02d}_{int(round(temp))}K.dcd",
                        log=f"sys_heat_{idx:02d}_{int(round(temp))}K.log",
                        chk=f"sys_heat_{idx:02d}_{int(round(temp))}K.chk",
                        temp=temp,
                        gamma_ln=args.heat_gamma_ln,
                        dt=args.heat_dt,
                        steps=args.heat_steps,
                        interval=min(args.interval, args.heat_steps),
                        npt=False,
                        k=args.heat_k,
                        reset_velocities=(idx == 1),
                    )
                    write_stage(
                        f, f"NVT heating rung {idx}: {temp:g} K", out_state, parts
                    )
                    current_state = out_state

            parts = md_parts(
                args,
                state_in=current_state,
                restart="sys_NVT.xml",
                traj="sys_NVT.dcd",
                log="sys_NVT.log",
                chk="sys_NVT.chk",
                temp=args.temp,
                gamma_ln=args.nvt_gamma_ln,
                dt=args.nvt_dt,
                steps=nvt_steps,
                interval=args.interval,
                npt=False,
                k=args.nvt_k,
                reset_velocities=False,
            )
            write_stage(
                f, f"Final restrained NVT at {args.temp:g} K", "sys_NVT.xml", parts
            )
            current_state = "sys_NVT.xml"

        if "npt" in phases:
            parts = md_parts(
                args,
                state_in=current_state,
                restart="sys_NPT.xml",
                traj="sys_NPT.dcd",
                log="sys_NPT.log",
                chk="sys_NPT.chk",
                temp=args.temp,
                gamma_ln=args.npt_gamma_ln,
                dt=args.npt_dt,
                steps=npt_steps,
                interval=args.npt_interval,
                npt=True,
                k=args.npt_k,
                reset_velocities=False,
            )
            write_stage(f, "Restrained NPT equilibration", "sys_NPT.xml", parts)
            current_state = "sys_NPT.xml"

        if "md" in phases:
            if args.md_relax:
                relax_specs = [
                    (
                        "sys_md_relax_1.xml",
                        "Production relaxation 1: 0.5 fs, high friction, restrained",
                        args.relax1_steps,
                        args.relax1_dt,
                        args.relax1_gamma_ln,
                        args.relax1_k,
                    ),
                    (
                        "sys_md_relax_2.xml",
                        "Production relaxation 2: 1.0 fs, medium friction, weak restraint",
                        args.relax2_steps,
                        args.relax2_dt,
                        args.relax2_gamma_ln,
                        args.relax2_k,
                    ),
                    (
                        "sys_md_relax_3.xml",
                        "Production relaxation 3: 1.0 fs, production friction, very weak restraint",
                        args.relax3_steps,
                        args.relax3_dt,
                        args.relax3_gamma_ln,
                        args.relax3_k,
                    ),
                ]

                for idx, (out_state, title, steps, dt, gamma_ln, k) in enumerate(
                    relax_specs, start=1
                ):
                    parts = md_parts(
                        args,
                        state_in=current_state,
                        restart=out_state,
                        traj=out_state.replace(".xml", ".dcd"),
                        log=out_state.replace(".xml", ".log"),
                        chk=out_state.replace(".xml", ".chk"),
                        temp=args.temp,
                        gamma_ln=gamma_ln,
                        dt=dt,
                        steps=steps,
                        interval=min(args.interval, steps),
                        npt=True,
                        k=k,
                        reset_velocities=False,
                    )
                    write_stage(f, title, out_state, parts)
                    current_state = out_state

            for i in range(1, args.prod_segments + 1):
                out_state = f"sys_md_{i}.xml"
                k = args.production_k if args.restrain_production else None
                parts = md_parts(
                    args,
                    state_in=current_state,
                    restart=out_state,
                    traj=f"sys_md_{i}.dcd",
                    log=f"sys_md_{i}.log",
                    chk=f"sys_md_{i}.chk",
                    temp=args.temp,
                    gamma_ln=args.gamma_ln,
                    dt=args.dt,
                    steps=prod_steps,
                    interval=args.interval,
                    npt=True,
                    k=k,
                    reset_velocities=False,
                )
                write_stage(
                    f,
                    f"Production MD segment {i}: dt={args.dt:g} fs",
                    out_state,
                    parts,
                )
                current_state = out_state

        f.write('\necho "Finished at $(date)"\n')

    out.chmod(0o755)

    if not args.skip_bash_check:
        result = subprocess.run(
            ["bash", "-n", str(out)], text=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Generated script failed bash -n:\n{result.stderr}")

    print(f"Wrote {out}")
    print("Syntax check: OK")
    print(f"Production dt: {args.dt:g} fs")
    if args.dt > 1.0:
        print(
            "WARNING: production dt > 1 fs. Use only if your system is stable with constraints/HMR."
        )


if __name__ == "__main__":
    main()

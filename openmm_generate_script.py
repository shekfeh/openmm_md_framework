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
    """Return a multiline shell command with safe continuations.

    Each continued line ends with exactly ' \\', i.e. a space before the
    backslash. The final line has no backslash. Therefore a following 'fi'
    can never be escaped into the command.
    """
    if not parts:
        raise ValueError("empty command")
    lines: list[str] = []
    for i, part in enumerate(parts):
        suffix = " \\" if i < len(parts) - 1 else ""
        lines.append(f"{indent}{part}{suffix}")
    return "\n".join(lines) + "\n"


def write_stage(f, title: str, target: str, parts: list[str]) -> None:
    f.write(f"\n# {title}\n")
    f.write(f"if [ ! -f {q(target)} ]; then\n")
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
) -> list[str]:
    parts = [
        "python",
        q(args.md_script),
        "--xml",
        "system.xml",
        "-i",
        q(args.inpcrd),
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
        "--cuda",
        q(args.cuda),
    ]
    if args.restrain_mask and args.reference and k is not None:
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

    p.add_argument("--reference", default="prot_amber.pdb")
    p.add_argument("--restrain-mask", default="!:WAT&!@H=")
    p.add_argument("--min-k", type=float, default=10.0)
    p.add_argument("--heat-k", type=float, default=10.0)
    p.add_argument("--eq-k", type=float, default=2.5)
    p.add_argument("--restrain-production", action="store_true")

    p.add_argument(
        "--dt",
        type=float,
        default=2.0,
        help="Default production/equilibration timestep in fs",
    )
    p.add_argument(
        "--temp", type=float, default=298.15, help="Final target temperature in K"
    )
    p.add_argument(
        "--gamma-ln",
        dest="gamma_ln",
        type=float,
        default=1.0,
        help="Default Langevin friction 1/ps",
    )
    p.add_argument("--interval", type=int, default=1000)
    p.add_argument("--npt-interval", type=int, default=10000)
    p.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    p.add_argument("--cuda", default="0")

    p.add_argument(
        "--heating",
        action="store_true",
        default=True,
        help="Use gradual NVT heating before full NVT. Default: on",
    )
    p.add_argument(
        "--no-heating",
        action="store_false",
        dest="heating",
        help="Disable gradual heating",
    )
    p.add_argument(
        "--heat-temps",
        default="50,100,150,200,250",
        help="Comma-separated heating temperatures before final temp",
    )
    p.add_argument(
        "--heat-steps", type=int, default=50000, help="Steps per heating rung"
    )
    p.add_argument("--heat-dt", type=float, default=0.5, help="Heating timestep in fs")
    p.add_argument(
        "--heat-gamma-ln", type=float, default=5.0, help="Heating friction in 1/ps"
    )
    p.add_argument(
        "--min-time",
        type=float,
        default=0.025,
        help="Approximate minimization length equivalent in ns; kept for compatibility.",
    )
    p.add_argument(
        "--nvt-time", type=float, default=1.0, help="Final NVT time after heating, ns"
    )
    p.add_argument(
        "--npt-time", type=float, default=2.0, help="NPT equilibration time, ns"
    )
    p.add_argument(
        "--md-time", type=float, default=10.0, help="Total production time, ns"
    )
    p.add_argument("--prod-segments", type=int, default=5)
    p.add_argument("--skip-bash-check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phases = set(args.phases)
    if args.prod_segments < 1:
        raise ValueError("--prod-segments must be >= 1")

    heating_temps = []
    if args.heating:
        heating_temps = [
            float(x.strip()) for x in args.heat_temps.split(",") if x.strip()
        ]
        heating_temps = [t for t in heating_temps if t < args.temp]

    nvt_steps = ns_to_steps(args.nvt_time, args.dt)
    npt_steps = ns_to_steps(args.npt_time, args.dt)
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
                gamma_ln=args.gamma_ln,
                dt=args.dt,
                steps=nvt_steps,
                interval=args.interval,
                npt=False,
                k=args.eq_k,
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
                gamma_ln=args.gamma_ln,
                dt=args.dt,
                steps=npt_steps,
                interval=args.npt_interval,
                npt=True,
                k=args.eq_k,
            )
            write_stage(f, "Restrained NPT equilibration", "sys_NPT.xml", parts)
            current_state = "sys_NPT.xml"

        if "md" in phases:
            for i in range(1, args.prod_segments + 1):
                out_state = f"sys_md_{i}.xml"
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
                    k=args.eq_k if args.restrain_production else None,
                )
                write_stage(f, f"Production MD segment {i}", out_state, parts)
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


if __name__ == "__main__":
    main()

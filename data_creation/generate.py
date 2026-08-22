import argparse
import importlib
import os
import shutil
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.integrate import ode

warnings.filterwarnings("error")


def generate(n, args):
    """Solve model with ODE solver and save trajectory, parameters, and initial state."""

    seed = n
    TRIES = 10

    domain = args["domain"]
    model = args["model"]
    total_samples = args["total_samples"]
    T = args["T"]
    N_t = args["N_t"]
    d_constants = args["d_constants"]
    d_states = args["d_states"]
    outpath = args["out_path"]
    rand_start = args["rand_start"]

    for tryyy in range(TRIES):

        np.random.seed(seed)

        # ------------------------------------------------------------
        # Load ODE model
        # ------------------------------------------------------------

        model_path = (
            "models/" + domain + "/" + model + "/model.py"
        )

        spec = importlib.util.spec_from_file_location(
            "model_module",
            model_path
        )

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        # ------------------------------------------------------------
        # Initialize constants and state variables
        # ------------------------------------------------------------

        LEGEND_STATES, legend_algebraic, legend_voi, legend_constants = (
            module.createLegends()
        )

        init_states, constants = module.initConsts()

        # Randomize initial states
        init_states = [
            np.random.uniform(
                x * (1 - d_states),
                x * (1 + d_states)
            )
            for x in init_states
        ]

        # Randomize ODE parameters
        constants = [
            np.random.uniform(
                x * (1 - d_constants),
                x * (1 + d_constants)
            )
            for x in constants
        ]

        # ------------------------------------------------------------
        # Keep a copy of the ground-truth values
        # ------------------------------------------------------------

        # These are the actual parameters used to generate
        # this trajectory.
        theta = np.asarray(
            constants,
            dtype=np.float32
        )

        # These are the actual initial conditions used to
        # generate this trajectory.
        y0 = np.asarray(
            init_states,
            dtype=np.float32
        )

        # ------------------------------------------------------------
        # Set timespan
        # ------------------------------------------------------------

        voi = np.linspace(
            0,
            T * 2,
            N_t * 2
        )

        # ------------------------------------------------------------
        # Solve ODE
        # ------------------------------------------------------------

        try:

            r = ode(module.computeRates)

            r.set_integrator(
                "lsoda",
                method="bdf",
                atol=1e-6,
                rtol=1e-6,
                max_step=1
            )

            r.set_initial_value(
                init_states,
                voi[0]
            )

            r.set_f_params(
                constants
            )

            states = np.array(
                [
                    [0.0] * len(voi)
                ] * module.sizeStates
            )

            states[:, 0] = init_states

            for i, t in enumerate(voi[1:]):

                if r.successful():

                    r.integrate(t)

                    states[:, i + 1] = r.y

                else:

                    break

        except (
            Exception,
            RuntimeWarning,
            UserWarning
        ):

            seed += total_samples

            continue

        # ------------------------------------------------------------
        # Random crop
        # ------------------------------------------------------------

        if rand_start:

            start = np.random.randint(
                0,
                N_t
            )

        else:

            start = 0

        # ------------------------------------------------------------
        # Construct trajectory
        # ------------------------------------------------------------

        times = pd.DataFrame(
            {
                "t": voi[:N_t]
            }
        )

        states = pd.DataFrame(
            states[
                :,
                start:start + N_t
            ].T,
            columns=LEGEND_STATES
        )

        states = pd.concat(
            [
                times,
                states
            ],
            axis=1
        )

        # ------------------------------------------------------------
        # Validate trajectory
        # ------------------------------------------------------------

        if (
            np.max(
                np.absolute(
                    states.values
                )
            ) > 1e15
        ):

            seed += total_samples

            continue

        if states.isna().any().any():

            seed += total_samples

            continue

        # ------------------------------------------------------------
        # Save trajectory
        # ------------------------------------------------------------

        trajectory_path = (
            f"{outpath}/ts-data-{str(n)}.parquet"
        )

        states.to_parquet(
            trajectory_path,
            index=None
        )

        # ------------------------------------------------------------
        # Save ground-truth parameters
        # and initial state
        # ------------------------------------------------------------

        metadata = {
            "theta": theta,
            "y0": y0
        }

        metadata_path = (
            f"{outpath}/ts-data-{str(n)}.npz"
        )

        np.savez(
            metadata_path,
            theta=theta,
            y0=y0
        )

        return

    raise Exception("NO more TRIES")


def create(args):

    total_processes = cpu_count()

    domain = args["domain"]
    model = args["model"]

    if "final" in args and args["final"]:

        out_path = (
            f"data/benchmark_datasets/{model}/"
        )

    else:

        out_path = (
            f"data/raw_data/{domain}/{model}/"
            f"T={args['T']}_"
            f"ds={args['d_states']}_"
            f"dc={args['d_constants']}/"
        )

    args["out_path"] = out_path

    if not os.path.isdir(
        f"data/raw_data/{domain}/"
    ):

        os.mkdir(
            f"data/raw_data/{domain}/"
        )

    if not os.path.isdir(
        f"data/raw_data/{domain}/{model}"
    ):

        os.mkdir(
            f"data/raw_data/{domain}/{model}"
        )

    if os.path.isdir(out_path):

        shutil.rmtree(out_path)

    os.mkdir(out_path)

    iters = range(
        args["total_samples"]
    )

    with Pool(
        processes=total_processes
    ) as pool:

        pool.starmap(
            generate,
            [
                (i, args)
                for i in iters
            ]
        )

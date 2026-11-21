import argparse
import importlib.util
import os
import shutil
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.integrate import ode

warnings.filterwarnings("error")


def generate(n, args):
    """
    Solve model with ODE solver and save:
        - trajectory
        - ground-truth parameters
        - ground-truth initial state

    A sample is saved ONLY if the ODE solver successfully
    reaches the complete requested time range and the resulting
    trajectory is finite.
    """

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

        # ============================================================
        # Load ODE model
        # ============================================================

        model_path = f"models/{domain}/{model}/model.py"

        spec = importlib.util.spec_from_file_location(
            "model_module",
            model_path
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # ============================================================
        # Initialize constants and state variables
        # ============================================================

        (
            LEGEND_STATES,
            legend_algebraic,
            legend_voi,
            legend_constants
        ) = module.createLegends()

        init_states, constants = module.initConsts()

        # ============================================================
        # Randomize initial states
        # ============================================================

        init_states = [
            np.random.uniform(
                x * (1.0 - d_states),
                x * (1.0 + d_states)
            )
            for x in init_states
        ]

        # ============================================================
        # Randomize ODE parameters
        # ============================================================

        constants = [
            np.random.uniform(
                x * (1.0 - d_constants),
                x * (1.0 + d_constants)
            )
            for x in constants
        ]

        # ============================================================
        # Ground truth
        # ============================================================

        theta = np.asarray(
            constants,
            dtype=np.float32
        )

        y0 = np.asarray(
            init_states,
            dtype=np.float32
        )

        # ============================================================
        # Set timespan
        # ============================================================

        voi = np.linspace(
            0.0,
            T * 2.0,
            N_t * 2
        )

        # ============================================================
        # Solve ODE
        # ============================================================

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

            # Store trajectory.
            # We intentionally initialize with NaN rather than zero.
            #
            # This is important:
            # if integration stops early, we can detect the missing
            # part of the trajectory instead of accidentally saving zeros.
            states = np.full(
                (module.sizeStates, len(voi)),
                np.nan,
                dtype=np.float64
            )

            # Initial condition
            states[:, 0] = init_states

            integration_success = True

            # --------------------------------------------------------
            # Integrate every requested time point
            # --------------------------------------------------------

            for i, t in enumerate(voi[1:], start=1):

                if not r.successful():
                    integration_success = False
                    break

                r.integrate(t)

                # Check solver status immediately
                if not r.successful():
                    integration_success = False
                    break

                # Check returned state
                if not np.all(np.isfinite(r.y)):
                    integration_success = False
                    break

                states[:, i] = r.y

            # --------------------------------------------------------
            # IMPORTANT:
            # Require the COMPLETE trajectory.
            # --------------------------------------------------------

            if not integration_success:
                seed += total_samples
                continue

            if not np.all(np.isfinite(states)):
                seed += total_samples
                continue

        except (
            Exception,
            RuntimeWarning,
            UserWarning
        ):

            seed += total_samples
            continue

        # ============================================================
        # Random crop
        # ============================================================

        if rand_start:

            start = np.random.randint(
                0,
                N_t
            )

        else:

            start = 0

        end = start + N_t

        # Safety check
        if end > len(voi):
            seed += total_samples
            continue

        # ============================================================
        # Construct trajectory
        # ============================================================

        times = pd.DataFrame(
            {
                "t": voi[:N_t]
            }
        )

        trajectory = pd.DataFrame(
            states[:, start:end].T,
            columns=LEGEND_STATES
        )

        trajectory = pd.concat(
            [
                times,
                trajectory
            ],
            axis=1
        )

        # ============================================================
        # Validate final trajectory
        # ============================================================

        values = trajectory.drop(columns=["t"]).to_numpy(
            dtype=np.float64
        )

        # Reject NaN / Inf
        if not np.all(np.isfinite(values)):
            seed += total_samples
            continue

        # Reject exploding trajectories
        if np.max(np.abs(values)) > 1e15:
            seed += total_samples
            continue

        # ------------------------------------------------------------
        # DO NOT reject zeros in general.
        #
        # A zero can theoretically be a legitimate model value.
        # What matters here is that the trajectory is complete.
        # ------------------------------------------------------------

        # ============================================================
        # Save trajectory
        # ============================================================

        trajectory_path = (
            f"{outpath}/ts-data-{str(n)}.parquet"
        )

        trajectory.to_parquet(
            trajectory_path,
            index=None
        )

        # ============================================================
        # Save ground-truth parameters and initial state
        # ============================================================

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

    raise Exception(
        f"NO more TRIES for sample {n}"
    )


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

    # ================================================================
    # Create directories
    # ================================================================

    if not os.path.isdir(
        f"data/raw_data/{domain}/"
    ):
        os.makedirs(
            f"data/raw_data/{domain}/"
        )

    if not os.path.isdir(
        f"data/raw_data/{domain}/{model}"
    ):
        os.makedirs(
            f"data/raw_data/{domain}/{model}"
        )

    # Remove old dataset
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)

    # ================================================================
    # Generate samples
    # ================================================================

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

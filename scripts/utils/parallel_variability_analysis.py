import numpy as np
import pandas as pd
from optlang.symbolics import Zero
from cobra.core import Configuration
from cobra.util import solver as sutil
from tqdm.auto import tqdm
import multiprocessing
import os
import pickle
from pathlib import Path
from platform import system
from tempfile import mkstemp
from types import TracebackType
from typing import Any, Callable, Optional, Tuple, Type
from cobra import Model
from optlang.exceptions import SolverError

""""
This python script contains functions that are used to conduct variability analysis on given variables.
It is based on the code from the cobra package, but has been modified to work for all variable types
I.T. 2025
"""

__all__ = ("ProcessPool",)

def _init_win_worker(filename: str) -> None:
    """Retrieve worker initialization code from a pickle file and execute it."""
    with open(filename, mode="rb") as handle:
        func, *args = pickle.load(handle)
    func(*args)

class ProcessPool:
    """A process pool that handles Windows-specific performance issues."""

    def __init__(
        self,
        processes: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        maxtasksperchild: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a process pool with improved performance on Windows.
        
        On Windows, the initializer function is passed via a pickle file to avoid
        slow process spawning issues.
        """
        super().__init__(**kwargs)
        self._filename = None

        if initializer is not None and system() == "Windows":
            descriptor, self._filename = mkstemp(suffix=".pkl")
            with os.fdopen(descriptor, mode="wb") as handle:
                pickle.dump((initializer,) + initargs, handle)
            initializer = _init_win_worker
            initargs = (self._filename,)
        
        self._pool = multiprocessing.Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
        )

    def __getattr__(self, name: str, **kwargs) -> Any:
        """Delegate attribute access to the underlying multiprocessing pool."""
        return getattr(self._pool, name, **kwargs)

    def __enter__(self) -> "ProcessPool":
        """Enable usage as a context manager."""
        self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Ensure proper cleanup when exiting the context manager."""
        try:
            self._pool.close()
            self._pool.join()
        finally:
            self._clean_up()
        return self._pool.__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Close the pool and clean up resources."""
        try:
            self._pool.close()
        finally:
            self._clean_up()

    def _clean_up(self) -> None:
        """Remove temporary files if they exist."""
        if self._filename is not None and Path(self._filename).exists():
            Path(self._filename).unlink()

configuration = Configuration()

def _init_worker(model: "Model", sense: str) -> None:
    """Initialize a global model object for multiprocessing workers."""
    global _model
    _model = model
    _model.solver.objective.direction = sense

def variability_analysis(
    model: "Model",
    var_list,
    processes: int = 1
) -> pd.DataFrame:
    """
    Perform flux variability analysis (FVA) on a given model.
    
    Parameters:
    - model (Model): The metabolic model.
    - var_list (list): List of variable (reaction) IDs to analyze.
    - processes (int): Number of parallel processes to use.
    
    Returns:
    - pd.DataFrame: A DataFrame with the minimum and maximum flux values.
    """
    num_vars = len(var_list)
    processes = min(processes, num_vars)

    fva_result = pd.DataFrame(
        {
            "minimum": np.zeros(num_vars, dtype=float),
            "maximum": np.zeros(num_vars, dtype=float),
        },
        index=var_list,
    )
    
    with model:
        model.objective = Zero  # Reset objective
        for bound in ("minimum", "maximum"):
            if processes > 1:
                chunk_size = 5
                with ProcessPool(
                    processes,
                    initializer=_init_worker,
                    initargs=(model, bound[:3]),
                ) as pool:
                    for rxn_id, value in tqdm(
                        pool.imap_unordered(_variability_step, var_list, chunksize=chunk_size),
                        total=len(var_list),
                        desc=f"Calculating {bound}"
                    ):
                        fva_result.at[rxn_id, bound] = value
            else:
                _init_worker(model, bound[:3])
                for rxn_id, value in map(_variability_step, var_list):
                    fva_result.at[rxn_id, bound] = value
    
    return fva_result[["minimum", "maximum"]]

def _variability_step(var_id: str) -> Tuple[str, float]:
    """
    Compute the flux variability for a single reaction.
    
    Parameters:
    - var_id (str): The reaction ID.
    
    Returns:
    - Tuple[str, float]: The reaction ID and its computed flux bound.
    """
    global _model
    var = _model.variables[var_id]

    try:
        _model.solver.objective.set_linear_coefficients({var: 1})
        _model.slim_optimize()

        # Check if a valid solution exists
        if _model.solver.status != "optimal":
            return var_id, float("nan")

        value = _model.solver.objective.value or float("nan")
    except SolverError:
        print(f"Warning: Solver failed for {var_id}, returning NaN.")
        value = float("nan")

    # Reset objective coefficient
    _model.solver.objective.set_linear_coefficients({var: 0})
    return var_id, value
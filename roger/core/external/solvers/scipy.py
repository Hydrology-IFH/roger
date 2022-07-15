import numpy as onp
import scipy.sparse
import scipy.sparse.linalg as spalg

from roger import logger, roger_kernel, roger_routine, distributed, runtime_state as rst
from roger.variables import allocate
from roger.core.operators import update, at, numpy as npx
from roger.core.external.solvers.base import LinearSolver
from roger.core.external.poisson_matrix import assemble_poisson_matrix


class SciPySolver(LinearSolver):
    @roger_routine(
        local_variables=(
            "z0",
            "maskCatch",
        ),
        dist_safe=False,
    )
    def __init__(self, state):
        self._matrix, self._boundary_mask = self._assemble_poisson_matrix(state)

        jacobi_precon = self._jacobi_preconditioner(state, self._matrix)
        self._matrix = jacobi_precon * self._matrix
        self._rhs_scale = jacobi_precon.diagonal()
        self._extra_args = {}

        logger.info("Computing ILU preconditioner...")
        ilu_preconditioner = spalg.spilu(self._matrix.tocsc(), drop_tol=1e-6, fill_factor=100)
        self._extra_args["M"] = spalg.LinearOperator(self._matrix.shape, ilu_preconditioner.solve)

    def _scipy_solver(self, state, rhs, x0, boundary_val):
        orig_shape = x0.shape
        orig_dtype = x0.dtype

        rhs = npx.where(self._boundary_mask, rhs, boundary_val)  # set right hand side on boundaries

        rhs = onp.asarray(rhs.reshape(-1) * self._rhs_scale, dtype="float64")
        x0 = onp.asarray(x0.reshape(-1), dtype="float64")

        linear_solution, info = spalg.bicgstab(
            self._matrix,
            rhs,
            x0=x0,
            atol=1e-8,
            tol=0,
            maxiter=1000,
            **self._extra_args,
        )

        if info > 0:
            logger.warning("Solver did not converge after {} iterations", info)

        return npx.asarray(linear_solution, dtype=orig_dtype).reshape(orig_shape)

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Main solver. Solves a 2D Poisson equation. Uses scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            x0: Initial guess
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.

        """
        rhs_global, x0_global, boundary_val = gather_variables(state, rhs, x0, boundary_val)

        if rst.proc_rank == 0:
            linear_solution = self._scipy_solver(state, rhs_global, x0_global, boundary_val=boundary_val)
        else:
            linear_solution = npx.empty_like(rhs)

        return scatter_variables(state, linear_solution)

    @staticmethod
    def _jacobi_preconditioner(state, matrix):
        """
        Construct a simple Jacobi preconditioner
        """
        settings = state.settings

        eps = 1e-20
        precon = allocate(state.dimensions, ("x", "y"), fill=1, local=False)
        diag = npx.reshape(matrix.diagonal().copy(), (settings.nx + 4, settings.ny + 4))[2:-2, 2:-2]
        precon = update(precon, at[2:-2, 2:-2], npx.where(npx.abs(diag) > eps, 1.0 / (diag + eps), 1.0))
        precon = onp.asarray(precon)
        return scipy.sparse.dia_matrix((precon.reshape(-1), 0), shape=(precon.size, precon.size)).tocsr()

    @staticmethod
    def _assemble_poisson_matrix(state):
        diags, offsets, boundary_mask = assemble_poisson_matrix(state)

        # flatten offsets (as expected by scipy.sparse)
        offsets = tuple(-dx * diags[0].shape[1] - dy for dx, dy in offsets)

        diags = tuple(onp.asarray(diag.reshape(-1)) for diag in (diags))

        matrix = scipy.sparse.dia_matrix(
            (diags, offsets),
            shape=(diags[0].size, diags[0].size),
            dtype="float64",
        ).T.tocsr()

        return matrix, boundary_mask


@roger_kernel
def gather_variables(state, rhs, x0, boundary_val):
    rhs_global = distributed.gather(rhs, state.dimensions, ("x", "y"))
    x0_global = distributed.gather(x0, state.dimensions, ("x", "y"))

    if boundary_val is None:
        boundary_val = x0_global
    else:
        boundary_val = distributed.gather(boundary_val, state.dimensions, ("x", "y"))

    return rhs_global, x0_global, boundary_val


@roger_kernel
def scatter_variables(state, linear_solution):
    return distributed.scatter(linear_solution, state.dimensions, ("x", "y"))

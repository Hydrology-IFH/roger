import sys

import numpy as np
from mpi4py import MPI

from roger import runtime_settings as rs, runtime_state as rst
from roger.distributed import gather

rs.diskless_mode = True

if rst.proc_num > 1:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4


from roger.setups.svat import SVATSetup  # noqa: E402
from roger.tools.make_toy_setup import make_setup  # noqa: E402

sim = SVATSetup(
    override=dict(
        nx=8,
        ny=8
    )
)

#TODO: make setup safe for distributed

if rst.proc_num == 1:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    try:
        sim.setup()
        sim.run()
    except Exception as exc:
        print(str(exc))
        comm.Abort(1)
        raise

    other_theta = np.empty_like(sim.state.variables.theta)
    comm.Recv(other_theta, 0)

    np.testing.assert_allclose(sim.state.variables.theta, other_theta)
else:
    sim.setup()
    sim.run()

    theta_global = gather(sim.state.variables.theta, sim.state.dimensions, ("x", "y"))

    # if rst.proc_rank == 0:
    #     rs.mpi_comm.Get_parent().Send(np.array(theta_global), 0)

import sys

import numpy as np
from mpi4py import MPI

from roger import runtime_settings as rs, runtime_state as rst
from roger.distributed import gather

rs.linear_solver = "scipy"
rs.diskless_mode = True

if rst.proc_num > 1:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4


from roger.setups.dummy import DUMMYSetup  # noqa: E402

sim = DUMMYSetup(
    override=dict(
        runlen=86400 * 10,
    )
)

if rst.proc_num == 1:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    try:
        sim.setup()
        sim.run()
    except Exception as exc:
        print(str(exc))
        comm.Abort(1)
        raise

    other_S = np.empty_like(sim.state.variables.S)
    comm.Recv(other_S, 0)

    np.testing.assert_allclose(sim.state.variables.S, other_S)
else:
    sim.setup()
    sim.run()

    S_global = gather(sim.state.variables.S, sim.state.dimensions, ("x", "y"))

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(np.array(S_global), 0)

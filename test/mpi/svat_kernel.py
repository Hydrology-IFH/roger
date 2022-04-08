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


from roger.setups.svat import SVATSetup  # noqa: E402
from roger.tools.make_toy_setup import make_setup  # noqa: E402

sim = SVATSetup(
    override=dict(
        nx=42,
        ny=30
    )
)
make_setup(sim._base_path, event_type='rain', ndays=10, nrows=42, ncols=30)

if rst.proc_num == 1:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    try:
        sim.setup()
        sim.run()
    except Exception as exc:
        print(str(exc))
        comm.Abort(1)
        raise

    other_swe = np.empty_like(sim.state.variables.swe)
    comm.Recv(other_swe, 0)

    np.testing.assert_allclose(sim.state.variables.swe, other_swe)
else:
    sim.setup()
    sim.run()

    swe_global = gather(sim.state.variables.swe, sim.state.dimensions, ("x", "y"))

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(np.array(swe_global), 0)

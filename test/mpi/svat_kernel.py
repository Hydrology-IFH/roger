import sys
import os
import shutil

import numpy as np
from mpi4py import MPI
from pathlib import Path

from roger import runtime_settings as rs, runtime_state as rst
from roger.distributed import gather

rs.diskless_mode = True

if rst.proc_num > 1:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4


from roger.setups.svat import SVATSetup  # noqa: E402
from roger.tools.make_toy_data import make_toy_forcing  # noqa: E402

dict_settings = dict(
    nx=8,
    ny=8
)

sim = SVATSetup(
    override=dict_settings
)
# create temporary directory
sim._base_path = Path(__file__).parent / "tmp"
if not os.path.exists(sim._base_path):
    os.mkdir(sim._base_path)

# run toy setup on single process and let the other processes wait
make_toy_forcing(sim._base_path, ndays=10, nrows=dict_settings["nx"], ncols=dict_settings["ny"])

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

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(np.array(theta_global), 0)

# delete temporary directory
shutil.rmtree(sim._base_path, ignore_errors=True)

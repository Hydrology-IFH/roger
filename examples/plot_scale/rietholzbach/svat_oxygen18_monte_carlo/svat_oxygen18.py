from pathlib import Path
import h5netcdf
import numpy as onp
import yaml
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option(
    "-tms",
    "--transport-model-structure",
    type=click.Choice(
        [
            "complete-mixing",
            "piston",
            "advection-dispersion-power",
            "time-variant_advection-dispersion-power",
            "preferential-power",
            "older-preference-power",
            "advection-dispersion-kumaraswamy",
            "time-variant_advection-dispersion-kumaraswamy",
        ]
    ),
    default="advection-dispersion-power",
)
@click.option("-ss", "--sas-solver", type=click.Choice(["RK4", "Euler", "deterministic"]), default="deterministic")
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(nsamples, transport_model_structure, sas_solver, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, random_uniform
    from roger.tools.setup import write_forcing_tracer
    from roger.core.transport import delta_to_conc, conc_to_delta

    class SVATOXYGEN18Setup(RogerSetup):
        """A SVAT transport model for oxygen-18."""

        _base_path = Path(__file__).parent
        _tm_structure = transport_model_structure.replace("_", " ")
        _input_dir = _base_path / "input"
        _tmp_dir = Path(tmp_dir)
        if transport_model_structure in ["complete-mixing", "piston"]:
            _sim_file = "SVAT_best100.nc"
        else:
            _sim_file = f"SVAT_best100_bootstrap.nc"
        # load parameter boundaries
        _file_params = _base_path / "param_bounds.yml"
        if transport_model_structure in ["complete-mixing", "piston"]:
            _bounds = None
        else:
            with open(_file_params, "r") as file:
                _bounds = yaml.safe_load(file)[_tm_structure]

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj)) + 1

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["Time"]
                return len(onp.array(var_obj)) * 60 * 60 * 24

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            identifier = f"SVATOXYGEN18_{transport_model_structure}_{sas_solver}"
            settings.identifier = identifier
            settings.sas_solver = sas_solver
            settings.sas_solver_substeps = 6
            if settings.sas_solver in ["RK4", "Euler"]:
                settings.h = 1 / settings.sas_solver_substeps

            settings.nx, settings.ny = nsamples, 1
            settings.nitt = self._get_nitt(self._input_dir, "forcing_tracer.nc")
            settings.ages = 1500
            settings.nages = settings.ages + 1
            settings.runlen_warmup = 2 * 365 * 24 * 60 * 60
            settings.runlen = self._get_runlen(self._input_dir, "forcing_tracer.nc")

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1996-12-31 00:00:00"

            settings.enable_offline_transport = True
            settings.enable_oxygen18 = True
            settings.tm_structure = self._tm_structure

            settings.d18O_min = -19.3
            settings.d18O_max = -0.9

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "dt_secs",
                "dt",
                "ages",
                "nages",
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables
            settings = state.settings

            # temporal grid
            vs.dt_secs = 60 * 60 * 24
            vs.dt = 60 * 60 * 24 / (60 * 60)
            vs.ages = update(vs.ages, at[:], npx.arange(1, settings.nages))
            vs.nages = update(vs.nages, at[:], npx.arange(settings.nages))
            # grid of model runs
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

        @roger_routine
        def set_look_up_tables(self, state):
            pass

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_sat_rz",
                "S_sat_ss",
                "sas_params_evap_soil",
                "sas_params_cpr_rz",
                "sas_params_transp",
                "sas_params_q_rz",
                "sas_params_q_ss",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_pwp_rz = update(
                vs.S_pwp_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc("S_pwp_rz", self._base_path / "input", self._sim_file)[:, :, 0],
            )
            vs.S_pwp_ss = update(
                vs.S_pwp_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc("S_pwp_ss", self._base_path / "input", self._sim_file)[:, :, 0],
            )
            vs.S_sat_rz = update(
                vs.S_sat_rz,
                at[2:-2, 2:-2],
                self._read_var_from_nc("S_sat_rz", self._base_path / "input", self._sim_file)[:, :, 0],
            )
            vs.S_sat_ss = update(
                vs.S_sat_ss,
                at[2:-2, 2:-2],
                self._read_var_from_nc("S_sat_ss", self._base_path / "input", self._sim_file)[:, :, 0],
            )

            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "piston":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 50)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 50)
            elif settings.tm_structure == "preferential-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_transp"][0],
                        self._bounds["k_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_rz"][0],
                        self._bounds["k_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_ss"][0],
                        self._bounds["k_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
            elif settings.tm_structure == "advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_transp"][0],
                        self._bounds["k_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_rz"][0],
                        self._bounds["k_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_ss"][0],
                        self._bounds["k_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
            elif settings.tm_structure == "time-variant advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 3],
                    random_uniform(
                        self._bounds["c1_transp"][0],
                        self._bounds["c1_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 4],
                    random_uniform(
                        self._bounds["c2_transp"][0],
                        self._bounds["c2_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 3],
                    random_uniform(
                        self._bounds["c1_q_rz"][0],
                        self._bounds["c1_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 4],
                    random_uniform(
                        self._bounds["c2_q_rz"][0],
                        self._bounds["c2_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 3],
                    random_uniform(
                        self._bounds["c1_q_ss"][0],
                        self._bounds["c1_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 4],
                    random_uniform(
                        self._bounds["c2_q_ss"][0],
                        self._bounds["c2_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])
            elif settings.tm_structure == "older-preference-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_transp"][0],
                        self._bounds["k_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_rz"][0],
                        self._bounds["k_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["k_q_ss"][0],
                        self._bounds["k_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
            elif settings.tm_structure == "advection-dispersion-kumaraswamy":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["a_transp"][0],
                        self._bounds["a_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 2],
                    random_uniform(
                        self._bounds["b_transp"][0],
                        self._bounds["b_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["a_q_rz"][0],
                        self._bounds["a_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 2],
                    random_uniform(
                        self._bounds["b_q_rz"][0],
                        self._bounds["b_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["a_q_ss"][0],
                        self._bounds["a_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 2],
                    random_uniform(
                        self._bounds["b_q_ss"][0],
                        self._bounds["b_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
            elif settings.tm_structure == "time-variant advection-dispersion-kumaraswamy":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 37)
                vs.sas_params_transp = update(
                    vs.sas_params_transp,
                    at[2:-2, 2:-2, 1],
                    random_uniform(
                        self._bounds["a_transp"][0],
                        self._bounds["a_transp"][1],
                        tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                    )[2:-2, 2:-2],
                )
                c1 = random_uniform(
                    self._bounds["c1_transp"][0],
                    self._bounds["c1_transp"][1],
                    tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                )[2:-2, 2:-2]
                c2 = random_uniform(
                    self._bounds["c2_transp"][0],
                    self._bounds["c2_transp"][1],
                    tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])),
                )[2:-2, 2:-2]
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], c1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], c2)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 36)
                vs.sas_params_q_rz = update(
                    vs.sas_params_q_rz,
                    at[2:-2, 2:-2, 2],
                    random_uniform(
                        self._bounds["b_q_rz"][0],
                        self._bounds["b_q_rz"][1],
                        tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                    )[2:-2, 2:-2],
                )
                c1 = random_uniform(
                    self._bounds["c1_q_rz"][0],
                    self._bounds["c1_q_rz"][1],
                    tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                )[2:-2, 2:-2]
                c2 = random_uniform(
                    self._bounds["c2_q_rz"][0],
                    self._bounds["c2_q_rz"][1],
                    tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])),
                )[2:-2, 2:-2]
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], c1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], c2)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 36)
                vs.sas_params_q_ss = update(
                    vs.sas_params_q_ss,
                    at[2:-2, 2:-2, 2],
                    random_uniform(
                        self._bounds["b_q_ss"][0],
                        self._bounds["b_q_ss"][1],
                        tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                    )[2:-2, 2:-2],
                )
                c1 = random_uniform(
                    self._bounds["c1_q_ss"][0],
                    self._bounds["c1_q_ss"][1],
                    tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                )[2:-2, 2:-2]
                c2 = random_uniform(
                    self._bounds["c2_q_ss"][0],
                    self._bounds["c2_q_ss"][1],
                    tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])),
                )[2:-2, 2:-2]
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], c1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], c2)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=["S_snow", "S_rz", "S_rz_init", "S_ss", "S_ss_init", "S_s", "itt", "taup1"],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_snow = update(
                vs.S_snow,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc("S_snow", self._base_path / "input", self._sim_file)[
                    :, :, vs.itt, npx.newaxis
                ],
            )
            vs.S_rz = update(
                vs.S_rz,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc("S_rz", self._base_path / "input", self._sim_file)[
                    :, :, vs.itt, npx.newaxis
                ],
            )
            vs.S_ss = update(
                vs.S_ss,
                at[2:-2, 2:-2, :vs.taup1],
                self._read_var_from_nc("S_ss", self._base_path / "input", self._sim_file)[
                    :, :, vs.itt, npx.newaxis
                ],
            )
            vs.S_s = update(
                vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_rz[2:-2, 2:-2, :vs.taup1] + vs.S_ss[2:-2, 2:-2, :vs.taup1]
            )
            vs.S_rz_init = update(vs.S_rz_init, at[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, 0])
            vs.S_ss_init = update(vs.S_ss_init, at[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, 0])

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # uniform StorAge
            arr0 = allocate(state.dimensions, ("x", "y"))
            vs.sa_rz = update(
                vs.sa_rz,
                at[2:-2, 2:-2, :vs.taup1, 1:],
                npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[
                    :, :, npx.newaxis, :
                ],
            )
            vs.sa_ss = update(
                vs.sa_ss,
                at[2:-2, 2:-2, :vs.taup1, 1:],
                npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[
                    :, :, npx.newaxis, :
                ],
            )

            vs.SA_rz = update(
                vs.SA_rz,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.SA_ss = update(
                vs.SA_ss,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
            )

            vs.sa_s = update(
                vs.sa_s,
                at[2:-2, 2:-2, :, :],
                vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
            )
            vs.SA_s = update(
                vs.SA_s,
                at[2:-2, 2:-2, :, 1:],
                npx.cumsum(vs.sa_s[2:-2, 2:-2, :, :], axis=-1),
            )

            if settings.enable_oxygen18:
                vs.C_snow = update(vs.C_snow, at[2:-2, 2:-2, :vs.taup1], npx.nan)
                vs.C_iso_snow = update(vs.C_iso_snow, at[2:-2, 2:-2, :vs.taup1], npx.nan)
                vs.C_iso_rz = update(vs.C_iso_rz, at[2:-2, 2:-2, :vs.taup1], -10.5)
                vs.C_iso_ss = update(vs.C_iso_ss, at[2:-2, 2:-2, :vs.taup1], -10.5)
                vs.C_rz = update(
                    vs.C_rz,
                    at[2:-2, 2:-2, :vs.taup1],
                    delta_to_conc(state, vs.C_iso_rz[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, :],
                    vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, 0],
                    0,
                )
                vs.C_ss = update(
                    vs.C_ss,
                    at[2:-2, 2:-2, :vs.taup1],
                    delta_to_conc(state, vs.C_iso_ss[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, :],
                    vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, 0],
                    0,
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :, :],
                    npx.where(
                        vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0,
                        vs.msa_rz[2:-2, 2:-2, :, :]
                        * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :]))
                        + vs.msa_ss[2:-2, 2:-2, :, :]
                        * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])),
                        0,
                    ),
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :vs.taup1, 0],
                    0,
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.tau],
                    npx.sum(
                        npx.where(
                            vs.sa_s[2:-2, 2:-2, vs.tau, :] > 0,
                            vs.msa_s[2:-2, 2:-2, vs.tau, :]
                            * (
                                vs.sa_s[2:-2, 2:-2, vs.tau, :]
                                / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]
                            ),
                            0,
                        ),
                        axis=-1,
                    ),
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.taum1],
                    vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.taum1],
                    conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.tau],
                    conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.csa_rz = update(
                    vs.csa_rz,
                    at[2:-2, 2:-2, vs.tau, :],
                    conc_to_delta(state, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_ss = update(
                    vs.csa_ss,
                    at[2:-2, 2:-2, vs.tau, :],
                    conc_to_delta(state, vs.msa_ss[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_s = update(
                    vs.csa_s,
                    at[2:-2, 2:-2, vs.tau, :],
                    conc_to_delta(state, vs.msa_s[2:-2, 2:-2, vs.tau, :]),
                )

        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "PREC_DIST_DAILY",
                "INF_MAT_RZ",
                "INF_PF_RZ",
                "INF_PF_SS",
                "TRANSP",
                "EVAP_SOIL",
                "CPR_RZ",
                "Q_RZ",
                "Q_SS",
                "S_RZ",
                "S_SS",
                "S_S",
                "S_SNOW",
                "C_ISO_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.PREC_DIST_DAILY = update(
                vs.PREC_DIST_DAILY,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("prec", self._base_path / "input", self._sim_file),
            )
            vs.INF_MAT_RZ = update(
                vs.INF_MAT_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("inf_mat_rz", self._base_path / "input", self._sim_file),
            )
            vs.INF_PF_RZ = update(
                vs.INF_PF_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("inf_mp_rz", self._base_path / "input", self._sim_file)
                + self._read_var_from_nc("inf_sc_rz", self._base_path / "input", self._sim_file),
            )
            vs.INF_PF_SS = update(
                vs.INF_PF_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("inf_ss", self._base_path / "input", self._sim_file),
            )
            vs.TRANSP = update(
                vs.TRANSP,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("transp", self._base_path / "input", self._sim_file),
            )
            vs.EVAP_SOIL = update(
                vs.EVAP_SOIL,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("evap_soil", self._base_path / "input", self._sim_file),
            )
            vs.CPR_RZ = update(
                vs.CPR_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("cpr_rz", self._base_path / "input", self._sim_file),
            )
            vs.Q_RZ = update(
                vs.Q_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("q_rz", self._base_path / "input", self._sim_file),
            )
            vs.Q_SS = update(
                vs.Q_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("q_ss", self._base_path / "input", self._sim_file),
            )
            vs.S_RZ = update(
                vs.S_RZ,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("S_rz", self._base_path / "input", self._sim_file),
            )
            vs.S_SS = update(
                vs.S_SS,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("S_ss", self._base_path / "input", self._sim_file),
            )
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            vs.S_SNOW = update(
                vs.S_SNOW,
                at[2:-2, 2:-2, :],
                self._read_var_from_nc("S_snow", self._base_path / "input", self._sim_file),
            )

            if settings.enable_oxygen18:
                vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, 0], npx.nan)
                vs.C_ISO_IN = update(
                    vs.C_ISO_IN,
                    at[2:-2, 2:-2, 1:],
                    self._read_var_from_nc("d18O", self._input_dir, "forcing_tracer.nc"),
                )
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, :], delta_to_conc(state, vs.C_ISO_IN)[2:-2, 2:-2, :])

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables

            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], vs.PREC_DIST_DAILY[2:-2, 2:-2, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, vs.tau], vs.S_SNOW[2:-2, 2:-2, vs.itt])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            # mixing of isotopes while snow accumulation
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau],
                npx.where(
                    vs.S_snow[2:-2, 2:-2, vs.tau] > 0,
                    npx.where(
                        npx.isnan(vs.C_snow[2:-2, 2:-2, vs.tau]),
                        vs.C_in[2:-2, 2:-2],
                        (vs.prec[2:-2, 2:-2, vs.tau] / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau]))
                        * vs.C_in[2:-2, 2:-2]
                        + (
                            vs.S_snow[2:-2, 2:-2, vs.tau]
                            / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau])
                        )
                        * vs.C_snow[2:-2, 2:-2, vs.taum1],
                    ),
                    npx.nan,
                ),
            )
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau],
                npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] <= 0, npx.nan, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )
            vs.C_iso_snow = update(
                vs.C_iso_snow,
                at[2:-2, 2:-2, vs.tau],
                conc_to_delta(state, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )

            # mix isotopes from snow melt and rainfall
            vs.C_in = update(
                vs.C_in,
                at[2:-2, 2:-2],
                npx.where(
                    npx.isfinite(vs.C_snow[2:-2, 2:-2, vs.taum1]),
                    vs.C_snow[2:-2, 2:-2, vs.taum1],
                    npx.where(vs.prec[2:-2, 2:-2, vs.tau] > 0, vs.C_IN[2:-2, 2:-2, vs.itt], 0),
                ),
            )
            vs.C_iso_in = update(vs.C_iso_in, at[2:-2, 2:-2], conc_to_delta(state, vs.C_in[2:-2, 2:-2]))

        @roger_routine
        def set_diagnostics(self, state, base_path=_tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["average"].output_variables = ["C_iso_q_ss"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz", "sas_params_q_ss"]
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1
            if base_path:
                diagnostics["constant"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.C_snow = update(
            vs.C_snow,
            at[2:-2, 2:-2, vs.taum1],
            vs.C_snow[2:-2, 2:-2, vs.tau],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1],
            vs.prec[2:-2, 2:-2, vs.tau],
        )
        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1],
            vs.ta[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
            prec=vs.prec,
            C_snow=vs.C_snow,
            S_snow=vs.S_snow,
        )

    model = SVATOXYGEN18Setup()
    write_forcing_tracer(model._input_dir, "d18O")
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
from bmiroger import BmiRoger
from roger.bmimodels.oneD import ONEDSetup


def main():
    base_path = Path(__file__).parent
    # initialize the SVAT model of RoGeR
    model = ONEDSetup(base_path)
    # initialize the BMI interface of RoGeR
    interface = BmiRoger(model=model)
    interface.initialize(base_path)

    # run the model
    while interface.get_current_time() < interface.get_end_time():
        interface.update_until(interface._model._config["OUTPUT_FREQUENCY"])
        # get variables for coupling as numpy arrays
        perc = np.zeros(interface.get_grid_node_count())
        interface.get_value("q_ss", perc)
        # add here a groundwater model and use perc as upper boundary condition
        # update variables from another model
        # here we use a dummy value to set the groundwater level
        z_gw = np.zeros(interface.get_grid_node_count())
        z_gw[:] = 12.0
        interface.set_value("z_gw", z_gw)
    interface.finalize()


if __name__ == "__main__":
    main()

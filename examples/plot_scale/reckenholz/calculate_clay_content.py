    
from pathlib import Path
import numpy as np
import pandas as pd


theta_pwp = 0.189
theta_fc = 0.189 + 0.1247
theta_sat = 0.189 + 0.1247 + 0.1062

# calculate pore-size distribution index
lambda_bc = (
            np.log(theta_fc / theta_sat)
            - np.log(theta_pwp/ theta_sat)
        ) / (np.log(15850) - np.log(63))

# calculate bubbling pressure
ha = ((theta_pwp / theta_sat) ** (1.0 / lambda_bc) * (-15850))

# calculate soil water content at pF = 6
theta_6 = ((ha / (-(10**6))) ** lambda_bc * theta_sat)

# calculate clay content
clay = (0.71 * (theta_6 - 0.01) / 0.3)

print(clay)
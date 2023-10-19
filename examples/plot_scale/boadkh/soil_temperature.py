from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

base_path = Path(__file__).parent

# load data
file = base_path / "input" / "freiburg" / "TA.txt"
df_ta = pd.read_csv(
    file,
    sep=r"\s+",
    skiprows=0,
    header=0,
    parse_dates=[[0, 1, 2, 3, 4]],
    index_col=0,
    na_values=-9999,
)
df_ta.index = pd.to_datetime(df_ta.index, format="%Y %m %d %H %M")
df_ta.index = df_ta.index.rename("Index")
df = df_ta.resample("Y").mean()
df_ta_year = pd.DataFrame(index=df_ta.index)
# df_ta_year = df_ta_year.join(df).bfill()
df_ta_year = df_ta.rolling('364D', min_periods=364).mean().bfill()
df_a = pd.DataFrame(index=df_ta.index)
df_a.loc[:, 'A'] = df_ta.iloc[:, 0].values - df_ta_year.iloc[:, 0].values
df_a_year = df_a.rolling('364D', min_periods=364).mean().bfill()

a_year = np.abs(df_a.iloc[:, 0].values)
ta_year = df_ta_year.iloc[:, 0].values
doy = df_ta.index.dayofyear.values
theta = np.random.uniform(0.6,1.0,len(a_year))

z_soil = 0.6
phi_soil = 91
damp_soil = 0.15

df_ta_soil = pd.DataFrame(index=df_ta.index)
df_ta_soil.loc[:, 'temp_soil'] = ta_year + a_year * np.exp((-0.5 * z_soil)/damp_soil) * np.sin((2 * np.pi) * (doy / 365) - (2 * np.pi) * (phi_soil/365)/2 - ((0.5 * z_soil)/damp_soil))
df_ta_soil.loc[:, 'temp_soil1'] = ta_year + a_year * np.exp((-0.5 * z_soil)/(damp_soil*theta)) * np.sin((2 * np.pi) * (doy / 365) - (2 * np.pi) * (phi_soil/365)/2 - ((0.5 * z_soil)/(damp_soil*theta)))


fig, axes = plt.subplots(figsize=(6, 2))
# axes.plot(df_ta.index, df_ta["TA"], ls="-", color="red", lw=1)
axes.plot(df_ta_soil.index, df_ta_soil["temp_soil"], ls="--", color="black", lw=1)
axes.plot(df_ta_soil.index, df_ta_soil["temp_soil1"], ls="--", color="blue", lw=1)
axes.set_ylabel("Temperature [°C]")
axes.set_xlabel("Time [year]")
fig.tight_layout()
file = base_path / "figures" / "soil_temperature.png"
fig.savefig(file, dpi=300)
plt.close("all")
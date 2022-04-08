import pandas as pd
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context('paper', font_scale=1.5)


def join_obs_on_sim(idx, sim_vals, df_obs, rm_na=False):
    """Join observed values on simulated values.

    Args
    ----------
    idx : pd.DatetimeIndex
        time index

    sim_vals : onp.ndarray
        simulated values

    df_obs : pd.DataFrame
        observed values

    rm_na : boolean, optional
        whether NaNs are removed. default is False.

    Returns
    ----------
    df : pd.DataFrame
        DataFrame containing simulated and observed values.
    """
    # dataframe with simulated values
    df_sim = pd.DataFrame(data=sim_vals, index=idx, columns=['sim'])

    # dataframe with observed values
    df = pd.DataFrame(index=idx)
    df_obs = df.join(df_obs)
    df_obs.columns = ['obs']

    df = df_sim.join(df_obs)

    if rm_na:
        df = df.dropna()

    return df


def plot_obs_sim(df, y_lab='', fmt_x='num', ls_obs='line', x_lab='Time'):
    """Plot observed and simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    fmt_x : str, optional
        Format of x-axis. Default is numerical ('num'). Alternatively, date
        format can be used ('date').

    ls_obs : str, optional
        linestyle of observations

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    if fmt_x == 'num':
        # convert datetime index to numerical index
        idx = time_to_num(df.index, time='days')
    elif fmt_x == 'date':
        idx = df.index

    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9,4))
    if (ls_obs == 'line'):
        axs.plot(idx, df.iloc[:, 1], lw=1.5, color='blue', alpha=0.5)
    axs.scatter(idx, df.iloc[:, 1], color='blue', s=1, alpha=0.5)
    axs.plot(idx, df.iloc[:, 0], lw=1, ls='-.', color='red')
    axs.set_xlim((idx[0], idx[-1]))
    if y_lab == r'$\delta^{18}$O [permil]':
        axs.set_ylim((-20, 5))
    elif y_lab == r'$\delta^{2}$H [permil]':
        axs.set_ylim((-160, -20))
    axs.set_ylabel(y_lab)
    axs.set_xlabel(x_lab)
    fig.tight_layout()

    return fig


def plot_obs_sim_year(df, y_lab, start_month_hyd_year=10, ls_obs='line', x_lab='Time'):
    """Plot observed and simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    start_month_hyd_year : int, optional
        starting month of hydrologic year

    ls_obs : str, optional
        linestyle of observations

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    figs : list
        list with figures
    """
    figs = []
    df = assign_hyd_year(df.copy(), start_month_hyd_year=start_month_hyd_year)
    years = pd.unique(df.hyd_year)
    for year in years:
        df_year = df.loc[df.hyd_year == year].copy()
        df_year.loc[df_year.isnull().any(axis=1)] = 0
        # plot observed and simulated values
        fig, axs = plt.subplots(figsize=(9,4))
        if (ls_obs == 'line'):
            axs.plot(df_year.index, df_year.iloc[:, 1], lw=1.5, color='blue', alpha=0.5)
        axs.scatter(df_year.index, df_year.iloc[:, 1], color='blue', s=1, alpha=0.5)
        axs.plot(df_year.index, df_year.iloc[:, 0], lw=1, ls='-.', color='red')
        axs.set_xlim((df_year.index[0], df_year.index[-1]))
        if y_lab == r'$\delta^{18}$O [permil]':
            axs.set_ylim((-20, 5))
        elif y_lab == r'$\delta^{2}$H [permil]':
            axs.set_ylim((-160, -20))
        axs.set_ylabel(y_lab)
        axs.set_xlabel(str(year))
        if (len(df_year.index) > 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        elif (len(df_year.index) < 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_obs_sim_cum(df, y_lab, fmt_x='num', x_lab='Time'):
    """Plot cumulated observed and simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    fmt_x : str, optional
        Format of x-axis. Default is numerical ('num'). Alternatively, date
        format can be used ('date').

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    if fmt_x == 'num':
        # convert datetime index to numerical index
        idx = time_to_num(df.index, time='days')
    elif fmt_x == 'date':
        idx = df.index

    df.loc[df.isna().any(axis=1)] = 0

    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9,4))
    axs.plot(idx, df.iloc[:, 1].cumsum(), lw=1.5, color='blue', alpha=0.5)
    axs.plot(idx, df.iloc[:, 0].cumsum(), lw=1, ls='-.', color='red')
    if ('PET' in df.columns.to_list()):
        axs.plot(idx, df.loc[:, 'PET'].cumsum(), lw=1.5, ls=':', color='silver')
    axs.set_xlim((idx[0], idx[-1]))
    axs.set_ylabel(y_lab)
    axs.set_xlabel(x_lab)
    fig.tight_layout()

    return fig


def plot_obs_sim_cum_year(df, y_lab, start_month_hyd_year=10, x_lab='Time'):
    """Plot cumulated observed and simulated values for each hydrologic year.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    start_month_hyd_year : int, optional
        starting month of hydrologic year

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    figs : list
        list with figures
    """
    figs = []
    df = assign_hyd_year(df.copy(), start_month_hyd_year=start_month_hyd_year)
    years = pd.unique(df.hyd_year)
    for year in years:
        df_year = df.loc[df.hyd_year == year].copy()
        df_year.loc[df_year.isnull().any(axis=1)] = 0
        # plot observed and simulated values
        fig, axs = plt.subplots(figsize=(6,4))
        axs.plot(df_year.index, df_year.iloc[:, 1].cumsum(), lw=1, color='blue')
        axs.plot(df_year.index, df_year.iloc[:, 0].cumsum(), lw=1, ls='-.', color='red')
        if ('PET' in df_year.columns.to_list()):
            axs.plot(df_year.index, df_year.loc[:, 'PET'].cumsum(), lw=1.5, ls=':', color='silver')
        axs.set_xlim((df_year.index[0], df_year.index[-1]))
        axs.set_ylabel(y_lab)
        axs.set_xlabel(str(year))
        if (len(df_year.index) > 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        elif (len(df_year.index) < 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_obs_sim_cum_year_facet(df, y_lab, start_month_hyd_year=10, x_lab='Time'):
    """Plot cumulated observed and simulated values for each hydrologic year.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    start_month_hyd_year : int, optional
        starting month of hydrologic year

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    fig : Figure
    """
    df = assign_hyd_year(df.copy(), start_month_hyd_year=start_month_hyd_year)
    df_cs = pd.DataFrame(index=df.index, columns=['sim', 'obs'])
    years = pd.unique(df.hyd_year)
    for year in years:
        df_cs.loc[(df.hyd_year == year), 'sim'] = df.loc[(df.hyd_year == year), 'sim'].cumsum()
        df_cs.loc[(df.hyd_year == year), 'obs'] = df.loc[(df.hyd_year == year), 'obs'].cumsum()
    df_cs.loc[df_cs.isnull().any(axis=1)] = 0

    # DataFrame from wide to long format
    df_sim = df_cs.iloc[:, 0].to_frame()
    df_sim.columns = ['sim_obs']
    df_sim['type'] = 'sim'
    df_obs = df_cs.iloc[:, 1].to_frame()
    df_obs.columns = ['sim_obs']
    df_obs['type'] = 'obs'
    df_sim_obs = pd.concat([df_sim, df_obs])
    df_sim_obs_long = pd.melt(df_sim_obs, id_vars=['type'], value_vars=['sim_obs'], ignore_index=False)
    df_sim_obs_long = assign_hyd_year(df_sim_obs_long.copy(), start_month_hyd_year=start_month_hyd_year)
    df_sim_obs_long['time'] = df_sim_obs_long.index
    df_sim_obs_long.loc[df_sim_obs_long.isnull().any(axis=1)] = 0
    df_sim_obs_long = df_sim_obs_long.drop(columns=['variable'])
    df_sim_obs_long = df_sim_obs_long.astype(dtype= {"type" : str,
                                                     "value" : onp.float64,
                                                     "hyd_year" : onp.int64,
                                                     "time" : onp.datetime64})
    df_sim_obs_long.index = range(len(df_sim_obs_long.index))
    # Plot the lines on facets
    g = sns.relplot(
        data=df_sim_obs_long,
        x="time", y="value",
        hue="type", col="hyd_year",
        kind="line", palette=["r", "b"],
        facet_kws=dict(sharex=False),
        height=4, aspect=.7, col_wrap=4
    )
    g.set_ylabels(y_lab)
    g.set_xlabels(x_lab)
    g.set_xticklabels(rotation=90)
    for axs in g.axes.flatten():
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    g.set_titles(template='{col_name}')
    g._legend.remove()
    g.tight_layout()
    fig = g.fig

    return fig


def time_to_num(idx, time='days'):
    """Convert DatetimeIndex to numeric range. Conversion is based either on
    days or hours.

    Args
    ----------
    idx : pd.DatetimeIndex
        variable time index

    time : str, optional
        time unit

    Returns
    ----------
    idx_num : onp.array
        numerical date range
    """
    if time == 'days':
        idx_num = idx.to_series().diff().astype('timedelta64[m]').cumsum()/1440
        idx_num.iloc[0] = 0
        idx_num = idx_num.values

    elif time == 'hours':
        idx_num = idx.to_series().diff().astype('timedelta64[m]').cumsum()/60
        idx_num.iloc[0] = 0
        idx_num = idx_num.values

    return idx_num


def assign_hyd_year(df, start_month_hyd_year=10):
    r"""
    Assign hydrologic year.

    Parameters
    ----------
    df : DataFrame
        contains hydrologic values

    start_month_hyd_year : int, optional
        starting month of hydrologic year

    Returns
    ----------
    DataFrame
        contains hydrologic values and a column with the assigned hydrologic
        year
    """
    df.loc[:, 'hyd_year'] = df.index.year
    df.loc[(df.index.month >= start_month_hyd_year), 'hyd_year'] += 1

    return df

def assign_seasons(df):
    r"""
    Assign seasons.

    Parameters
    ----------
    df : DataFrame
        contains hydrologic values

    Returns
    ----------
    DataFrame
        contains hydrologic values and a column with the assigned seasons
    """
    idx_winter = (df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2)
    idx_spring = (df.index.month == 3) | (df.index.month == 4) | (df.index.month == 5)
    idx_summer = (df.index.month == 6) | (df.index.month == 7) | (df.index.month == 8)
    idx_autumn = (df.index.month == 9) | (df.index.month == 10) | (df.index.month == 11)
    df.loc[idx_winter, 'seas'] = 'DJF'
    df.loc[idx_spring, 'seas'] = 'MAM'
    df.loc[idx_summer, 'seas'] = 'JJA'
    df.loc[idx_autumn, 'seas'] = 'SON'

    return df


def calc_api(prec, w, k):
    r"""
    Calculate antecedent precipitation index (API).

    Parameters
    ----------
    prec : (N,)array_like
        precipitation values

    w : int
        window width

    k : float
        decay constant ranges between 0.8 and 0.98

    Returns
    ----------
    api : (N,)array_like
        antecedent precipitation index
    """
    api = onp.empty(prec.shape)
    api.fill(onp.nan)
    weights = k**onp.arange(1, w+1)[::-1]
    for i in range(w+1, api.shape[0]):
        api[i] = onp.sum(prec[(i-w):i] * weights)
    return api


def calc_napi(prec, w, k):
    r"""
    Calculate normalized antecedent precipitation index (NAPI).

    Parameters
    ----------
    prec : (N,)array_like
        precipitation values

    w : int
        window width

    k : float
        decay constant ranges between 0.8 and 0.98

    Returns
    ----------
    api : (N,)array_like
        antecedent precipitation index
    """
    napi = onp.empty(prec.shape)
    napi.fill(onp.nan)
    weights = k**onp.arange(0, w+1)[::-1]
    weights_sum = onp.sum(k**onp.arange(1, w+1)[::-1])
    for i in range(w+1, napi.shape[0]):
        api = onp.sum(prec[(i-w):i+1] * weights)
        api_mean = onp.mean(prec[(i-w):i]) * weights_sum
        napi[i] = api / api_mean
    return napi

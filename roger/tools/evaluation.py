import pandas as pd
import numpy as onp
import scipy as sp
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


def plot_sim(df, y_lab='', ls_obs='line', x_lab='Time', ylim=None):
    """Plot simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    ls_obs : str, optional
        linestyle of observations

    x_lab : str, optional
        label of x-axis

    ylim : tuple, optional
        y-axis limits

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9, 4))
    axs.plot(df.index, df.iloc[:, 0], lw=1, ls='-', color='black')
    axs.set_xlim((df.index[0], df.index[-1]))
    if ylim:
        axs.set_ylim(ylim)
    axs.set_ylabel(y_lab)
    axs.set_xlabel(x_lab)
    fig.tight_layout()

    return fig


def plot_sim_cum(df, y_lab='', ls_obs='line', x_lab='Time', ylim=None):
    """Plot simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    ls_obs : str, optional
        linestyle of observations

    x_lab : str, optional
        label of x-axis

    ylim : tuple, optional
        y-axis limits

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9, 4))
    axs.plot(df.index, df.iloc[:, 0].cumsum(), lw=1, ls='-', color='black')
    axs.set_xlim((df.index[0], df.index[-1]))
    if ylim:
        axs.set_ylim(ylim)
    axs.set_ylabel(y_lab)
    axs.set_xlabel(x_lab)
    fig.tight_layout()

    return fig


def plot_obs_sim(df, y_lab='', ls_obs='line', x_lab='Time', ylim=None):
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

    ylim : tuple, optional
        y-axis limits

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9, 4))
    if (ls_obs == 'line'):
        axs.plot(df.index, df.iloc[:, 1], lw=1.5, color='blue', alpha=0.5)
    axs.scatter(df.index, df.iloc[:, 1], color='blue', s=1, alpha=0.5)
    axs.plot(df.index, df.iloc[:, 0], lw=1, ls='-.', color='red')
    axs.set_xlim((df.index[0], df.index[-1]))
    if ylim:
        axs.set_ylim(ylim)
    axs.set_ylabel(y_lab)
    axs.set_xlabel(x_lab)
    fig.tight_layout()

    return fig


def plot_obs_sim_year(df, y_lab, start_month_hyd_year=10, ls_obs='line', x_lab='Time', ylim=None):
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

    ylim : tuple, optional
        y-axis limits

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
        if ylim:
            axs.set_ylim(ylim)
        axs.set_ylabel(y_lab)
        axs.set_xlabel(str(year))
        if (len(df_year.index) > 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        elif (len(df_year.index) < 120):
            axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_obs_sim_cum(df, y_lab, x_lab='Time'):
    """Plot cumulated observed and simulated values.

    Args
    ----------
    df : pd.DataFrame
        Dataframe with simulated and observed values

    y_lab : str
        label of y-axis

    x_lab : str, optional
        label of x-axis

    Returns
    ----------
    fig : Figure
        Plot for observed and simulated values
    """
    df.loc[df.isna().any(axis=1)] = 0

    # plot observed and simulated values
    fig, axs = plt.subplots(figsize=(9, 4))
    axs.plot(df.index, df.iloc[:, 1].cumsum(), lw=1.5, color='blue', alpha=0.5)
    axs.plot(df.index, df.iloc[:, 0].cumsum(), lw=1, ls='-.', color='red')
    if ('PET' in df.columns.to_list()):
        axs.plot(df.index, df.loc[:, 'PET'].cumsum(), lw=1.5, ls=':', color='silver')
    axs.set_xlim((df.index[0], df.index[-1]))
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


def calc_rmse(obs, sim):
    """
    Root mean square error (RMSE)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        Root mean square error (RMSE)
    """
    eff = onp.sqrt(onp.mean((sim - obs)**2))

    return eff



def calc_mae(obs, sim):
    """
    Mean absolute error (MAE)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        Mean absolute error (MAE)
    """
    abs_err = onp.abs(sim - obs)
    eff = onp.mean(abs_err)

    return eff


def calc_mre(obs, sim):
    """
    Mean relative error (MRE)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        Mean relative error (MRE)
    """
    rel_err = (sim - obs) / obs
    eff = onp.mean(rel_err)

    return eff

def calc_mare(obs, sim):
    """
    Mean absolute relative error (MARE)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        Mean absolute relative error (MARE)
    """
    abs_err = onp.abs(sim - obs)
    rel_err = abs_err/obs
    eff = onp.mean(rel_err)

    return eff


def calc_ve(obs, sim):
    """
    Volumetric efficiency (VE)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        Volumetric efficiency (VE)
    """
    abs_err = onp.abs(sim - obs)
    sum_abs_err = onp.sum(abs_err)
    sum_obs = onp.sum(obs)
    eff = sum_abs_err/sum_obs

    return eff


def calc_rbs(obs, sim):
    """
    Relative bias of sums (RBS)

    Parameters
    ----------
    obs : (N,)array_like
        observed time series

    sim : (N,)array_like
        simulated time series

    Returns
    ----------
    eff : float
        relative bias of sums (RBS)
    """
    eff = (onp.sum(sim) - onp.sum(obs)) / onp.sum(obs)

    return eff


def calc_temp_cor(obs, sim, r="pearson"):
    """
    Calculate temporal correlation between observed and simulated
    time series.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    r : str, optional
        Either Spearman correlation coefficient ('spearman') or Pearson
        correlation coefficient ('pearson') can be used to describe the
        temporalcorrelation. The default is to calculate the Pearson
        correlation.

    Returns
    ----------
    temp_cor : float
        correlation between observed and simulated time series

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_temp_cor(obs, sim)
    0.8940281850583509
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    if r == "spearman":
        r = sp.stats.spearmanr(obs, sim)
        temp_cor = r[0]

        if onp.isnan(temp_cor):
            temp_cor = 0

    elif r == "pearson":
        r = sp.stats.pearsonr(obs, sim)
        temp_cor = r[0]

        if onp.isnan(temp_cor):
            temp_cor = 0

    return temp_cor


def calc_kge_beta(obs, sim):
    r"""
    Calculate the beta term of Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    kge_beta : float
        alpha value

    Notes
    ----------
    .. math::

        \beta = \frac{\mu_{sim}}{\mu_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_kge_beta(obs, sim)
    1.0980392156862746

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    # calculate alpha term
    obs_mean = onp.mean(obs)
    sim_mean = onp.mean(sim)
    kge_beta = sim_mean / obs_mean

    return kge_beta


def calc_kge_alpha(obs, sim):
    r"""
    Calculate the alpha term of the Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    kge_alpha : float
        alpha value

    Notes
    ----------
    .. math::

        \alpha = \frac{\sigma_{sim}}{\sigma_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge_alpha(obs, sim)
    1.2812057455166919

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    obs_std = onp.std(obs)
    sim_std = onp.std(sim)
    kge_alpha = sim_std / obs_std

    return kge_alpha


def calc_kge_gamma(obs, sim):
    r"""
    Calculate the gamma term of Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    kge_gamma : float
        gamma value

    Notes
    ----------
    .. math::

        \gamma = \frac{CV_{sim}}{CV_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge_gamma(obs, sim)
    1.166812375381273

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    obs_mean = onp.mean(obs)
    sim_mean = onp.mean(sim)
    obs_std = onp.std(obs)
    sim_std = onp.std(sim)
    obs_cv = obs_std / obs_mean
    sim_cv = sim_std / sim_mean
    kge_gamma = sim_cv / obs_cv

    return kge_gamma


def calc_kge(obs, sim, r="pearson", var="std"):
    r"""
    Calculate Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    r : str, optional
        Either Spearman correlation coefficient ('spearman'; Pool et al. 2018)
        or Pearson correlation coefficient ('pearson'; Gupta et al. 2009) can
        be used to describe the temporal correlation. The default is to
        calculate the Pearson correlation.

    var : str, optional
        Either coefficient of variation ('cv'; Kling et al. 2012) or standard
        deviation ('std'; Gupta et al. 2009, Pool et al. 2018) to describe the
        gamma term. The default is to calculate the standard deviation.

    Returns
    ----------
    eff : float
        Kling-Gupta-Efficiency

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge(obs, sim)
    0.683901305466148

    Notes
    ----------
    .. math::

        KGE = 1 - \sqrt{(\beta - 1)^2 + (\alpha - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\frac{\mu_{sim}}{\mu_{obs}} - 1)^2 + (\frac{\sigma_{sim}}{\sigma_{obs}} - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\beta - 1)^2 + (\gamma - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\frac{\mu_{sim}}{\mu_{obs}} - 1)^2 + (\frac{CV_{sim}}{CV_{obs}} - 1)^2 + (r - 1)^2}

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # calculate alpha term
    obs_mean = onp.mean(obs)
    sim_mean = onp.mean(sim)
    kge_beta = sim_mean / obs_mean

    # calculate KGE with gamma term
    if var == "cv":
        kge_gamma = calc_kge_gamma(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        eff = 1 - onp.sqrt(
            (kge_beta - 1) ** 2 + (kge_gamma - 1) ** 2 + (temp_cor - 1) ** 2
        )

    # calculate KGE with beta term
    elif var == "std":
        kge_alpha = calc_kge_alpha(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        eff = 1 - onp.sqrt(
            (kge_beta - 1) ** 2 + (kge_alpha - 1) ** 2 + (temp_cor - 1) ** 2
        )

    return eff

def calc_nse(obs, sim):
    r"""
    Calculate Nash-Sutcliffe-Efficiency (NSE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    eff : float
        Nash-Sutcliffe-Efficiency

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = onp.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = onp.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> nse.calc_nse(obs, sim)
    0.5648252536640361

    Notes
    ----------
    .. math::

        NSE = 1 - \frac{\sum_{t=1}^{t=T} (Q_{sim}(t) - Q_{obs}(t))^2}{\sum_{t=1}^{t=T} (Q_{obs}(t) - \overline{Q_{obs}})^2}


    References
    ----------
    Nash, J. E., and Sutcliffe, J. V.: River flow forecasting through
    conceptual models part I - A discussion of principles, Journal of
    Hydrology, 10, 282-290, 10.1016/0022-1694(70)90255-6, 1970.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    sim_obs_diff = onp.sum((sim - obs) ** 2)
    obs_mean = onp.mean(obs)
    obs_diff_mean = onp.sum((obs - obs_mean) ** 2)
    eff = 1 - (sim_obs_diff / obs_diff_mean)

    return eff

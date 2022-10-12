Analysis of Roger output
========================
In this tutorial, we will use `xarray <http://xarray.pydata.org/en/stable/>`__ and `matplotlib <https://matplotlib.org>`__ to load, analyze, and plot the model output. You can run these commands in `IPython <https://ipython.readthedocs.io/en/stable/>`__ or a `Jupyter Notebook <https://jupyter.org>`__. Just make sure to install the dependencies first::

  $ pip install xarray matplotlib netcdf4 cftime

The analysis below is conducted for a single year simulation of the :doc:`1D </reference/models/oneD>` setup from the :doc:`model gallery </reference/model-gallery>`.

Let's start by importing some packages:

.. ipython:: python

    import xarray as xr
    from cftime import num2date
    import numpy as np
    from pathlib import Path

Set the paths to the example data:

.. ipython:: python

    OUTPUT_FILES = {
        "rates": Path().absolute() / "_data" / "ONED.rates.nc",
        "collect": Path().absolute() / "_data" / "ONED.collect.nc",
        "maximum": Path().absolute() / "_data" / "ONED.maximum.nc",
    }

Most of the heavy lifting will be done by ``xarray``, which provides a data structure and API for working with labeled N-dimensional arrays. ``xarray`` datasets automatically keep track how the values of the underlying arrays map to locations in space and time, which makes them immensely useful for analyzing model output.

Load and aggregate rates
------------------------

In order to load our first output file and display its content execute the following two commands:

.. ipython:: python

    ds_rates = xr.open_dataset(OUTPUT_FILES["rates"])
    ds_rates

    # convert to datetime
    days = (ds_rates['Time'].values / np.timedelta64(24 * 60 * 60, "s"))
    date = num2date(days, units=f"days since {ds_rates['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_rates = ds_rates.assign_coords(Time=date)

Now, we select the time series of ``inf_mat`` and plot the daily data:

.. ipython:: python
    :okwarning:

    @savefig inf_mat_daily.png width=5in
    ds_rates["inf_mat"].isel(x=0, y=0).plot()

To compute the monthly sums and plot the monthly data:

.. ipython:: python
    :okwarning:

    @savefig inf_mat_monthly.png width=5in
    (
        ds_rates["inf_mat"]
        .isel(x=0, y=0).resample(Time='1M')
        .sum()
        .plot()
    )

#!/usr/bin/env python

import functools

import click


def get_mask_data(mask):
    import numpy as np

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def save_image(data, path):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.flipud(data)).convert("1").save(path)


def create_mask(infile, outfile, variable="catchment", scale=None):
    """Creates a mask image from a given netCDF file"""
    import numpy as np
    import h5netcdf

    with h5netcdf.File(infile, "r") as topo:
        catchment = np.array(topo.variables[variable])
    data = get_mask_data(catchment)
    save_image(data, outfile)


@click.command("roger-create-mask")
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--variable", default="z", help="Variable holding topography data (default: catchment)")
@click.option("-o", "--outfile", default="topography.png", help="Output filename (default: topography.png)")
@functools.wraps(create_mask)
def cli(*args, **kwargs):
    create_mask(**kwargs)

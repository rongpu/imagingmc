# Use tractor to compute NEA (in arcsec^2) as a function of seeing FWHM for REX objects
# shifter --image docker:legacysurvey/legacypipe:DR10.0.0 /bin/bash

import sys, os, glob, time, warnings, gc
import numpy as np
# import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from multiprocessing import Pool

from tractor import *
from tractor.galaxy import *
from tractor.sersic import *


def get_nea(shape_r):

    tim = Image(data=np.zeros((H, W)), invvar=np.ones((H, W))/(0.01**2),
                psf=PixelizedPSF(psfex_img))
    source = ExpGalaxy(PixPos(H//2, W//2), Flux(1.), GalaxyShape(shape_r/0.262, 1, 45.))
    tractor = Tractor([tim], [source])
    mod = tractor.getModelImage(0)

    nea = np.sum(mod)**2/np.sum(mod**2)  # in unit of number of pixels

    return nea


shape_r_list = np.arange(0., 5.01, 0.01)

for band in ['g', 'r', 'z']:

    output_path = '/global/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/nea_vs_fwhm_{}_1024.fits'.format(band)

    cat = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/stacked_psfex_{}_1024.fits'.format(band)))
    print(len(cat))

    mask = cat['fwhm_mean']!=0
    cat = cat[mask]
    print(len(cat))

    cat['nea'] = np.zeros((len(cat), len(shape_r_list)))

    for index in range(len(cat)):

        psfex_img = cat['psf_mask'][index]

        W, H = 161, 161

        n_processes = 32
        with Pool(processes=n_processes) as pool:
            nea_list = pool.map(get_nea, shape_r_list, chunksize=1)

        cat['nea'][index] = 0.262**2 * np.array(nea_list)  # in units of arcsec^2

    cat.remove_column('psf_mask')
    # cat.write()

    hdul = fitsio.FITS(output_path, mode='rw', clobber=True)
    hdul.write(None)
    data = np.array(cat.copy())
    hdr = {}
    hdr['R_MIN'] = shape_r_list.min()
    hdr['R_MAX'] = shape_r_list.max()
    hdr['R_DELTA'] = shape_r_list[1]-shape_r_list[0]
    hdul.write(data, header=hdr)
    hdul.close()


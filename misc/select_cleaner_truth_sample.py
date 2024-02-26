# Remove objects that have large sky residuals or are blended
from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits


cat = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean.fits'))
print(len(cat))

# Remove objects with large sky residuals
for band in ['g', 'r', 'i', 'z']:
    cat[band+'_sky'] = (cat['apflux_blobresid_'+band][:, -1]-cat['apflux_blobresid_'+band][:, -2]) / (np.pi*7**2-np.pi*5**2)
mask = (cat['g_sky']>-0.002) & (cat['g_sky']<0.002)
print(np.sum(mask)/len(mask))
mask &= (cat['r_sky']>-0.003) & (cat['r_sky']<0.003)
print(np.sum(mask)/len(mask))
mask &= (cat['z_sky']>-0.006) & (cat['z_sky']<0.006)
print(np.sum(mask)/len(mask))
mask_skyres = mask.copy()

# Remove blended objects
# fracflux<-0.01 objects are all close blends
mask = (cat['fracflux_g']>-0.01) & (cat['fracflux_g']<0.5)
print(np.sum(mask)/len(mask))
mask &= (cat['fracflux_r']>-0.01) & (cat['fracflux_r']<0.5)
print(np.sum(mask)/len(mask))
mask &= (cat['fracflux_z']>-0.01) & (cat['fracflux_z']<0.5)
print(np.sum(mask)/len(mask))
mask_fracflux = mask.copy()

mask = mask_skyres & mask_fracflux
print(np.sum(mask)/len(mask))

tt = Table()
tt['clean'] = mask.copy()
tt.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_cleaner.fits.gz', overwrite=False)

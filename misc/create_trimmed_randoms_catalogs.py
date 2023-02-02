# Create random catalogs with only the necessary columns to reduce read time

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

n_randoms_catalogs = 64
columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV', 'PHOTSYS']

randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
randoms_paths = randoms_paths[:n_randoms_catalogs]

for randoms_path in randoms_paths:
    output_path = '/global/cfs/cdirs/desi/users/rongpu/imaging_mc/trimmed_randoms/'+os.path.basename(randoms_path).replace('.fits', '-trim.fits')
    print(output_path)
    if os.path.isfile(output_path):
        continue
    cat = Table(fitsio.read(randoms_path, columns=columns))
    print(len(cat))
    cat.write(output_path)

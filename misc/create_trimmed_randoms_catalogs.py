# Create random catalogs with only the necessary columns to reduce read time

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
import healpy as hp

n_randoms_catalogs = 64
columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV', 'PHOTSYS', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']

randoms_paths = sorted(glob.glob('/dvs_ro/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
randoms_paths = randoms_paths[:n_randoms_catalogs]

ebv_nside = 256
ebv_map = Table(fitsio.read('/dvs_ro/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/kp3_maps/v1_desi_ebv_{}.fits'.format(ebv_nside)))
ebv_map = ebv_map[ebv_map['N_STAR_GR']>0]
ebv_map['EBV_DESI'] = ebv_map['EBV_DESI_GR']

for randoms_path in randoms_paths:
    output_path = '/global/cfs/cdirs/desi/users/rongpu/imaging_mc/trimmed_randoms/'+os.path.basename(randoms_path).replace('.fits', '-trim.fits')
    print(output_path)
    if os.path.isfile(output_path):
        continue
    cat = Table(fitsio.read(randoms_path, columns=columns))
    print(len(cat))
    cat.write(output_path)

    # Add DESI EBV
    cat['id'] = np.arange(len(cat))
    id_original = np.array(cat['id']).copy()
    cat['HPXPIXEL'] = hp.ang2pix(ebv_nside, cat['RA'], cat['DEC'], nest=False, lonlat=True)
    cat = join(cat, ebv_map[['HPXPIXEL', 'EBV_DESI']], keys='HPXPIXEL', join_type='left').filled(-99.)
    cat.sort('id')
    assert np.all(cat['id']==id_original)
    cat = cat[['EBV_DESI']]
    cat.write(output_path.replace('/dvs_ro/', '/global/').replace('-trim.fits', '-desi_ebv.fits'))


from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import healpy as hp


cat = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10-deep/cosmos/catalogs/cosmos.fits'))
cat['decam_id'] = np.arange(len(cat))
print(len(cat))

cat1 = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10-deep/cosmos/catalogs/cosmos_apflux_blobresid.fits'))
cat = hstack([cat, cat1])
print(len(cat))

elgmask = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10-deep/cosmos/catalogs/cosmos_elgmask_v1.fits.gz'))
cat = hstack([cat, elgmask], join_type='exact')
print(len(cat))

bad_pix = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/ism_mask/bad_pixels_v1_512_ring.fits'))
pix = hp.ang2pix(512, cat['ra'], cat['dec'], nest=False, lonlat=True)
mask = ~np.in1d(pix, bad_pix['HPXPIXEL'])
cat = cat[mask]
print(len(cat), np.sum(mask)/len(mask))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cat['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_g'])))-9) - 3.214*cat['ebv']
    cat['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_r'])))-9) - 2.165*cat['ebv']
    cat['galdepth_imag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_i'])))-9) - 1.592*cat['ebv']
    cat['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_z'])))-9) - 1.211*cat['ebv']

des_depths = {'g': 24.89, 'r': 24.7, 'z': 23.45}
des_depths['i'] = 24.0  # this is not the actual DES depth
min_depths = {}
for band in ['g', 'r', 'i', 'z']:
    min_depths[band] = des_depths[band] + 1.
print(min_depths)

mask = cat['elg_mask']==0
for band in ['g', 'r', 'i', 'z']:
    mask &= cat['galdepth_{}mag_ebv'.format(band)] > min_depths[band]
cat = cat[mask]
print(len(cat), np.sum(mask)/len(mask))

from desiutil import brick
tmp = brick.Bricks(bricksize=0.25)
cat['brickid'] = tmp.brickid(cat['ra'], cat['dec'])

cat.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean.fits', overwrite=False)

########################################################################################################################################

subs = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/data/deep_field_subsets/catalogs/cosmos_subsets_rongpu_dr10.fits'))
brickid_list = np.unique(subs['brickid'])

mask = np.in1d(cat['brickid'], subs['brickid'])
cat = cat[mask]
print(len(cat), np.sum(mask)/len(mask))

cat.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean_in_subsets.fits', overwrite=False)

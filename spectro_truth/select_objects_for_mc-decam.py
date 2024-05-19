from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits
import healpy as hp


cat = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_redshift_catalog-decam.fits'))

# mask = np.full(len(cat), True)
# plt.figure(figsize=(10, 10))
# plt.plot(cat['TARGET_RA'][mask], cat['TARGET_DEC'][mask], '.', ms=1, alpha=0.5)
# plt.grid(alpha=0.5)
# plt.gca().invert_xaxis()
# plt.show()

mask_oii = (cat['OII_FLUX']>0) & (cat['OII_FLUX_IVAR']>0)
mask_oii &= np.log10(cat['OII_FLUX'] * np.sqrt(cat['OII_FLUX_IVAR'])) > 0.9 - 0.2 * np.log10(cat['DELTACHI2'])
print(np.sum(mask_oii), np.sum(mask_oii)/len(mask_oii))
cat = cat[mask_oii]
print(len(cat))

# mask = np.full(len(cat), True)
# plt.figure(figsize=(10, 10))
# plt.plot(cat['TARGET_RA'][mask], cat['TARGET_DEC'][mask], '.', ms=1, alpha=0.5)
# plt.grid(alpha=0.5)
# plt.gca().invert_xaxis()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.hist(cat['Z'], 100, alpha=0.5, range=(-0.01, 1.79))
# plt.xlabel('Redshift')
# plt.show()

tt = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_cleaner.fits.gz'))
print(len(cat))
cat = join(cat, tt, join_type='inner')
print(len(cat))

# plt.figure(figsize=(10, 10))
# plt.plot(cat['TARGET_RA'], cat['TARGET_DEC'], '.', ms=1, alpha=0.5)
# plt.grid(alpha=0.5)
# plt.gca().invert_xaxis()
# plt.show()

mask = ~cat['bad_sky'].copy()
mask &= ~cat['blended'].copy()
print(np.sum(mask)/len(mask))
cat = cat[mask]
print(len(cat))

# plt.figure(figsize=(10, 10))
# plt.plot(cat['TARGET_RA'], cat['TARGET_DEC'], '.', ms=1, alpha=0.5)
# plt.grid(alpha=0.5)
# plt.gca().invert_xaxis()
# plt.show()

cat.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/mc/mc_decam_20240513.fits', overwrite=True)



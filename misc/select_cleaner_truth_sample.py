# Remove objects that have large sky residuals or are blended
from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits


cat = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean.fits'))
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

mask_good = mask_skyres & mask_fracflux
print(np.sum(mask_good)/len(mask_good))

################## Remove point sources using tarctor and HSC point-source classifications ##################

hsc = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', columns=['ra']))
print(len(hsc))
idx = np.where((hsc['ra']>140) & (hsc['ra']<160))[0]
hsc = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', rows=idx))
print(len(hsc))
sys.path.append(os.path.expanduser('~/git/Python/user_modules/'))
from match_coord import match_coord
idx1, idx2, d2d, d_ra, d_dec = match_coord(hsc['ra'], hsc['dec'], cat['ra'], cat['dec'], search_radius=1., plot_q=True)
cat['id'] = np.arange(len(cat))
hsc = hsc[idx1]
cat_psf = cat[idx2].copy()

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    cat_psf['gmag'] = 22.5 - 2.5*np.log10(cat_psf['flux_g']) - 3.214 * cat_psf['ebv']
    cat_psf['rmag'] = 22.5 - 2.5*np.log10(cat_psf['flux_r']) - 2.165 * cat_psf['ebv']
    cat_psf['imag'] = 22.5 - 2.5*np.log10(cat_psf['flux_i']) - 1.592 * cat_psf['ebv']
    cat_psf['zmag'] = 22.5 - 2.5*np.log10(cat_psf['flux_z']) - 1.211 * cat_psf['ebv']
    cat_psf['gfibermag'] = 22.5 - 2.5*np.log10(cat_psf['fiberflux_g']) - 3.214 * cat_psf['ebv']

dpsf = cat_psf['type']=='PSF'
hpsf = hsc['i_extendedness_value']==0
mask = dpsf & hpsf
# Apply color cuts along the stellar locus so we only flag stars
mask &= (cat_psf['gmag']-cat_psf['rmag'])>1.2*(cat_psf['rmag']-cat_psf['zmag'])+0.1
print('Tractor and HSC point sources:', np.sum(mask))
cat_psf = cat_psf[mask]

mask_good &= ~np.in1d(cat['id'], cat_psf['id'])
print(np.sum(mask_good)/len(mask_good))

###############################################################################################################

tt = Table()
tt['clean'] = mask_good.copy()
tt.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_cleaner.fits.gz', overwrite=False)

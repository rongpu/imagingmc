from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits
import healpy as hp

sys.path.append(os.path.expanduser('~/git/Python/user_modules/'))
from match_coord import match_coord


######################################################## DECam ########################################################

cat = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10-deep/cosmos/catalogs/cosmos.fits'))
cat['decam_id'] = np.arange(len(cat))
print(len(cat))

elgmask = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10-deep/cosmos/catalogs/cosmos_elgmask_v1.fits.gz'))
cat = hstack([cat, elgmask], join_type='exact')

# Depth cuts
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cat['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_g'])))-9) - 3.214*cat['ebv']
    cat['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_r'])))-9) - 2.165*cat['ebv']
    cat['galdepth_imag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_i'])))-9) - 1.592*cat['ebv']
    cat['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['galdepth_z'])))-9) - 1.211*cat['ebv']
des_depths = {'g': 24.89, 'r': 24.7, 'z': 23.45}
min_depths = {}
for band in ['g', 'r', 'z']:
    min_depths[band] = des_depths[band] + 1.
print(min_depths)
mask = np.full(len(cat), True)
for band in ['g', 'r', 'z']:
    mask &= cat['galdepth_{}mag_ebv'.format(band)] > min_depths[band]
cat = cat[mask]
print(len(cat), np.sum(mask)/len(mask))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    cat['gmag'] = 22.5 - 2.5*np.log10(cat['flux_g']) - 3.214 * cat['ebv']
    cat['rmag'] = 22.5 - 2.5*np.log10(cat['flux_r']) - 2.165 * cat['ebv']
    cat['zmag'] = 22.5 - 2.5*np.log10(cat['flux_z']) - 1.211 * cat['ebv']
    cat['imag'] = 22.5 - 2.5*np.log10(cat['flux_i']) - 1.592 * cat['ebv']
    cat['gfibermag'] = 22.5 - 2.5*np.log10(cat['fiberflux_g']) - 3.214 * cat['ebv']

radius = 1.2
idx1, idx2, d2d, d_ra, d_dec = match_coord([150.1166], [2.2058], cat['ra'], cat['dec'], search_radius=radius*3600, plot_q=False, keep_all_pairs=True)
cat = cat[idx2]
print(len(cat))

mask = cat['elg_mask'] & 2**0 == 0
print(np.sum(mask), np.sum(mask)/len(mask))
cat = cat[mask]

mask = cat['gfibermag']<24.5
mask &= cat['gfibermag']>19.5
mask &= (cat['gmag']-cat['rmag'])<0.9
mask &= (cat['gmag']-cat['rmag'])<1.32-0.7*(cat['rmag']-cat['zmag'])
print(np.sum(mask), np.sum(mask)/len(mask))
cat = cat[mask]

######################################################## HSC ########################################################

hsc = Table(fitsio.read('/global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', columns=['ra']))
idx = np.where((hsc['ra']>140) & (hsc['ra']<160))[0]
hsc = Table(fitsio.read('/global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', rows=idx))
hsc1 = Table(fitsio.read('/global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit_apflux_and_masks-reduced.fits', rows=idx, columns=['object_id', 'g_apertureflux_15_flux']))
# srun -N 1 -C cpu -c 256 -t 04:00:00 -q interactive python read_pixel_bitmask.py --tracer elg --input /global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits --output /global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced-elg_mask_v1.fits.gz
hsc_elgmask = Table(fitsio.read('/global/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced-elg_mask_v1.fits.gz', rows=idx))
print(len(hsc))
assert len(hsc)==len(hsc1) and np.all(hsc['object_id']==hsc1['object_id']) and len(hsc)==len(hsc_elgmask)
hsc = hstack([hsc, hsc1[['g_apertureflux_15_flux']], hsc_elgmask])
print(len(hsc))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hsc['gmag'] = -2.5 * np.log10(hsc['g_cmodel_flux']/3630.78) + 22.5 - hsc['a_g']
    hsc['rmag'] = -2.5 * np.log10(hsc['r_cmodel_flux']/3630.78) + 22.5 - hsc['a_r']
    hsc['imag'] = -2.5 * np.log10(hsc['i_cmodel_flux']/3630.78) + 22.5 - hsc['a_i']
    hsc['zmag'] = -2.5 * np.log10(hsc['z_cmodel_flux']/3630.78) + 22.5 - hsc['a_z']
    hsc['ymag'] = -2.5 * np.log10(hsc['y_cmodel_flux']/3630.78) + 22.5 - hsc['a_y']
    hsc['gfibermag'] = -2.5 * np.log10(hsc['g_apertureflux_15_flux']/3630.78) + 22.5 - hsc['a_g']

# radius = 2
# area = np.pi * radius**2
# idx1, idx2, d2d, d_ra, d_dec = match_coord([150.1166], [2.2058], hsc['ra'], hsc['dec'], search_radius=radius*3600, plot_q=False, keep_all_pairs=True)
# hsc = hsc[idx2]
# print(len(hsc))

mask = hsc['elg_mask'] & 2**0 == 0
print(np.sum(mask), np.sum(mask)/len(mask))
hsc = hsc[mask]

mask = hsc['gfibermag']<24.55
mask &= hsc['gfibermag']>19.5
mask &= (hsc['gmag']-hsc['rmag'])<0.8
mask &= (hsc['gmag']-hsc['rmag'])<1.22-0.7*(hsc['rmag']-hsc['zmag'])
print(np.sum(mask), np.sum(mask)/len(mask))
hsc = hsc[mask]

######################################################## Combine the two catalogs ########################################################

hsc.rename_column('object_id', 'hsc_id')

# remove duplicates in the combined catalog
# Remove any HSC target that is within 1 arcsec of a DECam target
idx1, idx2, d2d, d_ra, d_dec = match_coord(cat['ra'], cat['dec'], hsc['ra'], hsc['dec'], search_radius=1., plot_q=True, keep_all_pairs=True)
not_duplicate = ~np.in1d(np.arange(len(hsc)), idx2)
print(np.sum(not_duplicate)/len(not_duplicate))
hsc['is_target'] = not_duplicate.copy()

cat.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets-decam.fits')
hsc.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets-hsc.fits')

hsc = hsc[not_duplicate]

tt = vstack([cat[['ra', 'dec', 'decam_id']], hsc[['ra', 'dec', 'hsc_id']]]).filled(-99)
tt.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets.fits', overwrite=False)


# Create stacked PSFEx models as a function of PSF_FWHM

import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from multiprocessing import Pool


def get_psfex(index):

    # index = np.where(cat1['ccd_id_str']==ccd_id_str)[0][0]
    image_filename = cat1['image_filename'][index]
    psfex_filename = image_filename[:image_filename.find('.fits.fz')]+'-psfex.fits'
    psfex_path = os.path.join('/global/cfs/projectdirs/cosmo/work/legacysurvey/dr10/calib/psfex', psfex_filename)
    psfex = Table(fitsio.read(psfex_path, columns=['ccdname', 'psf_mask']))
    ii = np.where(psfex['ccdname']==cat1['ccdname'][index])[0][0]
    psf_mask = np.array(psfex['psf_mask'][ii][0])

    return psf_mask


n_max = 1024
psf_max = {'g': 4.0, 'r': 3.0, 'z': 3.}

ccd_columns = ['image_filename', 'expnum', 'ccdname', 'filter']
cat = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/survey-ccds-dr10-v4.fits', columns=ccd_columns))
cat['ccd_id_str'] = np.char.add(np.array(cat['expnum']).astype(str), cat['ccdname'])
print(len(cat))

psfex = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/dr10dev/misc/survey-ccds-dr10-v4-psfex-fwhm.fits',))
psfex['ccd_id_str'] = np.char.add(np.array(psfex['expnum']).astype(str), psfex['ccdname'])
mask = psfex['failure']==False
psfex = psfex[mask]
cat = join(cat, psfex[['ccd_id_str', 'psf_fwhm', 'median_psf_fwhm', 'moffat_alpha', 'moffat_beta', 'failure']], keys='ccd_id_str', join_type='inner')
print(len(cat))

dr9 = Table(fitsio.read('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-ccds-decam-dr9.fits.gz', columns=['expnum', 'ccdname', 'ccd_cuts']))
dr9['ccd_id_str'] = np.char.add(np.array(dr9['expnum']).astype(str), dr9['ccdname'])
print(len(dr9))
mask = dr9['ccd_cuts']==0
dr9 = dr9[mask]
print(len(dr9))

mask = np.in1d(cat['ccd_id_str'], dr9['ccd_id_str'])
cat = cat[mask]
print(len(cat))

cat['psf_fwhm_arcsec'] = cat['psf_fwhm']*0.262

cat.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/survey-ccds-dr10-v4-psfex-stacking.fits.gz')

cat = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/survey-ccds-dr10-v4-psfex-stacking.fits.gz'))

np.random.seed(18231)

for band in ['g', 'r', 'z']:

    print(band)

    psf_bins = np.arange(0, psf_max[band]+0.01, 0.01)

    mask0 = cat['filter']==band

    psfstack = Table()
    psfstack['fwhm_bin'] = (psf_bins[:-1]+psf_bins[1:])/2
    psfstack['fwhm_mean'] = 0.
    psfstack['psf_mask'] = np.zeros((len(psfstack), 63, 63))

    for index in range(len(psf_bins)-1):

        # if index%100==0:
        print(index, '/', len(psf_bins)-1)

        mask = mask0 & (cat['psf_fwhm_arcsec']>psf_bins[index]) & (cat['psf_fwhm_arcsec']<psf_bins[index+1])
        if np.sum(mask)==0:
            continue

        cat1 = cat[mask].copy()
        if len(cat1)>n_max:
            idx = np.random.choice(len(cat1), size=n_max, replace=False)
            cat1 = cat1[idx]
            # print(band, index, len(cat1))

        n_processes = 256
        with Pool(processes=n_processes) as pool:
            res = pool.map(get_psfex, np.arange(len(cat1)), chunksize=1)

        psfstack['fwhm_mean'][index] = np.mean(cat1['psf_fwhm_arcsec'])
        psfstack['psf_mask'][index] = np.mean(res, axis=0)

    psfstack.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/stacked_psfex_{}_{}.fits'.format(band, n_max))


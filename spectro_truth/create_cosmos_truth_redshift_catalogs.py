from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits
import healpy as hp


#################################### Combine redrock catalogs ####################################

columns_1 = ['TARGETID', 'CHI2', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'DELTACHI2']
columns_2 = ['TARGETID', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'MORPHTYPE', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'PARALLAX', 'EBV', 'FLUX_W1', 'FLUX_W2', 'FIBERFLUX_Z', 'MASKBITS', 'PHOTSYS', 'DESI_TARGET', 'BGS_TARGET', 'COADD_NUMEXP', 'COADD_EXPTIME', 'COADD_NUMNIGHT', 'COADD_NUMTILE', 'SCND_TARGET']
columns_4 = ['TARGETID', 'TSNR2_ELG', 'TSNR2_BGS', 'TSNR2_QSO', 'TSNR2_LRG']
emline_columns = ['TARGETID', 'OII_FLUX', 'OII_FLUX_IVAR', 'OIII_FLUX', 'OIII_FLUX_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR', 'HGAMMA_FLUX', 'HGAMMA_FLUX_IVAR', 'HDELTA_FLUX', 'HDELTA_FLUX_IVAR']

# cp /dvs_ro/cfs/cdirs/desicollab/users/raichoor/laelbg/schlafly2/healpix/tertiary38-thru20240313/emline* .
# cp /dvs_ro/cfs/cdirs/desicollab/users/raichoor/laelbg/schlafly2/healpix/tertiary38-thru20240313/redrock* .
# redrock_fns = glob.glob('/dvs_ro/cfs/cdirs/desicollab/users/raichoor/laelbg/schlafly2/healpix/tertiary38-thru20240313/redrock-*.fits')
redrock_fns = glob.glob('/dvs_ro/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/tertiary38-thru20240313/redrock-*.fits')

cat_all = []

for fn in redrock_fns:

    fns = glob.glob(os.path.join(fn, 'redrock*.fits'))

    tmp1 = Table(fitsio.read(fn, ext=1, columns=columns_1))
    tmp2 = Table(fitsio.read(fn, ext=2, columns=columns_2))
    tmp4 = Table(fitsio.read(fn, ext=4, columns=columns_4))

    emline_fn = fn.replace('redrock-', 'emline-')
    tmp5 = Table(fitsio.read(emline_fn, ext=1, columns=emline_columns))

    assert np.all(tmp1['TARGETID']==tmp2['TARGETID']) and np.all(tmp1['TARGETID']==tmp4['TARGETID']) and np.all(tmp1['TARGETID']==tmp5['TARGETID'])
    cat = tmp1.copy()
    cat = join(cat, tmp2, keys='TARGETID')
    cat = join(cat, tmp4, keys='TARGETID')
    cat = join(cat, tmp5, keys='TARGETID')
    cat['fn'] = os.path.basename(fn)
    cat_all.append(cat)

cat = vstack(cat_all)
print(len(cat))

mask = cat['COADD_FIBERSTATUS']==0
print(np.sum(mask), np.sum(mask)/len(mask))
cat = cat[mask]

cat['EFFTIME_LRG'] = cat['TSNR2_LRG'] * 12.15
mask = cat['EFFTIME_LRG']>800
print(np.sum(mask), np.sum(mask)/len(mask))
cat = cat[mask]

print(len(cat), len(np.unique(cat['TARGETID'])))

# Select objects targeted as ELG truth sample
fa = Table(fitsio.read('/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary/0038/tertiary-targets-0038-assign.fits'))
print('fa', len(fa))
mask = fa['TERTIARY_TARGET']=='ELG_PRINCIPAL'
fa = fa[mask]
print('fa', len(fa))
print(len(cat))
cat = join(cat, fa[['TARGETID', 'ORIG_ROW']], keys='TARGETID', join_type='inner')
print(len(cat))

parent = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets.fits'))
parent['ORIG_ROW'] = np.arange(len(parent))
cat = join(cat, parent[['ORIG_ROW', 'decam_id', 'hsc_id']], keys='ORIG_ROW', join_type='inner')
print(len(cat))

cat.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_redshift_catalog.fits', overwrite=True)

#################################### DECam catalog ####################################

decam = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets-decam.fits'))
decam = join(cat, decam, keys='decam_id', join_type='inner')
print(len(decam))
decam.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_redshift_catalog-decam.fits', overwrite=True)

#################################### HSC catalog ####################################

tt = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_targets-hsc.fits'))
mask = cat['hsc_id']!=-99
cat1 = join(cat[mask], tt, keys='hsc_id', join_type='inner')
print(len(cat1))

# # Remove objects already in the targeted sample
# mask = np.in1d(tt['hsc_id'], cat1['hsc_id'])
# print(np.sum(mask))
# tt = tt[~mask]

sys.path.append(os.path.expanduser('~/git/Python/user_modules/'))
from match_coord import match_coord

mask = cat['hsc_id']==-99
cat2 = cat[mask].copy()
idx1, idx2, d2d, d_ra, d_dec = match_coord(cat2['TARGET_RA'], cat2['TARGET_DEC'], tt['ra'], tt['dec'], search_radius=0.5, plot_q=True, keep_all_pairs=False)
cat2 = cat2[idx1]
cat2['hsc_id'] = tt['hsc_id'][idx2]
cat2 = join(cat2, tt, keys='hsc_id', join_type='inner')

hsc = vstack([cat1, cat2])
print(len(hsc), len(np.unique(hsc['hsc_id'])), len(hsc)-len(np.unique(hsc['hsc_id'])))
assert len(hsc)==len(np.unique(hsc['hsc_id']))

sys.path.append(os.path.expanduser('~/git/Python/user_modules/'))
from user_common import poly_fit_nd, poly_val_nd

X = np.column_stack([hsc['gmag']-hsc['rmag'], hsc['rmag']-hsc['imag'], hsc['imag']-hsc['zmag'], hsc['zmag']-hsc['ymag'], hsc['gmag']])

tmp = np.load('/global/u2/r/rongpu/notebooks/imaging_mc/spectroscopic_truth/hsc_decam_transformation/hsc_transform.npz')
coeffs_gr, powers_arr_gr, coeffs_rz, powers_arr_rz = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']
hsc['g-r_decam'] = poly_val_nd(X, coeffs_gr, powers_arr_gr)
hsc['r-z_decam'] = poly_val_nd(X, coeffs_rz, powers_arr_rz)

hsc.write('/global/cfs/cdirs/desicollab/users/rongpu/imaging_mc/spectro_truth/cosmos_spectro_truth_redshift_catalog-hsc.fits', overwrite=True)


# Simulate the errors in SFD using DESI EBV
# It takes ~1 hour on 1 Perlmutter node for repeats=64 and n_randoms_catalogs=32, which simulates 32*64*2500=5M per sq. deg. MC "objects".
# WORK IN PROGRESS

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import healpy as hp
from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline

from select_desi_targets import select_elg_simplified


n_processes = 128
repeats = n_processes * 2  # number of MC sims for each random point
n_randoms_catalogs = 28

apply_source_detection = False

nside = 128

# from https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients
ext_coeffs = {'u': 3.995, 'g': 3.214, 'r': 2.165, 'i': 1.592, 'z': 1.211, 'y': 1.064}

# from psfex_fwhm_vs_nea.ipynb
fwhm_scaling = {'g': 0.963, 'r': 0.960, 'z': 0.952}

mean = [0, 0, 0, 0]
# covariance matrix for not-PSF ELGs; from subset_vs_truth-elg-covariances.ipynb
# the components are flux_g, flux_r, flux_z, fiberflux_g
cov_matrix = np.array([[ 1.85954303,  0.76665273,  0.50608253,  0.96187622],
                       [ 0.76665273,  1.91858102,  0.52088174, -0.14843237],
                       [ 0.50608253,  0.52088174,  1.43089346, -0.1186636 ],
                       [ 0.96187622, -0.14843237, -0.1186636 ,  1.21517644]])

tmp_dir = '/pscratch/sd/r/rongpu/imaging_mc/tmp8/'
if not os.path.isdir(os.path.dirname(tmp_dir)):
    os.makedirs(os.path.dirname(tmp_dir))

get_nea = {}
for band in ['g', 'r', 'z']:
    nea_path = '/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/nea_vs_fwhm_{}_1024.fits'.format(band)
    nea = Table(fitsio.read(nea_path))
    nea_arr = np.array(nea['nea']).T
    hdr = fitsio.read_header(nea_path, ext=1)
    shape_r_grid = np.arange(hdr['R_MIN'], hdr['R_MAX']+hdr['R_DELTA'], hdr['R_DELTA'])
    fwhm_grid = np.array(nea['fwhm_bin'])
    get_nea[band] = RectBivariateSpline(shape_r_grid, fwhm_grid, nea_arr).ev

ebv_nside = 128
ebv_map = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/maps/dgr_map_combined_{}.fits'.format(ebv_nside)))
ebv_map = ebv_map[ebv_map['n_star']>0]

# Load ZP offset maps; RING orderinrg
dr9_south_offsets = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/data/gaia_dr3/misc/gaia_xp_dr9_south_offset_maps_{}.fits'.format(ebv_nside)))
dr9_north_offsets = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/data/gaia_dr3/misc/gaia_xp_dr9_north_offset_maps_{}.fits'.format(ebv_nside)))
mask = (dr9_north_offsets['DEC']>32.375) & (dr9_north_offsets['RA']>90) & (dr9_north_offsets['RA']<300)
dr9_north_offsets = dr9_north_offsets[mask]
mask = ~np.in1d(dr9_south_offsets['HPXPIXEL'], dr9_north_offsets['HPXPIXEL'])
dr9_south_offsets = dr9_south_offsets[mask]
dr9_offsets = vstack([dr9_south_offsets, dr9_north_offsets])
dr9_offsets.sort('HPXPIXEL')

ebv_map = join(ebv_map, dr9_offsets, keys='HPXPIXEL', join_type='inner')
print(len(ebv_map))

mask = np.isfinite(ebv_map['gmag_diff_median']) & np.isfinite(ebv_map['rmag_diff_median']) & np.isfinite(ebv_map['zmag_diff_median'])
ebv_map = ebv_map[mask]
print(np.sum(mask)/len(mask), len(ebv_map))

ebv_map['EBV_DESI'] = (ebv_map['delta_gr_wmean']-(ebv_map['gmag_diff_median']-ebv_map['rmag_diff_median']))/1.049


def detection(cat, nsigma=6.):

    detected = np.full(len(cat), False)
    for band in ['g', 'r', 'z']:
        detected |= cat['flux_'+band] * np.sqrt(cat['flux_ivar_'+band]) > nsigma

    # https://github.com/legacysurvey/legacypipe/blob/fe4fd85c665d808097d8f231269783841ea633bc/py/legacypipe/detection.py#L66-L92
    weights = {'flat': {'g': 1., 'r': 1., 'i': 1., 'z': 1.}, 'red': {'g': 2.5, 'r': 1., 'i': 0.632, 'z': 0.4}}
    for sedname in ['flat', 'red']:
        stacked_flux, stacked_ivar = np.zeros(len(cat)), np.zeros(len(cat))
        for band in ['g', 'r', 'z']:
            stacked_flux += cat['flux_'+band] * cat['flux_ivar_'+band] / weights[sedname][band]  # inverse variance-weighted stacking
            stacked_ivar +=cat['flux_ivar_'+band] / weights[sedname][band]**2
        stacked_flux /= stacked_ivar
        detected |= stacked_flux * np.sqrt(stacked_ivar) > nsigma

    return detected


def quicksim(truth, cat):
    '''
    Get simulated fluxe measurements.
    truth columns: flux_grz, fiberflux_g, shape_r
    cat columns: psfsize_grz, psfdepth_grz, ebv
    '''
    nea = {}
    for band in ['g', 'r', 'z']:
        nea[band] = np.zeros(len(truth), dtype='float32')
        nea[band][truth['is_psf']] = np.array(4 * np.pi * (cat['psfsize_'+band][truth['is_psf']]/2.3548)**2, dtype='float32')
        nea[band][~truth['is_psf']] = np.array(get_nea[band](truth['shape_r'][~truth['is_psf']], fwhm_scaling[band]*cat['psfsize_'+band][~truth['is_psf']]), dtype='float32')
    # nea = Table(nea)

    flux_err = {}
    for band in ['g', 'r', 'z']:
        pix_ivar = cat['psfdepth_'+band] * 4 * np.pi * (cat['psfsize_'+band]/2.3548)**2  # pixel-level inverse variance per arcsec^2
        flux_err[band] = np.sqrt(nea[band]/pix_ivar)

    noise = {}
    noise['g'], noise['r'], noise['z'], noise['gfiber'] = np.random.multivariate_normal(mean, cov_matrix, len(cat)).T

    sim = Table()

    fiberflux_bands = ['g']
    for band in fiberflux_bands:
        fiberflux_ratio = truth['fiberflux_{}_ec'.format(band)] / truth['flux_{}_ec'.format(band)]
        noise[band+'fiber'][truth['is_psf']] = noise[band][truth['is_psf']]  # Most PSF objects have fiberflux perfectly correlated with flux
        sim['fiberflux_'+band] = np.array(truth['fiberflux_{}_ec'.format(band)] / 10**(0.4*ext_coeffs[band]*cat['ebv_desi']) + noise[band+'fiber'] * fiberflux_ratio*flux_err[band], dtype='float32')

    for band in ['g', 'r', 'z']:
        sim['flux_'+band] = np.array(truth['flux_{}_ec'.format(band)] / 10**(0.4*ext_coeffs[band]*cat['ebv_desi']) + noise[band] * flux_err[band], dtype='float32')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for band in ['g', 'r', 'z']:
            sim[band+'mag'] = 22.5 - 2.5*np.log10(sim['flux_'+band]) - ext_coeffs[band] * cat['ebv_sfd']
        for band in ['g']:
            sim[band+'fibermag'] = 22.5 - 2.5*np.log10(sim['fiberflux_'+band]) - ext_coeffs[band] * cat['ebv_sfd']

    if apply_source_detection:
        for band in ['g', 'r', 'z']:
            sim['flux_ivar_'+band] = 1/flux_err[band]**2
        sim['detected'] = detection(sim)
        sim = sim[['gmag', 'rmag', 'zmag', 'gfibermag', 'detected']]
    else:
        sim = sim[['gmag', 'rmag', 'zmag', 'gfibermag']]

    # Add zero-point offset
    for band in ['g', 'r', 'z']:
        sim[band+'mag'] -= cat[band+'mag_diff_median']
    for band in ['g']:
        sim[band+'fibermag'] -= cat[band+'mag_diff_median']

    return sim


def elgsim(foo):
    '''
    Run ELG selection on the simulated fluxes.
    '''

    if len(cat)<=len(truth):
        replace = False
    else:
        replace = True

    # all worker processes inherit the same random seed; need to explicitly set different seeds
    np.random.seed((os.getpid() * int(time.time()*1000)) % 123456789)

    idx = np.random.choice(len(truth), size=len(cat), replace=replace)
    sim = quicksim(truth[idx], cat)

    elglop, elgvlo = select_elg_simplified(sim)

    if apply_source_detection:
        elglop &= sim['detected']
        elgvlo &= sim['detected']

    return elglop, elgvlo


time_start = time.time()

truth = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean.fits'))
print('truth', len(truth))

# extinction-corrected fluxes
for band in ['g', 'r', 'z']:
    truth['flux_{}_ec'.format(band)] = truth['flux_'+band]*10**(0.4*ext_coeffs[band]*truth['ebv'])
for band in ['g']:
    truth['fiberflux_{}_ec'.format(band)] = truth['fiberflux_'+band]*10**(0.4*ext_coeffs[band]*truth['ebv'])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    truth['gfibermag'] = 22.5 - 2.5*np.log10(truth['fiberflux_g']) - ext_coeffs['g'] * truth['ebv']

# This can be optimized significantly to increase the MC acceptance rate
mask = truth['gfibermag']<25.1
truth = truth[mask]
print('truth', len(truth))

truth['is_psf'] = truth['type']=='PSF'
print('PSF', np.sum(truth['is_psf']))

################# Use HSC photo-z's to apply redshift cuts #################

hsc = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', columns=['ra', 'dec']))

ramin, ramax, decmin, decmax = [148.88288012505353, 151.33493558588006, 0.7232354726772253, 3.3279329307424557]
mask = (hsc['ra']>ramin) & (hsc['ra']<ramax) & (hsc['dec']>decmin) & (hsc['dec']<decmax)
idx = np.where(mask)[0]

hsc_columns = ['ra', 'dec', 'demp_photoz_best']
hsc = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/target/analysis/truth/parent/hsc-pdr3-dud-no_mag_limit-reduced.fits', rows=idx, columns=hsc_columns))

sys.path.append(os.path.expanduser('~/git/Python/user_modules/'))
from match_coord import match_coord

idx1, idx2, d2d, d_ra, d_dec = match_coord(hsc['ra'], hsc['dec'], truth['ra'], truth['dec'], search_radius=1., plot_q=False)

truth = truth[idx2]
hsc = hsc[idx1]

truth['Z'] = hsc['demp_photoz_best']

mask = (truth['Z']>0.8) & (truth['Z']<=1.1)
# mask = (truth['Z']>1.1) & (truth['Z']<=1.6)
print('Redshift cut', np.sum(mask)/len(mask), np.sum(mask))
truth = truth[mask]

##############################################################################

# Only keep necessary columns to save memory
truth = truth[['flux_g_ec', 'flux_r_ec', 'flux_z_ec', 'fiberflux_g_ec', 'shape_r', 'is_psf']]

columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV', 'PHOTSYS']

# randoms_paths = sorted(glob.glob('/dvs_ro/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
randoms_paths = sorted(glob.glob('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/trimmed_randoms/randoms-[0-9]*-trim.fits'))
randoms_paths = randoms_paths[:n_randoms_catalogs]

print('Running MC...')

for randoms_path in randoms_paths:
    print(randoms_path)

    output_path = os.path.join(tmp_dir, 'mc_elg_'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1.fits'))
    if os.path.isfile(output_path):
        continue

    cat = Table(fitsio.read(randoms_path, columns=columns))
    cat1 = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/desi_mask/randoms/elgmask_v1/'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1.fits')))
    cat = hstack([cat, cat1])
    min_nobs = 1
    mask = (cat['NOBS_G']>=min_nobs) & (cat['NOBS_R']>=min_nobs) & (cat['NOBS_Z']>=min_nobs)
    mask &= (cat['PSFDEPTH_G']>0) & (cat['PSFDEPTH_R']>0) & (cat['PSFDEPTH_Z']>0)
    mask &= (cat['PSFSIZE_G']>0) & (cat['PSFSIZE_R']>0) & (cat['PSFSIZE_Z']>0)
    cat = cat[mask]
    print(len(cat))
    mask = cat['elg_mask']==0
    cat = cat[mask]
    print('ELG mask', len(cat))

    cat['HPXPIXEL'] = hp.ang2pix(ebv_nside, cat['RA'], cat['DEC'], nest=False, lonlat=True)
    # mask = np.in1d(cat['HPXPIXEL'], ebv_map['HPXPIXEL'])
    # cat = cat[mask]
    cat = join(cat, ebv_map[['HPXPIXEL', 'EBV_DESI', 'gmag_diff_median', 'rmag_diff_median', 'zmag_diff_median']], keys='HPXPIXEL', join_type='inner')
    print('Add EBV_DESI', len(cat))

    cat.rename_columns(cat.colnames, [ii.lower() for ii in cat.colnames])
    cat.rename_column('ebv', 'ebv_sfd')

    elglop_count, elgvlo_count = np.zeros(len(cat), dtype=int), np.zeros(len(cat), dtype=int)
    counter = repeats
    while counter>0:
        with Pool(processes=n_processes) as pool:
            res = pool.map(elgsim, np.zeros(np.minimum(counter, n_processes)))
        counter = np.maximum(counter-n_processes, 0)
        res = np.array(res, dtype=int)
        elglop_count += np.sum(res[:, 0], axis=0)
        elgvlo_count += np.sum(res[:, 1], axis=0)
        del res

    mc = Table()
    mc['ra'] = cat['ra']
    mc['dec'] = cat['dec']
    mc['photsys'] = cat['photsys']
    mc['elglop'] = elglop_count
    mc['elgvlo'] = elgvlo_count
    mc.write(output_path, overwrite=True)

print('MC done', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

################################################# Healpix density map ###########################################################

def healpix_stats(pix_idx):

    pix_list = pix_unique[pix_idx]

    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list
    hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, pix_list, nest=False, lonlat=True)
    hp_table['elglop'] = np.zeros(len(hp_table), dtype=int)
    hp_table['elgvlo'] = np.zeros(len(hp_table), dtype=int)

    for index in np.arange(len(pix_idx)):

        idx = pixorder[pixcnts[pix_idx[index]]:pixcnts[pix_idx[index]+1]]
        hp_table['elglop'][index] = np.sum(mc['elglop'][idx])
        hp_table['elgvlo'][index] = np.sum(mc['elgvlo'][idx])

    return hp_table


def read_mc_catalogs(randoms_path):
    output_path = os.path.join(tmp_dir, 'mc_elg_'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1.fits'))
    return Table(fitsio.read(output_path))


print('Loading MC catalogs...')
with Pool(processes=n_processes) as pool:
    res = pool.map(read_mc_catalogs, randoms_paths)
# # Remove None elements from the list
# for index in range(len(res)-1, -1, -1):
#     if res[index] is None:
#         res.pop(index)
mc = vstack(res)
print('MC catalogs loaded', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

print('Computing healpix map...')
for nside in [128, 256, 512]:
    pix_allobj = hp.pixelfunc.ang2pix(nside, mc['ra'], mc['dec'], nest=False, lonlat=True)
    pix_unique, pix_count = np.unique(pix_allobj, return_counts=True)
    pixcnts = pix_count.copy()
    pixcnts = np.insert(pixcnts, 0, 0)
    pixcnts = np.cumsum(pixcnts)
    pixorder = np.argsort(pix_allobj)
    # split among the processors
    pix_idx_split = np.array_split(np.arange(len(pix_unique)), n_processes)
    # start multiple worker processes
    with Pool(processes=n_processes) as pool:
        res = pool.map(healpix_stats, pix_idx_split)
    hp_table = vstack(res)
    hp_table.sort('HPXPIXEL')
    hp_table['n_randoms'] = pix_count

    hp_table.write('/global/cfs/cdirs/desi/users/rongpu/imaging_mc/mc/20230201/mc_elg_randoms_elgmask_v1_desi_ebv_and_zp_z_0.8_1.1_healpix_{}.fits'.format(nside), overwrite=False)

print('All done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))


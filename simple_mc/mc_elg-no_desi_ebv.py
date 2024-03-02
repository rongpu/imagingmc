# Do not use DESI EBV to avoid the noise from the DESI EBV map
# The results are used for studying the noise vs repeats / n_randoms_catalogs

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


time_start = time.time()

n_processes = 100
repeats = n_processes * 4  # number of MC sims for each random point
n_randoms_catalogs = 32

apply_source_detection = False
use_desi_ebv = True
count_truth = False

debug = True

hpx_output_dir = '/global/cfs/cdirs/desi/users/rongpu/imaging_mc/mc/20240227'
hpx_output_fn = 'mc_elg_desi_ebv'
tmp_dir = '/pscratch/sd/r/rongpu/imaging_mc/tmp0/'

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

get_nea = {}
for band in ['g', 'r', 'z']:
    nea_path = '/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/nea/nea_vs_fwhm_{}_1024.fits'.format(band)
    nea = Table(fitsio.read(nea_path))
    nea_arr = np.array(nea['nea']).T
    hdr = fitsio.read_header(nea_path, ext=1)
    shape_r_grid = np.arange(hdr['R_MIN'], hdr['R_MAX']+hdr['R_DELTA'], hdr['R_DELTA'])
    fwhm_grid = np.array(nea['fwhm_bin'])
    get_nea[band] = RectBivariateSpline(shape_r_grid, fwhm_grid, nea_arr).ev

if use_desi_ebv:
    ebv_nside = 256
    ebv_map = Table(fitsio.read('/dvs_ro/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/kp3_maps/v1_desi_ebv_{}.fits'.format(ebv_nside)))
    ebv_map = ebv_map[ebv_map['N_STAR_GR']>0]
    ebv_map['EBV_DESI'] = ebv_map['EBV_DESI_GR']


def print_debug(*msg):
    if debug:
        print('Debug:', *msg)


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
    sim['id'] = truth['id'].copy()

    # the fluxes are the observed flux without extinction correction
    fiberflux_bands = ['g']
    for band in fiberflux_bands:
        fiberflux_ratio = truth['fiberflux_{}_ec'.format(band)] / truth['flux_{}_ec'.format(band)]
        noise[band+'fiber'][truth['is_psf']] = noise[band][truth['is_psf']]  # Most PSF objects have fiberflux perfectly correlated with flux
        if use_desi_ebv:
            ebv_true = cat['ebv_desi']
        else:
            ebv_true = cat['ebv_sfd']
        sim['fiberflux_'+band] = np.array(truth['fiberflux_{}_ec'.format(band)] / 10**(0.4*ext_coeffs[band]*ebv_true) + noise[band+'fiber'] * fiberflux_ratio*flux_err[band], dtype='float32')

    for band in ['g', 'r', 'z']:
        sim['flux_'+band] = np.array(truth['flux_{}_ec'.format(band)] / 10**(0.4*ext_coeffs[band]*ebv_true) + noise[band] * flux_err[band], dtype='float32')

    # the magnitudes are the observed magnitudes after the SFD extinction correction
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
        sim = sim[['id', 'gmag', 'rmag', 'zmag', 'gfibermag', 'detected']]
    else:
        sim = sim[['id', 'gmag', 'rmag', 'zmag', 'gfibermag']]

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
    gc.collect()

    elglop, elgvlo = select_elg_simplified(sim)

    if apply_source_detection:
        elglop &= sim['detected']
        elgvlo &= sim['detected']

    if count_truth:
        idx_full = sim['id'].copy()
        argsort = np.argsort(idx_full)  # pre-sortint to speed up np.unique

        elglop1, elgvlo1, idx_full1 = elglop[argsort], elgvlo[argsort], idx_full[argsort]
        tcount = Table()
        tcount['id'], tcount['total'] = np.unique(idx_full1, return_counts=True)
        tmp = Table()
        tmp['id'], tmp['elglop_sel'] = np.unique(idx_full1[elglop1], return_counts=True)
        tcount = join(tcount, tmp, keys='id', join_type='left').filled(0)
        tmp = Table()
        tmp['id'], tmp['elgvlo_sel'] = np.unique(idx_full1[elgvlo1], return_counts=True)
        tcount = join(tcount, tmp, keys='id', join_type='left').filled(0)

        return elglop, elgvlo, tcount
    else:
        return elglop, elgvlo


truth = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_clean.fits'))
truth['id'] = np.arange(len(truth), dtype='int32')
print('truth', len(truth))

# additional quality cuts
cleaner = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/truth/cosmos_truth_cleaner.fits.gz'))
mask = cleaner['clean'].copy()
truth = truth[mask]
print('truth', len(truth))

# extinction-corrected fluxes
for band in ['g', 'r', 'z']:
    truth['flux_{}_ec'.format(band)] = truth['flux_'+band]*10**(0.4*ext_coeffs[band]*truth['ebv'])
for band in ['g']:
    truth['fiberflux_{}_ec'.format(band)] = truth['fiberflux_'+band]*10**(0.4*ext_coeffs[band]*truth['ebv'])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    truth['gfibermag'] = 22.5 - 2.5*np.log10(truth['fiberflux_g']) - ext_coeffs['g'] * truth['ebv']

mask = truth['gfibermag']<25.3
truth = truth[mask]
print('truth', len(truth))

truth['is_psf'] = truth['type']=='PSF'
print('PSF', np.sum(truth['is_psf']))

# Only keep necessary columns to save memory
truth = truth[['id', 'flux_g_ec', 'flux_r_ec', 'flux_z_ec', 'fiberflux_g_ec', 'shape_r', 'is_psf']]

columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV', 'PHOTSYS', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']

# randoms_paths = sorted(glob.glob('/dvs_ro/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
randoms_paths = sorted(glob.glob('/dvs_ro/cfs/cdirs/desi/users/rongpu/imaging_mc/trimmed_randoms/randoms-[0-9]*-trim.fits'))
randoms_paths = randoms_paths[:n_randoms_catalogs]

print('Running MC...')

if count_truth:
    tcount = truth[['id']].copy()  # truth count
    tcount['total'], tcount['elglop_sel'], tcount['elgvlo_sel'] = np.zeros((3, len(tcount)), dtype='int32')

for randoms_path in randoms_paths:
    print(randoms_path)

    if not os.path.isdir(os.path.dirname(tmp_dir)):
        os.makedirs(os.path.dirname(tmp_dir))

    output_path = os.path.join(tmp_dir, 'mc_elg_'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1.fits'))
    if os.path.isfile(output_path):
        continue

    print_debug('reading randoms', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
    cat = Table(fitsio.read(randoms_path, columns=columns))
    cat1 = Table(fitsio.read('/dvs_ro/cfs/cdirs/desi/users/rongpu/desi_mask/randoms/elgmask_v1/'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1.fits')))
    cat = hstack([cat, cat1])
    cat.rename_column('EBV', 'EBV_SFD')
    print_debug('done reading randoms', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

    if use_desi_ebv:
        cat1 = Table(fitsio.read(randoms_path.replace('-trim.fits', '-desi_ebv.fits')))
        cat = hstack([cat, cat1])
        mask = cat['EBV_DESI']!=-99.
        cat = cat[mask]
        print('Added DESI EBV', len(cat))
        print_debug('done adding DESI EBV', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

    min_nobs = 1
    mask = (cat['NOBS_G']>=min_nobs) & (cat['NOBS_R']>=min_nobs) & (cat['NOBS_Z']>=min_nobs)
    mask &= (cat['PSFDEPTH_G']>0) & (cat['PSFDEPTH_R']>0) & (cat['PSFDEPTH_Z']>0)
    mask &= (cat['PSFSIZE_G']>0) & (cat['PSFSIZE_R']>0) & (cat['PSFSIZE_Z']>0)
    cat = cat[mask]
    print('Basic quality cuts', len(cat))
    mask = cat['elg_mask']==0
    cat = cat[mask]
    print('ELG mask', len(cat))

    ####################### Remove shallow/bad regions #######################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_G'])))-9) - 3.214*cat['EBV_SFD']
        cat['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_R'])))-9) - 2.165*cat['EBV_SFD']
        cat['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_Z'])))-9) - 1.211*cat['EBV_SFD']
    # Remove Delta EBV outliers
    mask = cat['EBV_DESI'] - cat['EBV_SFD'] > -0.05
    mask &= cat['EBV_DESI'] - cat['EBV_SFD'] < 0.05
    print('Remove Delta EBV outliers', np.sum(mask)/len(mask))
    mask &= cat['EBV_SFD'] < 0.15
    print('Remove EBV outliers', np.sum(mask)/len(mask))
    mask &= cat['PSFSIZE_G'] < 2.4
    print('Remove PSFSIZE outliers', np.sum(mask)/len(mask))
    mask &= cat['PSFSIZE_R'] < 2.3
    print('Remove PSFSIZE outliers', np.sum(mask)/len(mask))
    mask &= cat['PSFSIZE_Z'] < 2.
    print('Remove PSFSIZE outliers', np.sum(mask)/len(mask))
    mask &= cat['galdepth_gmag_ebv'] > 23.75
    print('Remove GALDEPTH outliers', np.sum(mask)/len(mask))
    mask &= cat['galdepth_rmag_ebv'] > 23.15
    print('Remove GALDEPTH outliers', np.sum(mask)/len(mask))
    mask &= cat['galdepth_zmag_ebv'] > 22.65
    print('Remove GALDEPTH outliers', np.sum(mask)/len(mask))
    print('Fraction of area removed {:.1f}%'.format(np.sum(~mask)/len(mask)*100))
    cat = cat[mask]
    # Remove columns to save memory
    cat.remove_columns(['GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'])
    gc.collect()
    ###########################################################################

    cat.rename_columns(cat.colnames, [ii.lower() for ii in cat.colnames])

    elglop_count, elgvlo_count = np.zeros(len(cat), dtype='int16'), np.zeros(len(cat), dtype='int16')
    counter = repeats
    while counter>0:
        with Pool(processes=n_processes) as pool:
            print_debug('running the sims', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
            res = pool.map(elgsim, np.zeros(np.minimum(counter, n_processes)))
            print_debug('sims done', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
        counter = np.maximum(counter-n_processes, 0)
        if count_truth:
            res_stack = []
            for ii in range(len(res)):
                print_debug(ii+1, '/', len(res))
                elglop, elgvlo, tmp = res[ii]
                res_stack.append([elglop, elgvlo])
                tmp = join(tcount, tmp, keys='id', join_type='left').filled(0)
                tcount['total'] = tmp['total_1'] + tmp['total_2']
                tcount['elglop_sel'] = tmp['elglop_sel_1'] + tmp['elglop_sel_2']
                tcount['elgvlo_sel'] = tmp['elgvlo_sel_1'] + tmp['elgvlo_sel_2']
            res = np.array(res_stack, dtype='int16')
            print_debug('truth counting done', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
            del tmp, res_stack
            gc.collect()
        else:
            res = np.array(res, dtype='int16')
        elglop_count += np.sum(res[:, 0], axis=0)
        elgvlo_count += np.sum(res[:, 1], axis=0)
        print_debug('ELG counting done', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
        del res
        gc.collect()

    mc = Table()
    mc['ra'] = cat['ra']
    mc['dec'] = cat['dec']
    mc['photsys'] = cat['photsys']
    mc['elglop'] = np.array(elglop_count, dtype='int16')
    mc['elgvlo'] = np.array(elgvlo_count, dtype='int16')
    mc.write(output_path, overwrite=True)

tcount_path = os.path.join(tmp_dir, 'mc_elg_'+os.path.basename(randoms_path).replace('-trim.fits', '-elgmask_v1-tcount.fits'))
if count_truth:
    tcount.write(tcount_path, overwrite=True)

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

for nside in [128, 256, 512]:  # NSIDE of the output healpix map
    print('NSIDE', nside)
    output_path = os.path.join(hpx_output_dir, hpx_output_fn+'_{}.fits'.format(nside))

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

    if not os.path.isdir(os.path.dirname(hpx_output_fn)):
        os.makedirs(os.path.dirname(hpx_output_fn))
    hp_table.write(hpx_output_fn, overwrite=True)

print('All done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))


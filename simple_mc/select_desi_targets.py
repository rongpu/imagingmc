from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack
import fitsio


def select_lrg(cat, field='south'):

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    mask_quality = np.full(len(cat), True)

    mask_quality &= (cat['FLUX_IVAR_R'] > 0) & (cat['FLUX_R'] > 0)   # ADM quality in r.
    mask_quality &= (cat['FLUX_IVAR_Z'] > 0) & (cat['FLUX_Z'] > 0) & (cat['FIBERFLUX_Z'] > 0)   # ADM quality in z.
    mask_quality &= (cat['FLUX_IVAR_W1'] > 0) & (cat['FLUX_W1'] > 0)  # ADM quality in W1.

    mask_quality &= (cat['GAIA_PHOT_G_MEAN_MAG'] == 0) | (cat['GAIA_PHOT_G_MEAN_MAG'] > 18)  # remove bright GAIA sources

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    mask_quality &= cat['FIBERTOTFLUX_Z'] < 10**(-0.4*(17.5-22.5))

    # ADM observed in every band.
    mask_quality &= (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)

    # Apply masks
    maskbits = [1, 12, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    mask_quality &= mask_clean

    gmag = 22.5 - 2.5 * np.log10((cat['FLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    rmag = 22.5 - 2.5 * np.log10((cat['FLUX_R'] / cat['MW_TRANSMISSION_R']).clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10((cat['FLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10((cat['FLUX_W1'] / cat['MW_TRANSMISSION_W1']).clip(1e-7))
    zfibermag = 22.5 - 2.5 * np.log10((cat['FIBERFLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))

    mask_lrg = np.full(len(cat), True)

    if field=='south':
        mask_lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        mask_lrg &= zfibermag < 21.6                   # faint limit
        mask_lrg &= (gmag - w1mag > 2.9) | (rmag - w1mag > 1.8)  # low-z cuts
        mask_lrg &= (
            ((rmag - w1mag > (w1mag - 17.14) * 1.8)
             & (rmag - w1mag > (w1mag - 16.33) * 1.))
            | (rmag - w1mag > 3.3)
        )  # double sliding cuts and high-z extension
    else:
        mask_lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        mask_lrg &= zfibermag < 21.61                   # faint limit
        mask_lrg &= (gmag - w1mag > 2.97) | (rmag - w1mag > 1.8)  # low-z cuts
        mask_lrg &= (
            ((rmag - w1mag > (w1mag - 17.13) * 1.83)
             & (rmag - w1mag > (w1mag - 16.31) * 1.))
            | (rmag - w1mag > 3.4)
        )  # double sliding cuts and high-z extension

    mask_lrg &= mask_quality

    return mask_lrg


def select_elg(cat):

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    mask_quality = np.full(len(cat), True)

    mask_quality &= (cat['FLUX_IVAR_G'] > 0) & (cat['FLUX_G'] > 0) & (cat['FIBERFLUX_G'] > 0)
    mask_quality &= (cat['FLUX_IVAR_R'] > 0) & (cat['FLUX_R'] > 0)
    mask_quality &= (cat['FLUX_IVAR_Z'] > 0) & (cat['FLUX_Z'] > 0)

    # ADM observed in every band.
    mask_quality &= (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)

    # Apply masks
    maskbits = [1, 12, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    mask_quality &= mask_clean

    gmag = 22.5 - 2.5 * np.log10((cat['FLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    rmag = 22.5 - 2.5 * np.log10((cat['FLUX_R'] / cat['MW_TRANSMISSION_R']).clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10((cat['FLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    gfibermag = 22.5 - 2.5 * np.log10((cat['FIBERFLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))

    mask_elglop = np.full(len(cat), True)

    mask_elglop &= gmag > 20                       # bright cut.
    mask_elglop &= rmag - zmag > 0.15                  # blue cut.
    mask_elglop &= gfibermag < 24.1  # faint cut.
    mask_elglop &= gmag - rmag < 0.5*(rmag - zmag) + 0.1  # remove stars, low-z galaxies.

    mask_elgvlo = mask_elglop.copy()

    # ADM low-priority OII flux cut.
    mask_elgvlo &= gmag - rmag < -1.2*(rmag - zmag) + 1.6
    mask_elgvlo &= gmag - rmag >= -1.2*(rmag - zmag) + 1.3

    # ADM high-priority OII flux cut.
    mask_elglop &= gmag - rmag < -1.2*(rmag - zmag) + 1.3

    return mask_elglop, mask_elgvlo


def select_elg_simplified(cat):

    gmag = cat['gmag']
    rmag = cat['rmag']
    zmag = cat['zmag']
    gfibermag = cat['gfibermag']

    mask_quality = np.isfinite(gmag) & np.isfinite(rmag) & np.isfinite(zmag) & np.isfinite(gfibermag)

    mask_elglop = mask_quality.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        mask_elglop &= gmag > 20                       # bright cut.
        mask_elglop &= rmag - zmag > 0.15                  # blue cut.
        mask_elglop &= gfibermag < 24.1  # faint cut.
        mask_elglop &= gmag - rmag < 0.5*(rmag - zmag) + 0.1  # remove stars, low-z galaxies.

        mask_elgvlo = mask_elglop.copy()

        # ADM low-priority OII flux cut.
        mask_elgvlo &= gmag - rmag < -1.2*(rmag - zmag) + 1.6
        mask_elgvlo &= gmag - rmag >= -1.2*(rmag - zmag) + 1.3

        # ADM high-priority OII flux cut.
        mask_elglop &= gmag - rmag < -1.2*(rmag - zmag) + 1.3

    return mask_elglop, mask_elgvlo

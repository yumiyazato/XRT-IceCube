# Bulk Lorentz factor is valiable and set in this code
# rho is valiable and set as argv when run submit file 
# Lx is valiable and set in this code depending on rho (for smaller rho, bin step of larger Lx is relatively small)
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
import numpy as np
from astropy.io import fits
from scipy.integrate import quad
import pandas as pd
from scipy.integrate import dblquad
from scipy import interpolate
import scipy.stats as st
from scipy.special import gamma
from util.calc_diffuse_flux import GetDiffuseFlux
from signal_calculation import signal_calculation
import sys
from pathlib import Path

# scan all the fov with 10*10 window by 1 pixel and find centroid coordinates of excess spot 
# This is the first step of source detection
def centroid(filename):
    eventfile=fits.open(filename)
    events = eventfile['EVENTS'].data
    x = events['X'].astype(np.float64)
    y = events['Y'].astype(np.float64)
    xmin_hist, xmax_hist = x.min(), x.max()
    ymin_hist, ymax_hist = y.min(), y.max()
    bin_width = 1


    # bin boundary
    xedges = np.arange(xmin_hist, xmax_hist + bin_width, bin_width)
    yedges = np.arange(ymin_hist, ymax_hist + bin_width, bin_width)

    # histgram
    H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges])
    Hx, _, _   = np.histogram2d(x, y, bins=[xedges, yedges], weights=x)
    Hy, _, _   = np.histogram2d(x, y, bins=[xedges, yedges], weights=y)
    Hx2, _, _  = np.histogram2d(x, y, bins=[xedges, yedges], weights=x**2)
    Hy2, _, _  = np.histogram2d(x, y, bins=[xedges, yedges], weights=y**2)
 
    # H: histogram2d output  (xbins, ybins)
    A = H.T  # A.shape = (ybins, xbins) = (rows, cols)
    AX  = Hx.T
    AY  = Hy.T
    AX2 = Hx2.T
    AY2 = Hy2.T
    k = 10   # window size
    def ii(M):
        S = np.zeros((M.shape[0]+1, M.shape[1]+1), dtype=float)
        S[1:,1:] = M.cumsum(axis=0).cumsum(axis=1)
        return S
    SA   = ii(A)
    SAX  = ii(AX)
    SAY  = ii(AY)
    SAX2 = ii(AX2)
    SAY2 = ii(AY2)

   
    win_sum = (SA[k:, k:] - SA[:-k, k:] - SA[k:, :-k] + SA[:-k, :-k])  # shape (rows-k+1, cols-k+1)
    coords = np.argwhere(win_sum == win_sum.max())
    iy = coords[:, 0]
    ix = coords[:, 1]
    # 矩形和
    K  = SA[iy+k, ix+k] - SA[iy, ix+k] - SA[iy+k, ix] + SA[iy, ix]
    SX = SAX[iy+k, ix+k]- SAX[iy, ix+k]- SAX[iy+k, ix]+ SAX[iy, ix]
    SY = SAY[iy+k, ix+k]- SAY[iy, ix+k]- SAY[iy+k, ix]+ SAY[iy, ix]
    # find centrioid of the excess
    valid = K > 0
    iy, ix, K, SX, SY = iy[valid], ix[valid], K[valid], SX[valid], SY[valid]
    xc = SX / K
    yc = SY / K


    prec = 3  
    xy_r = np.column_stack([np.round(xc, prec), np.round(yc, prec)]) # delete dupulication

    # get representative index
    uniq_xy, uniq_idx = np.unique(xy_r, axis=0, return_index=True)


    iy_u = iy[uniq_idx]
    ix_u = ix[uniq_idx]
    xc_u = xc[uniq_idx]
    yc_u = yc[uniq_idx]
    K_u  = K[uniq_idx]


    return xc_u, yc_u, K_u 

# psf file 
psffile = fits.open('/home/yu/XRT/caldb/data/swift/xrt/cpf/psf/swxpsf20010101v006.fits')

PC_006 = psffile['PC_PSF_COEF'].data
parameters_6 = PC_006['COEF0']
p0 = parameters_6[0]
p1 = parameters_6[1]
p2 = parameters_6[2]
p3 = parameters_6[3]
def psf(r):
    psf = p0*np.exp(-(r**2/2/p1**2))+(1-p0)*(1+(r/p2)**2)**-p3
    return psf
def psf_xy(y, x):
    r=np.sqrt(x**2+y**2)
    return psf(r)

def psf_integral(r):
    p0 = parameters_6[0]
    p1 = parameters_6[1]
    p2 = parameters_6[2]
    p3 = parameters_6[3]
    psf = 2*np.pi*r*(p0*np.exp(-(r**2/2/p1**2))+(1-p0)*(1+(r/p2)**2)**-p3)
    return psf
total, _ = quad(psf_integral, 0, np.inf)


# function to get Xray side Likelihood 
def get_XLLH(evt_path):
    eventfile=fits.open(evt_path)
    events = eventfile['EVENTS'].data
    pi = events['PI']
    mask_pi = pi >= 40
    events =events[mask_pi]
    x = events['X']
    y = events['Y']
    xc_u, yc_u, K = centroid(evt_path)
    fitx_list = []
    fity_list = []
    L_sig_list = []
    L_sig_list = []
    L_bg_list = []
    difference_list = []
    for x_c, y_c in zip(xc_u, yc_u):
        xmax=int(x_c) + 30
        xmin=int(x_c) - 30
        ymax=int(y_c) + 30
        ymin=int(y_c) - 30
        omega = 61*61
        x_sel = x[(x<=xmax)&(xmin<=x)&(y<=ymax)&(ymin<=y)]
        y_sel = y[(x<=xmax)&(xmin<=x)&(y<=ymax)&(ymin<=y)]

        # initial value
        theta0 = np.array([x_c, y_c, 0.5])
        # boundary
        bounds = [(xmin, xmax),
                (ymin, ymax),
                (0.0, 1.0)]
        def XLLH(params):
            x_src, y_src, alpha = params
            dx=x_sel-x_src
            dy=y_sel-y_src
            
            result_vals = []
            for dx, dy in zip(dx, dy):
                r = np.hypot(dx, dy)
                if r < 10:
                    # double integral within ±0.5 pix 
                    val, _ = dblquad(
                        psf_xy,
                        dx - 0.5, dx + 0.5,  
                        lambda x: dy - 0.5,
                        lambda x: dy + 0.5
                    )
                else:
                    
                    val = psf(r)
                result_vals.append(val / total)  
            result_vals=np.array(result_vals)

            term = alpha *result_vals + (1-alpha) * (1 / omega)
            return -np.sum(np.log(term))
        # maximize by fitting alpha 
        res = minimize(
            XLLH, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 200}
        )
        L_sig = -res.fun
        L_bg=len(x_sel)*np.log((1/omega))
        L_sig_list.append(L_sig)
        L_bg_list.append(L_bg)
        difference_list.append(L_sig-L_bg)

        fitx, fity, _ = res.x
        fitx_list.append(fitx)
        fity_list.append(fity)
    
    L_sig_list=np.array(L_sig_list)
    L_bg_list=np.array(L_bg_list)
    fitx_list=np.array(fitx_list)
    fity_list=np.array(fity_list)
    difference_list = np.array(difference_list)
    L_sig_x = L_sig_list[np.argmax(difference_list)]
    L_bg_x = L_bg_list[np.argmax(difference_list)]
    fitx_x = fitx_list[np.argmax(difference_list)]
    fity_x = fity_list[np.argmax(difference_list)]
    
    return L_sig_x, L_bg_x, fitx_x, fity_x

#
alertlevel = 0 # As PDF, use GFU level
use_nuXunifiledmodel = True
if(use_nuXunifiledmodel):
    LZGamma = 3 # Lorentz Bulk factor  This is fixed
    LZGamma_ref = 5 # This can be 3, 5, 7, 10 (assumed Gamma in signal generattion)
    name_sigmodel = "nuX unifiled"
    fname_sigmodel = "sig_nuXuni"
else:
    name_sigmodel = "100% diffuse"
    fname_sigmodel = "sig_dif"
    
#--------------------  physics constants --------------------------------
T1d = 86400
T1yr = 365*T1d
keV = 1e-6 # in GeV
Mpc = 3.08568025e24 # cm
erg  = 6.242e2 # GeV
c = 2.99792e10 # cm/sec
Km = 1e5 # cm
H0 = 73.5*Km/Mpc # cm/sec/Mpc
omegaM = 0.3
omegaLambda = 0.7
arcmin = 1.0/60
#---------------- instances for computation -----------------------------
logfloatmax = np.int32(np.log(sys.float_info.max))
logfloatmin = np.int32(np.log(sys.float_info.min)+1)


def logsum(log1, log2):
    # calculate np.log(np.exp(log1)+np.exp(log2))
    return np.where(log1<log2, log2 + np.log1p(np.exp(log1-log2)), log1+np.log1p(np.exp(log2-log1)))

def pval2sigma(pval):
	return st.norm.ppf(1-pval)
def sigma2pval(sigma):
	return st.norm.sf(sigma)


if(use_nuXunifiledmodel):
    sigmodel = 1
else:
    sigmodel = 0
# This is used to make PDF
sc = signal_calculation(alertlevel=alertlevel, sigmodel=sigmodel)

# This is used to calculate redshift distribution 
sc_redshift = signal_calculation(alertlevel=2, sigmodel=sigmodel)


class DistanceCalculator(object):

    def __init__(self, zMin=1e-9, zMax=5, Nstep=1000):
        self.zMin = zMin
        self.zMax = zMax
        zs = np.append(0, np.logspace(np.log10(self.zMin), np.log10(self.zMax), Nstep))
        dz = zs[1:]-zs[:-1]
        vals = self.integrand(0.5*(zs[1:]+zs[:-1]))
        dists = np.append(0, np.cumsum(vals*dz)*c/H0/Mpc)
        self.dist_interporate =  interpolate.interp1d(zs, dists)
    
    def integrand(self, z):
        return 1/np.sqrt(omegaM*(1+z)*(1+z)*(1+z)+omegaLambda)

    def F(self, z):
        x = 1+z
        return np.sqrt(omegaM*x*x*x+omegaLambda)
    
    def getSingleSourceDistance(self, z):
        flg = np.where((z<0)|(z>self.zMax), 1, 0)
        z = np.where(flg==1, 0.1, z)
        return np.where(flg==1, -1, self.dist_interporate(z))
  
##-------------------------------------------------------------------------------------------

# suppress=3 : do not dump anything
# suppress=2 : only representative value
# suppress=1 : only representative value
# suppress=0 : for debug

suppress = 1
def myprint(string, flg):
    if(suppress>flg): return
    print(string)

#-----------------------------------------------------------------------

argv=[0]
# seed = 132
# if(len(argv)==1):
#     # seed=1321
#     Lx = 1e43
# elif(len(argv)==3):
#     Lx = float(argv[1])
#     seed = int(argv[2])
# else:
#     print("error: <Lx> <seed>")
#     sys.exit(1)
# myprint("seed={0}".format(seed),2)
# myprint("Lx={0:.2e}".format(Lx),2)
# np.random.seed(seed)

# alertlevel = 2 # gold/bronze
#alertlevel = 1 # signalness >= 0.1
alertlevel = 0 # all GFU 

if(alertlevel==2):
    Ntrial = 3000
    namelevel = "alert"
elif(alertlevel==1):
    Ntrial = 1000
    namelevel = "subalert"
elif(alertlevel==0):
    Ntrial = 200
    namelevel = "gfu"

myprint("event selection type = {0}".format(namelevel),2)
#----------------------- source related parameters ---------------------


Tlivetime = T1yr
dsc = DistanceCalculator()
zmax = 4
z = np.append(0, np.logspace(-8, np.log10(zmax),  200))
dz = z[1:]-z[:-1]
zc = 0.5*(z[1:]+z[:-1])
D = dsc.getSingleSourceDistance(z)
Dc = 0.5*(D[1:]+D[:-1])


def get_SFR(z):
    return np.where(z<1, (1+z)**3.4, (1+1)**3.4)

#-----------------------------------------------------------------------------------------
# neutrino related parameters
# signal model : whole diffuse nu is associated with x-rays

if(use_nuXunifiledmodel):
    sigmodel = 1
else:
    sigmodel = 0

# This is used to make PDF
sc = signal_calculation(alertlevel=alertlevel, sigmodel=sigmodel)

# This is used to calculate redshift distribution 
sc_redshift = signal_calculation(alertlevel=2, sigmodel=sigmodel)


gamma = 2.37
Enu_ref = 1e51 # 

phi0_ref = 1.44e-18
Enu_bolmax = 1e6
Enu_bolmin = 1e4

Tw_inday = 10**4.5/86400.0

if(gamma!=2):
    norm = (2-gamma)/(Enu_bolmax**(2-gamma)-Enu_bolmin**(2-gamma))
else:
    norm = 1/np.log(Enu_bolmax/Enu_bolmin)
   
pdf_atmos = sc.get_atmospdf()
atmos_tot = sc.get_atmos_tot()
myprint("Ntot (atmos numu) = {0:.2e} [1/yr]".format(atmos_tot*T1yr),2)

pdf_binsindec, pdf_binlog10E = sc.get_pdfbin()
dpdf_binlog10E = pdf_binlog10E[1:] - pdf_binlog10E[:-1]
pdf_bincsindec = 0.5*(pdf_binsindec[1:]+pdf_binsindec[:-1])
pdf_binclog10E = 0.5*(pdf_binlog10E[1:]+pdf_binlog10E[:-1])
pdf_bincsindec2d, pdf_binclog10E2d = np.meshgrid(pdf_bincsindec, pdf_binclog10E, indexing="ij") 

if(use_nuXunifiledmodel):
    sc.set_LZGamma(LZGamma)
    sc_redshift.set_LZGamma(LZGamma)
    sig_tot, dif_tot = sc.get_nuXuni_tot(LZGamma)
    myprint("Ntot (signal numu) = {0:.2e} [1/yr]".format(sig_tot*T1yr),2)
    myprint("Ntot (residual numu) = {0:.2e} [1/yr]".format(dif_tot*T1yr),2)
    # pdf[sindec][logErec]
    pdf_sig, pdf_dif = sc.get_pdf(LZGamma)
    # signalness = sig_tot*pdf_sig/(sig_tot*pdf_sig+dif_tot*pdf_dif+atmos_tot*pdf_atmos)
    # signalness_fill = np.copy(signalness)
    # # Fill void cell by referring to 0<=sindec<=0.25. This affects very highE and upgoing (~90deg) region.
    # signalness_fill_void = np.nanmean(signalness[(0<pdf_bincsindec)&(pdf_bincsindec<0.25)],axis=0)
    # fill_cond = ((pdf_bincsindec2d>-0.14)&(pdf_binclog10E2d>6.5)&(np.isnan(signalness_fill)))
    # signalness_fill[fill_cond] = np.broadcast_to(signalness_fill_void[None,:],signalness_fill.shape)[fill_cond]

else:
    sc.set_signalflux_par(phi0_ref, gamma)
    sc.set_Eref(1e5) 
    sc_redshift.set_signalflux_par(phi0_ref, gamma)
    sc_redshift.set_Eref(1e5) 
    diff_tot = sc.get_diff_tot(gamma)
    myprint("Ntot (diffuse numu) = {0:.2e} [1/yr]".format(diff_tot*T1yr),2)
    # pdf[sindec][logErec]
    pdf_nu      = sc.get_pdf(gamma)
    signalness = diff_tot*pdf_nu/(diff_tot*pdf_nu+atmos_tot*pdf_atmos)
    signalness_fill = np.copy(signalness)
    signalness_fill[((pdf_bincsindec2d>-0.14)&(pdf_binclog10E2d>6.5)&(np.isnan(signalness_fill)))] = 1


# itp_signalness = RegularGridInterpolator((pdf_bincsindec, pdf_binclog10E), signalness_fill, method='linear', bounds_error=False, fill_value=None)
# nu_mysignalness = itp_signalness(np.stack([nu_sindec, nu_logenergy], axis=-1))



if(False):
    x = np.ravel(pdf_bincsindec2d)
    y = np.ravel(pdf_binclog10E2d)
    v = np.stack([x, y], axis=-1)
    #plt.scatter(x, y, c=itp_signalness(v))
    #plt.show()
    plt.figure(figsize=(7,5))
    plt.grid()
    plt.title("Signal model: {0}".format(name_sigmodel), fontsize=15)
    mapp = plt.pcolormesh(pdf_bincsindec, pdf_binclog10E, signalness_fill.T)
    cbar = plt.colorbar(mapp)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Signalness", fontsize=16)
    plt.xlabel("sin$\delta$", fontsize=18)
    plt.ylabel("$\mathrm{log}_{10}(E/\mathrm{GeV})$", fontsize=18)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.ylim(3,8)
    plt.xticks(np.linspace(-1,1,9))
    plt.tight_layout()
    plt.savefig("./figs/signalness_{0}.png".format(fname_sigmodel))
    plt.show()

if(False):
    plt.figure(figsize=(6,6))
    plt.title("signalness comparison")
    plt.grid()
    plt.plot(nu_signalness, nu_mysignalness, "o", color="red")
    plt.plot([0,1],[0,1], color="black")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tick_params(labelsize=15)
    plt.show()

if(use_nuXunifiledmodel):
    logEPDF_sig = (pdf_sig/np.sum(pdf_sig, axis=1)[:,None])/dpdf_binlog10E[None,:] #あるsindecが決まったときのPDF
    logEPDF_sig[logEPDF_sig==0] = np.min(logEPDF_sig[logEPDF_sig!=0])
    
    bg_tot  = dif_tot+atmos_tot
    pdf_bg = (dif_tot*pdf_dif+atmos_tot*pdf_atmos)/bg_tot

    logEPDF_bg = (pdf_bg/np.sum(pdf_bg, axis=1)[:,None])/dpdf_binlog10E[None,:]
    logEPDF_bg[logEPDF_bg==0] = np.min(logEPDF_bg[logEPDF_bg!=0])
    
    sindec_PDF_sig  = np.sum(pdf_sig*dpdf_binlog10E[None,:], axis=1)
    sindec_PDF_bg   = np.sum(pdf_bg*dpdf_binlog10E[None,:], axis=1)

    itp_sindecPDF_sig      = interpolate.interp1d(pdf_binsindec, np.append(sindec_PDF_sig, sindec_PDF_sig[-1]))
    itp_sindecPDF_bg       = interpolate.interp1d(pdf_binsindec, np.append(sindec_PDF_bg,  sindec_PDF_bg[-1]))
    
else:
    logEPDF_sig = (pdf_nu/np.sum(pdf_nu, axis=1)[:,None])/dpdf_binlog10E[None,:]
    logEPDF_sig[logEPDF_sig==0] = np.min(logEPDF_sig[logEPDF_sig!=0])

    logEPDF_bg = (pdf_atmos/np.sum(pdf_atmos, axis=1)[:,None])/dpdf_binlog10E[None,:]
    logEPDF_bg[logEPDF_bg==0] = np.min(logEPDF_bg[logEPDF_bg!=0])

    sindec_PDF_sig      = np.sum(pdf_nu*dpdf_binlog10E[None,:], axis=1)
    sindec_PDF_bg       = np.sum(pdf_atmos*dpdf_binlog10E[None,:], axis=1)

    itp_sindecPDF_sig      = interpolate.interp1d(pdf_binsindec, np.append(sindec_PDF_sig, sindec_PDF_sig[-1]))
    itp_sindecPDF_bg       = interpolate.interp1d(pdf_binsindec, np.append(sindec_PDF_bg,  sindec_PDF_bg[-1]))

    sig_tot = diff_tot
    bg_tot  = atmos_tot

def get_nuEPDF(sindec, log10E, isBG):
    # return signal neutrino energy PDF as sindec and log10E (dN/dlogE (sindec is parameter))
    dE  = (pdf_binlog10E.size-1)*(log10E-pdf_binlog10E[0])/(pdf_binlog10E[-1]-pdf_binlog10E[0])
    iE0   = np.int32(dE)
    iE0 = np.where((iE0<0), 0, iE0)
    iE0 = np.where(iE0>pdf_binlog10E.size-2, pdf_binlog10E.size-2, iE0)
    dE -= iE0
    iE1 = np.where(iE0<pdf_binlog10E.size-2, iE0+1, pdf_binlog10E.size-2)

    dsin= (pdf_binsindec.size-1)*(sindec-pdf_binsindec[0])/(pdf_binsindec[-1]-pdf_binsindec[0])
    isin0 = np.int32(dsin)
    isin0 = np.where(isin0<0, 0, isin0)
    isin0 = np.where(isin0>pdf_binsindec.size-2, pdf_binsindec.size-2, isin0)
    dsin -= isin0
    isin1 = np.where(isin0<pdf_binsindec.size-2, isin0+1, pdf_binsindec.size-2)

    if(isBG):
        aPDF = logEPDF_bg
    else:
        aPDF = logEPDF_sig
        
    pdf00 = aPDF[isin0, iE0]
    pdf01 = aPDF[isin0, iE1]
    pdf10 = aPDF[isin1, iE0]
    pdf11 = aPDF[isin1, iE1]
    
    pdf0 = pdf00*(1-dE) + dE*pdf01
    pdf1 = pdf10*(1-dE) + dE*pdf11
    
    return pdf0*(1-dsin) + pdf1*dsin

def get_nusindecPDF(sindec, isBG):
    # return neutrino sindec PDF (dN/dsindec (energy is summed up))
    flg = np.where((sindec<pdf_binsindec[0])|(sindec>pdf_binsindec[-1]), 1, 0)
    sindec = np.where(flg==1, 0.3, sindec)
    if(isBG):
        return np.where(flg==1, np.nan, itp_sindecPDF_bg(sindec))
    else:
        return np.where(flg==1, np.nan, itp_sindecPDF_sig(sindec))




def get_nuELLH(sindec, log10Enu_rec):
    log_EPDF_sig = np.log(get_nuEPDF(sindec, log10Enu_rec, isBG=False)) 
    log_EPDF_bg  = np.log(get_nuEPDF(sindec, log10Enu_rec, isBG=True))
    return log_EPDF_sig, log_EPDF_bg

def get_nunumLLH(sindec):
    log_nunumLLH_sig = np.log(get_nusindecPDF(sindec, False)*sig_tot)
    log_nunumLLH_bg  = np.log(get_nusindecPDF(sindec, True) *bg_tot)
    return log_nunumLLH_sig, log_nunumLLH_bg

def conv_pix2RADEC(x, y, RX, RY, DELX, DELY, RPX, RPY):

    X =  np.radians((x - RPX) * DELX)
    Y =  np.radians((y - RPY) * DELY)
    R = np.sqrt(X**2+Y**2)
    RX=np.radians(RX)
    RY=np.radians(RY)

    phi = np.arctan2(X, -Y)
    th  = np.arctan(1/R)

    DEC = np.sin(th)*np.sin(RY) - np.cos(th)*np.cos(phi)*np.cos(RY)
    DEC = np.degrees(np.arcsin(DEC))
    RA = np.arctan2(np.cos(th)*np.sin(phi), np.sin(th)*np.cos(RY)+np.cos(th)*np.cos(phi)*np.sin(RY))
    RA = np.degrees(RX + RA)
    RA = RA%360
    return RA, DEC

def e50_to_sigma(E50):
    # 2D円対称ガウスの r50 から sigma へ
    return E50 / np.sqrt(2*np.log(2))

def wrap_dra(ra, ra0):
    """ΔRA を [-pi, pi) に wrap（ラジアン）"""
    dra = (ra - ra0 + np.pi) % (2*np.pi) - np.pi
    return dra

def pdf_neutrino_radec(ra_deg, dec_deg, ra0_deg, dec0_deg, E50_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0  = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    sigma = np.deg2rad(e50_to_sigma(E50_deg))  # あなたの定義に従う

    dra  = wrap_dra(ra, ra0)
    ddec = dec - dec0

    cos_dec0 = np.cos(dec0)
    cos_dec0 = np.clip(cos_dec0, 1e-12, None)

    q = (dra*cos_dec0)**2 + (ddec)**2
    # norm = cos_dec0 / (2*np.pi*sigma**2)
    norm = 1 / (2*np.pi*sigma**2)

    return norm * np.exp(-0.5*q/(sigma**2))


def get_nuposiLLH(target, x, y, ra0, dec0, E_50, A_R, Z_S, B_const):
    realFU_hdul = fits.open(f'{target}')
    realFU_RA_PNT = realFU_hdul['EVENTS'].header["RA_PNT"]
    realFU_DEC_PNT = realFU_hdul['EVENTS'].header["DEC_PNT"]
    realFU_TCRPX2 = realFU_hdul['EVENTS'].header["TCRPX2"]  
    realFU_TCDLT2 = realFU_hdul['EVENTS'].header["TCDLT2"]    
    realFU_TCRPX3 = realFU_hdul['EVENTS'].header["TCRPX3"]    
    realFU_TCDLT3 = realFU_hdul['EVENTS'].header["TCDLT3"]
    # ra_pnt = np.radians(eventfile['EVENTS'].header['RA_PNT'])
    # dec_pnt = np.radians(eventfile['EVENTS'].header['DEC_PNT'])
    # TCRPX2 = eventfile['EVENTS'].header["TCRPX2"]  
    # TCDLT2 = eventfile['EVENTS'].header["TCDLT2"]    
    # TCRPX3 = eventfile['EVENTS'].header["TCRPX3"]    
    # TCDLT3 = eventfile['EVENTS'].header["TCDLT3"]
    deg_circle=300*np.abs(realFU_TCDLT2)
    RA, DEC = conv_pix2RADEC(x, y, realFU_RA_PNT, realFU_DEC_PNT, realFU_TCDLT2, realFU_TCDLT3, realFU_TCRPX2, realFU_TCRPX3)
    # print(ra0, dec0, RA, DEC, target.stem)
    log_nuposi_pdf_sig = np.log(pdf_neutrino_radec(RA, DEC, ra0, dec0, E_50)) ## norm in allsky
    log_nuposi_pdf_bg = np.log(1/(2*np.pi*(1 - np.cos(np.radians(deg_circle))))) ## norm in the circle

    return log_nuposi_pdf_sig, log_nuposi_pdf_bg

    
#################################################################################


base_dir = Path("/home/yu/XRT/MultipleTiling") 
csv_path = Path("/home/yu/XRT/MultipleTiling/full_with_icname_swifttime_obsstart.csv")# This file contains real IceCube alert information position, error, energy, etc


ic_dirs = [] 


for p in sorted(base_dir.glob("IC*")):
    if not p.is_dir():
        continue
    
    icname = p.name
    ic_dirs.append((icname))

df = pd.read_csv(csv_path)

# alerts used in the analysis (TeV unit, northern sky, and BRONZE and GOLD)
mask = df['Energy'].notna()
mask2 = df['Dec'] > -5
mask3 = (df['NoticeType']=='BRONZE') |(df['NoticeType']=='GOLD')
mask_all = (mask)&(mask2)&(mask3)
# df_hit = df[(mask)&(mask2)].copy()
df_hit = df[mask_all].copy()


ic_info = []       

# IC_NAME -> df 
ic_to_row = {row["ICname"]: row for _, row in df_hit.iterrows()}

for icname in ic_dirs:
    row = ic_to_row.get(icname)
    if row is None:
        continue

    # DEC [deg] -> sindec
    ra = row['RA'] #deg
    dec = row['Dec'] #deg
    E_50 = row['Error50']/60 #arcmin to deg
    sindec = np.sin(np.deg2rad(row["Dec"]))
    energy_tev= row["Energy"]
    energy_gev = energy_tev*10**3

    ic_info.append(
        {
            "icname": icname,
            "ra": ra,
            "dec": dec,
            "E_50": E_50,
            "sindec": float(sindec),
            "energy_gev": float(energy_gev),
        }
    )

print("usable groups:", [info["icname"] for info in ic_info])


from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


# -------------------------
# calculate likelihood components (without beta)
# -------------------------
def process_evt_components(evt_file, target, sindec_src, log10Enu_rec_sig,
                           ra_0, dec_0, E_50, A_R, Z_S, B_const, PF_log):

    log_nuEPDF_LLH_sig, log_nuEPDF_LLH_bg = get_nuELLH(sindec_src, log10Enu_rec_sig)
    log_nunumLLH_sig,   log_nunumLLH_bg   = get_nunumLLH(sindec_src)

    log_X_LLH_sig, log_X_LLH_bg, fitx, fity = get_XLLH(evt_file)

    log_nuposiLLH_sig, log_nuposiLLH_bg = get_nuposiLLH(
        target, fitx, fity, ra_0, dec_0, E_50, A_R, Z_S, B_const
    )

    # s1: Xがsigのときの「sig項」
    s1 = log_nuEPDF_LLH_sig + log_nunumLLH_sig + log_X_LLH_sig + PF_log + log_nuposiLLH_sig
    # s0: Xがbgのときの「sig項」
    s0 = log_nuEPDF_LLH_sig + log_nunumLLH_sig + log_X_LLH_bg          + log_nuposiLLH_sig
    # b : 完全bg
    b  = log_nuEPDF_LLH_bg  + log_nunumLLH_bg  + log_X_LLH_bg          + log_nuposiLLH_bg

    return float(s1), float(s0), float(b)


# -------------------------
# calculate TS with beta
# -------------------------
def TS_of_beta(beta, s1_arr, s0_arr, b_arr, eps=1e-12):
    beta = float(np.clip(beta, eps, 1.0 - eps))

    # log L_sig(beta) = log( beta e^{s1} + (1-beta) e^{s0} + e^{b} )
    logL_sig = logsumexp(
        np.vstack([
            s1_arr + np.log(beta),
            s0_arr + np.log1p(-beta),   # log(1-beta)を安定に
            b_arr
        ]),
        axis=0
    )

    # log L_bg = log( e^{s0} + e^{b} )
    logL_bg = logsumexp(np.vstack([s0_arr, b_arr]), axis=0)

    return 2.0 * (np.sum(logL_sig) - np.sum(logL_bg))

# -------------------------
# fit beta by maximizing TS
# -------------------------
def maximize_TS_beta(s1_arr, s0_arr, b_arr):
  
    f = lambda beta: -TS_of_beta(beta, s1_arr, s0_arr, b_arr)

    try:
        res = minimize_scalar(f, bounds=(1e-6, 1.0 - 1e-6), method="bounded")
        beta_hat = float(res.x)
        TS_hat   = float(-res.fun)
        return beta_hat, TS_hat
    except Exception:
        print('ERR: scipy dead')



# -------------------------
# evt_dir -> {stem: (s1,s0,b)}
# -------------------------
def build_comp_map_for_dir(evt_dir, pathsP, targets,
                           sindec, log10Enu_rec_sig, ra_0, dec_0, E_50, A_R, Z_S, B_const, PF_log):
    evt_files = sorted(Path(evt_dir).glob("*.evt"))
    if len(evt_files) == 0:
        return {}

    comp_map = {}
    for evt_file in evt_files:
        evt_file = Path(evt_file)

        idx = next((k for k, p in enumerate(pathsP) if p.stem == evt_file.stem), None)
        if idx is None:
            continue

        target = targets[idx]
        s1, s0, b = process_evt_components(
            evt_file, target, sindec, log10Enu_rec_sig, ra_0, dec_0, E_50, A_R, Z_S, B_const, PF_log
        )
        comp_map[evt_file.stem] = (s1, s0, b)

    return comp_map


def run_one_trial(t, PF_log, rho):
    try:
        if rho < 2e-9:
            lumis = ['BG', '1e44', '5e44', '1e45', '5e45', '6e45', '7e45', '8e45', '9e45', '1e46', '2e46', '3e46', '4e46', '5e46', '1e47']
        else:
            lumis = ['BG', '1e44', '5e44', '1e45', '2e45', '3e45', '4e45', '5e45', '6e45', '7e45', '8e45', '9e45', '1e46', '1e47']
        result_by_lumi = {lumi: {"beta_hat": None, "TS_hat": None, "n_evt": 0} for lumi in lumis}

        # ---- make cache----
        prep = {}         # icname -> dict(pathsP=..., targets=..., params=...)
        bg_map_cache = {} # icname -> bg_map or None

        for info in ic_info:
            icname  = info["icname"]
            sindec  = info["sindec"]
            log10Enu_rec_sig = info["energy_gev"]
            ra_0    = info["ra"]
            dec_0   = info["dec"]
            E_50    = info["E_50"]

            # 1) pathsP / targets
            paths = np.load(f'{base_dir}/{icname}/combined/combined_{t}/combined_{t}.npy', allow_pickle=True)
            pathsP = [Path(p) for p in paths]

            exist_npy_name = 'exist_evt_8.6e+04_200.npy'
            npy_path = base_dir / f'{icname}' / exist_npy_name
            existlist = np.load(npy_path, allow_pickle=True)
            targets = np.array(sorted({Path(e) for e in existlist}))

            prep[icname] = dict(
                pathsP=pathsP, targets=targets,
                sindec=sindec, log10Enu_rec_sig=log10Enu_rec_sig,
                ra_0=ra_0, dec_0=dec_0, E_50=E_50
            )

            # 2) BG map（not depends on lumi)
            bg_base = Path(f'{base_dir}/{icname}/combined/combined_{t}/BG/2rxs_removed')
            if not bg_base.exists():
                bg_map_cache[icname] = None
                continue

            A_R, Z_S, B_const = 1, 1, 1 # this is no longer used 
            bg_map = build_comp_map_for_dir(
                bg_base, pathsP, targets,
                sindec, log10Enu_rec_sig, ra_0, dec_0, E_50,
                A_R, Z_S, B_const, PF_log
            )
            bg_map_cache[icname] = bg_map if len(bg_map) else None

    
        for lumi in lumis:
            s1_all, s0_all, b_all = [], [], []

            for info in ic_info:
                s1_ic, s0_ic, b_ic = [], [], []
                icname = info["icname"]

                bg_map = bg_map_cache.get(icname)
                if bg_map is None:
                    continue

                # extract from the cache
                P = prep[icname]
                pathsP  = P["pathsP"]
                targets = P["targets"]
                sindec  = P["sindec"]
                log10Enu_rec_sig = P["log10Enu_rec_sig"]
                ra_0    = P["ra_0"]
                dec_0   = P["dec_0"]
                E_50    = P["E_50"]


                
                if lumi == "BG":
                    use_map = bg_map
                else:
                    inj_base = Path(f'{base_dir}/{icname}/combined/combined_{t}/unified_gamma{LZGamma_ref:.0f}_modify_rng/{lumi}_{rho:.1e}/2rxs_removed')

                    # no inject → all BG
                    if (not inj_base.exists()):
                        use_map = bg_map
                    else:
                        inj_map = build_comp_map_for_dir(inj_base, pathsP, targets,
                                                         sindec, log10Enu_rec_sig, ra_0, dec_0, E_50, A_R, Z_S, B_const, PF_log)
                        if len(inj_map) == 0:
                            use_map = bg_map
                        else:
                            # BG base, convert tiling only when inject file exist 
                            use_map = dict(bg_map)
                            use_map.update(inj_map)

                # add s1, s0, b for each tile to s1_all, s0_all, b_all
                for (s1, s0, b) in use_map.values():
                    s1_all.append(s1)
                    s0_all.append(s0)
                    b_all.append(b)
                    s1_ic.append(s1)
                    s0_ic.append(s0)
                    b_ic.append(b)

                #save s1, s0, b for each IC events to use  
                fraction_dir = Path(f'{base_dir}/{icname}/combined/combined_{t}/unified_gamma{LZGamma_ref:.0f}_modify_rng/{lumi}_{rho:.1e}/fraction/')
                fraction_dir.mkdir(parents=True, exist_ok=True)
                np.save(f'{fraction_dir}/s1list.npy', np.asarray(s1_ic, dtype=float))
                np.save(f'{fraction_dir}/s0list.npy', np.asarray(s0_ic, dtype=float))
                np.save(f'{fraction_dir}/blist.npy', np.asarray(b_ic, dtype=float))
            # このlumiで 1個もevtが無い場合
            if len(s1_all) == 0:
                result_by_lumi[lumi] = {"beta_hat": None, "TS_hat": None, "n_evt": 0}
                continue

            s1_arr = np.asarray(s1_all, dtype=float)
            s0_arr = np.asarray(s0_all, dtype=float)
            b_arr  = np.asarray(b_all,  dtype=float)

            beta_hat, TS_hat = maximize_TS_beta(s1_arr, s0_arr, b_arr)
            result_by_lumi[lumi] = {"beta_hat": beta_hat, "TS_hat": TS_hat, "n_evt": int(len(s1_arr))}

        # save TS (sum of TS for all tiligs of all IC alerts)
        out = Path(base_dir) / 'TS_unified_modify_rng'/ f"TS_gamma{LZGamma_ref:.0f}_rho{rho:.1e}_PF{PF_log}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"TS_trial{t:04d}.npy", result_by_lumi)

    except Exception as e:
        print(f"[ERROR] trial {t}: {e}")

    return {"trial_index": t}


if __name__ == "__main__":
    t = int(sys.argv[1])
    PF_log = -float(sys.argv[2])
    rho = float(sys.argv[3])
    run_one_trial(t, PF_log, rho)



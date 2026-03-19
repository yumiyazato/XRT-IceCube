# This is supposed to be run not by condor but by Multiprocessing on inject.ipynb 
# Bulk Lorentz factor is valiable and set in this code
# rho is valiable and set in inject.ipynb 
# Lx is valiable and set in this code depending on rho (for smaller rho, bin step of larger Lx is relatively small)

from astropy.io import fits
from scipy.integrate import quad
from astropy.table import Table
from scipy.integrate import dblquad
from pathlib import Path
from astropy.table import Table
import pandas as pd
import sys
import numpy as np
from scipy import interpolate
import scipy.stats as st
import os
from util.calc_diffuse_flux import GetDiffuseFlux
from signal_calculation import signal_calculation
from NHabs import NHabs
from scipy.interpolate import RegularGridInterpolator
import shutil  
alertlevel = 0 # As PDF, use GFU level
use_nuXunifiledmodel = True
if(use_nuXunifiledmodel):
    LZGamma = 7 # Bulk Lorentz factor  # changeable
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
    signalness = sig_tot*pdf_sig/(sig_tot*pdf_sig+dif_tot*pdf_dif+atmos_tot*pdf_atmos)
    signalness_fill = np.copy(signalness)
    # Fill void cell by referring to 0<=sindec<=0.25. This affects very highE and upgoing (~90deg) region.
    signalness_fill_void = np.nanmean(signalness[(0<pdf_bincsindec)&(pdf_bincsindec<0.25)],axis=0)
    fill_cond = ((pdf_bincsindec2d>-0.14)&(pdf_binclog10E2d>6.5)&(np.isnan(signalness_fill)))
    signalness_fill[fill_cond] = np.broadcast_to(signalness_fill_void[None,:],signalness_fill.shape)[fill_cond]

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


itp_signalness = RegularGridInterpolator((pdf_bincsindec, pdf_binclog10E), signalness_fill, method='linear', bounds_error=False, fill_value=None)
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

# ------- singlet PDF calculation --------#
def get_singlet_PDF_generator(rho):
    bin_sd = np.linspace(-np.sin(np.radians(5)), 1, 41)
    binc_sd = 0.5*(bin_sd[1:]+bin_sd[:-1])
    dbin_sd = bin_sd[1:]-bin_sd[:-1]
    binc_sd2d, zc_2d = np.meshgrid(binc_sd, zc, indexing="ij")
    dbin_sd2d, dz_2d = np.meshgrid(dbin_sd, dz, indexing="ij")
    binc_sd2d = np.ravel(binc_sd2d)
    zc_2d = np.ravel(zc_2d)
    dbin_sd2d = np.ravel(dbin_sd2d)
    dz_2d = np.ravel(dz_2d)
    dist_2d = dsc.getSingleSourceDistance(zc_2d)
    dec_2d = np.arcsin(binc_sd2d)
    sc_redshift.nuXmodel.set_Gamma(LZGamma)
    sc_redshift.nuXmodel.maximize_CRloadingfactor(1, 0)
    delT = 2*Tw_inday*T1d/T1yr #10^4.5sは一年換算するとどれくらいかの指標

    n0 = rho*delT
    # Lnu = sc_redshift.get_nuXmodel_allflavor_emission_luminosity(LZGamma, log10Emin=4, log10Emax=6)/n0/erg #関数のところはn0=1のとき。n0を変えるとdiffuseを説明するのに必要なLも変わるから
    # myprint("Per source all flavor neutrino emission energy = {0:.2e} erg".format(Lnu*delT*T1yr), 2)
    mu = sc_redshift.get_mu(dec_2d, zc_2d)/4/np.pi/dist_2d**2/Mpc/Mpc/n0*delT*T1yr#関数のところはn0=1のとき。n0を変えるとdiffuseを説明するのに必要なLも変わるから
    poisson_term = mu*np.exp(-mu)
    # [sindec_c][zc] (number/z/sr/yr)
    dNdz = c*rho/H0/Mpc*get_SFR(zc_2d)*dist_2d**2/dsc.F(zc_2d)*poisson_term
    dNdz = np.reshape(dNdz, (binc_sd.size, zc.size))
    dNdz = np.sum(dNdz, axis=0)
    dNdz /= np.sum(dNdz*dz)

    generator_redshift = interpolate.interp1d(np.cumsum(np.append(0, dNdz*dz)), np.append(0, z[1:]))
    return generator_redshift

############################ summarize alert information which are used in analysis ##############

base_dir = Path("/home/yu/XRT/MultipleTiling") 
csv_path = Path("/home/yu/XRT/MultipleTiling/full_with_icname_swifttime_obsstart.csv") # This file contains real IceCube alert information position, error, energy, etc


ic_dirs = []  # IC event name list

for p in sorted(base_dir.glob("IC*")):
    if not p.is_dir():
        continue
    
    icname = p.name
    ic_dirs.append((icname))
# read csv
df = pd.read_csv(csv_path)

# select alerts with energy in TeV unit
mask = df['Energy'].notna()
# mask2 = np.sin(np.deg2rad(df['DEC'])) > pdf_binsindec[0]
# df_hit = df[(mask)&(mask2)].copy()
df_hit = df[mask].copy()



# IC_NAME -> df の行 への辞書
ic_to_row = {row["ICname"]: row for _, row in df_hit.iterrows()}

# summarize usable alerts information
ic_info = []       
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
    log10E=np.log10(energy_gev)

    ic_info.append(
        {
            "icname": icname,
            "ra": ra,
            "dec": dec,
            "E_50": E_50,
            "sindec": float(sindec),
            "energy_gev": float(energy_gev),
            "signalness": itp_signalness([sindec, log10E])
        }
    )

##########################################################################################


#################### X-ray related parameters ###############################

Ex_bol_min = 0.1
bound=0.5
a1=1
a2=2.2
Ex_bol_max = 10000


normX = erg/(bound*keV)**2/((1-(Ex_bol_min/bound)**(2-a1))/(2-a1) + ((Ex_bol_max/bound)**(2-a2)-1)/(2-a2))*keV

def get_normalized_flux(Ex):
    # 1keVから10MeVで1erg/sになるよう規格化
    return np.where(Ex<bound, normX*(Ex/bound)**(-a1), normX*(Ex/bound)**(-a2))


Ex_xrt_min = 0.2
Ex_xrt_max = 10

Ex_pi_min = 0.4 # since we make a PI threshold 40 (0.4 keV), the number of injection is evaluated by considering 0.4 to 10 keV  
Ex_pi_max = 10  # In short, we initially inject pseudo photons with > 0.4 keV but not assign PI for each photon and remove small PI photons later (because PI pdf file is too large) 
 
e_kev = np.load('/home/yu/XRT/e_kev.npy') #0.1-12 keV, 5eV step

#### effective area countrate #######
fm_ft_fq = np.load('/home/yu/XRT/fm_ft_fq.npy') # the product of mirror, transmission, and qe for each energy → Effective area 
vighdu = fits.open('/home/yu/XRT/caldb/data/swift/xrt/cpf/vign/swxvign20010101v001.fits') # vignetting information
mask_pi = (Ex_pi_min<= e_kev) & (e_kev <= Ex_pi_max) 
e_kev_mask_pi = e_kev[mask_pi]
fm_ft_fq_mask_pi = fm_ft_fq[mask_pi]

p0, p1, p2 = vighdu['VIG_COEF'].data[0]

nhabs = NHabs()
NH=2e20

###########################################################


################# calculate countrate considering effective area, vignetting, and flux ################## 
def countrate(offaxis, Lx, z):
    coef_E = p0 * p1**e_kev_mask_pi + p2
    offaxis2 = offaxis**2
    vignetting_E = 1 - coef_E * offaxis2  # shape: (ekev, 601, 601)
    vignetting_E = np.where(vignetting_E < 0, 0, vignetting_E)

    ## INTEGRATION for each pixel table ###########################
    aD = dsc.getSingleSourceDistance(z)
    spec = (Lx*get_normalized_flux(e_kev_mask_pi*(1+z))/4/np.pi/(aD*Mpc)**2)*nhabs.get_absorption(NH, e_kev_mask_pi*(1+z))

    func = vignetting_E*spec*fm_ft_fq_mask_pi
    dE = np.gradient(e_kev_mask_pi)
    crate_per_Lxandz = np.sum(func * dE, axis=0)
    return crate_per_Lxandz

##### spread probability using PSF#######################
limit = 30 # injection area

hdul_006 = fits.open('/home/yu/XRT/caldb/data/swift/xrt/cpf/psf/swxpsf20010101v006.fits') # PSF
PC_006 = hdul_006['PC_PSF_COEF'].data
parameters_6 = PC_006['COEF0']
def psf(r):
    p0 = parameters_6[0]
    p1 = parameters_6[1]
    p2 = parameters_6[2]
    p3 = parameters_6[3]
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
circleregion, _ = quad(psf_integral, 0, limit)
psfratio=circleregion/total
# print(psfratio)

#### spread probability (PSF does not depend on off axis so we prepare common probability table of 30*30 pix area)############
x = np.linspace(-limit, limit, 2*limit+1)
y = np.linspace(-limit, limit, 2*limit+1)
X, Y = np.meshgrid(x, y)
R = np.hypot(X, Y)
psf_vals = np.zeros_like(R)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        r = R[i, j]
        x0 = X[i, j]
        y0 = Y[i, j]
        if r < 10:
            val, _ = dblquad(
                psf_xy,
                x0 - 0.5, x0 + 0.5,
                lambda x: y0 - 0.5,
                lambda x: y0 + 0.5
            )
        else:
            val = psf(r)
        psf_vals[i, j] = val
# print(psf_vals.shape)
probability = np.where(R <= limit, psf_vals, np.nan)
nan_mask=np.isnan(probability)
prob_map_flat = np.nan_to_num(probability).flatten()
norm_prob_map_flat=prob_map_flat/np.sum(prob_map_flat)


################## determine injection position considering reported error of each alert ###########
def e50_to_sigma(E50):
    # 2D円対称ガウスの r50 から sigma へ
    return E50 / np.sqrt(2*np.log(2))

def sample_neutrino_radec(ra0_deg, dec0_deg, E50_deg, n=1):

    rng = np.random.default_rng()


    # 角度単位は一旦ラジアンで扱う
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    sigma_deg = e50_to_sigma(E50_deg)
    sigma = np.deg2rad(sigma_deg)

    # r は Rayleigh
    r = rng.rayleigh(scale=sigma, size=n)

    # 方向角は一様
    theta = rng.uniform(0, 2*np.pi, size=n)

    # 接平面でのオフセット
    d_dec = r * np.sin(theta)
    # RA方向は cos(dec) で縮むので補正
    cos_dec0 = np.cos(dec0)
    # 極付近の暴走回避
    cos_dec0 = np.clip(cos_dec0, 1e-12, None)
    d_ra = (r * np.cos(theta)) / cos_dec0

    ra = ra0 + d_ra
    dec = dec0 + d_dec

    # RA を 0..2π に折り返し
    ra = (ra + 2*np.pi) % (2*np.pi)

    return np.rad2deg(ra), np.rad2deg(dec)

def conv_RADEC2local(ra, dec, ra_pnt, dec_pnt, DELX, DELY, RPX, RPY):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    ra_pnt_rad = np.radians(ra_pnt)
    dec_pnt_rad = np.radians(dec_pnt)
    ca = np.cos(ra_pnt_rad)
    sa = np.sin(ra_pnt_rad)
    cd = np.cos(dec_pnt_rad)
    sd = np.sin(dec_pnt_rad)
    x = np.cos(dec_rad)*np.cos(ra_rad)
    y = np.cos(dec_rad)*np.sin(ra_rad)
    z = np.sin(dec_rad)
    x_ =    -sa*x +    ca*y
    y_ = -ca*sd*x - sa*sd*y + cd*z
    z_ =  ca*cd*x + sa*cd*y + sd*z
    x = RPX + np.degrees(x_) / DELX
    y = RPY + np.degrees(y_) / DELY
    return x, y

### BAD PIX information conversion (RAW to SKY) ########################################
def extended (path):
    original = fits.open(path)
    BADPIX = original['BADPIX'].data
    # YEXTENT open
    bad_rawx, bad_rawy = BADPIX['RAWX'], BADPIX['RAWY']
    type = BADPIX['TYPE']
    yextent = BADPIX['YEXTENT']
    bad_rawx_1 = bad_rawx[type==1]
    bad_rawx_2 = bad_rawx[type==2]
    bad_rawx_3 = bad_rawx[type==3]
    bad_rawy_1 = bad_rawy[type==1]
    bad_rawy_2 = bad_rawy[type==2]
    bad_rawy_3 = bad_rawy[type==3]
    yextent_2 = yextent[type==2]
    yextent_3 = yextent[type==3]
    xlist_2 = []
    ylist_2 = []
    for x2, y2, yex2 in zip(bad_rawx_2, bad_rawy_2, yextent_2):
        xlist_2.extend([x2]*yex2)
        ylist_2.extend([y2+i for i in range(yex2)])
    xarray_2 = np.array(xlist_2)
    yarray_2 = np.array(ylist_2)
    xlist_3 = []
    ylist_3 = []
    for x3, y3, yex3 in zip(bad_rawx_3, bad_rawy_3, yextent_3):
        ylist_3.extend([y3]*yex3)
        xlist_3.extend([x3+i for i in range(yex3)])
    xarray_3 = np.array(xlist_3)
    yarray_3 = np.array(ylist_3)
    extended_bad_rawx = np.concatenate([bad_rawx_1, xarray_2, xarray_3])
    extended_bad_rawy = np.concatenate([bad_rawy_1, yarray_2, yarray_3])
    return extended_bad_rawx, extended_bad_rawy

# BAD RAW TO BAD SKY 
def BADRAWtoBADSKY(skyx, skyy, rawx, rawy, badrawx, badrawy, start, stop, time):
    skyx_ex = skyx[(time>start) & (time<=stop)]
    skyy_ex = skyy[(time>start) & (time<=stop)]
    rawx_ex = rawx[(time>start) & (time<=stop)]
    rawy_ex = rawy[(time>start) & (time<=stop)]
    raws = np.vstack([rawx_ex, rawy_ex, np.ones_like(rawx_ex)]).T
    a, b, offsetx = np.linalg.lstsq(raws, skyx_ex, rcond=None)[0]
    c, d, offsety = np.linalg.lstsq(raws, skyy_ex, rcond=None)[0]
    badskyx = a*badrawx + b*badrawy + offsetx
    badskyy = c*badrawx + d*badrawy + offsety
    return badskyx, badskyy

### EXPAND BADRAW (3*3 region around bad pixel is regarde as bad)
def expand(badskyx, badskyy):
    badskyx = np.rint(badskyx).astype(int)
    badskyy = np.rint(badskyy).astype(int)

    # relative position 3×3
    dx = [-1, 0, 1]
    dy = [-1, 0, 1]


    xlist, ylist = [], []
    for x, y in zip(badskyx, badskyy):
        for ddx in dx:
            for ddy in dy:
                xlist.append(x + ddx)
                ylist.append(y + ddy)

    # 配列に変換
    badx_expanded = np.array(xlist, dtype=int)
    bady_expanded = np.array(ylist, dtype=int)
    # 重複を除去
    pairs = np.unique(np.column_stack([badx_expanded, bady_expanded]), axis=0)
    badx_expanded, bady_expanded = pairs[:,0], pairs[:,1]
    badset_expanded = set(zip(badx_expanded, bady_expanded))
    return badset_expanded
###########################################################################################


##### FOV (the center of fov is recoreded in det coordinates, so we convert it to sky coordinate. 307 and 296 are officially reported#####
def DETtoSKY(skyx, skyy, detx, dety, start, stop, time, fov_center_detx=307.0, fov_center_dety=296.0):
    skyx_ex = skyx[(time>start) & (time<=stop)]
    skyy_ex = skyy[(time>start) & (time<=stop)]
    detx_ex = detx[(time>start) & (time<=stop)]
    dety_ex = dety[(time>start) & (time<=stop)]
    dets = np.vstack([detx_ex, dety_ex, np.ones_like(detx_ex)]).T
    a, b, offsetx = np.linalg.lstsq(dets, skyx_ex, rcond=None)[0]
    c, d, offsety = np.linalg.lstsq(dets, skyy_ex, rcond=None)[0]
    fov_center_skyx = a*fov_center_detx + b*fov_center_dety + offsetx
    fov_center_skyy = c*fov_center_detx + d*fov_center_dety + offsety
    return fov_center_skyx, fov_center_skyy


################# process 1 file #############################

def process_evt_file(icdir, i, rho) -> dict:
    if rho < 2e-9:
        Lxlist = np.array([1e44, 5e44, 1e45, 5e45, 6e45, 7e45, 8e45, 9e45, 1e46, 2e46, 3e46, 4e46, 5e46, 1e47])
    else: 
        Lxlist = np.array([1e44, 5e44, 1e45, 2e45, 3e45, 4e45, 5e45, 6e45, 7e45, 8e45, 9e45, 1e46, 1e47])         
    result = {
        "icdir": str(icdir),
        "idx": int(i),
        "rho": float(rho),
        "status": "ok",
        "inject": None, 
        "z": None,
        "finalnumber": None,
        "note": None,
    }
    rng = np.random.default_rng()  #
    exist_npy_name = 'exist_evt_8.6e+04_200.npy'

    # if input IC name is not on ic_info (usable IC alerts), no result
    IC = Path(icdir).name 
    row = next((d for d in ic_info if d["icname"] == IC), None)
    if row is None:
        result["status"] = "no_ic_info"
        result["note"] = "no_ic_info"
        return result

    RA_0 = row['ra']
    DEC_0 = row['dec']
    E50_deg = row['E_50']
    signalness = row['signalness'].item()
    npy_path = icdir / exist_npy_name
    existlist = np.load(npy_path, allow_pickle=True)
    targets = np.array(sorted({Path(e) for e in existlist}))

    # if no tiling observation that pass (>200s, within 1day) , no result
    if targets.size == 0:
        result["status"] = "no_targets"
        result["note"] = f"existlist empty: {exist_npy_name}"
        return result


    outdir = Path(f'{icdir}/combined/combined_{i}')
    paths = np.load(f'{icdir}/combined/combined_{i}/combined_{i}.npy', allow_pickle=True)

    inject_list = []
    final_number_list = []
    zi_list = []
    ListOfFlux = []
    ListOfSource = []
    written = []
    radec_nu = []
    # injection for each Lx
    for luminosity in Lxlist:
        ra_source, dec_source = sample_neutrino_radec(RA_0, DEC_0, E50_deg, n=1)
        ra_source_scal = ra_source.item()
        dec_source_scal = dec_source.item()
        radec_nu.append((ra_source_scal, dec_source_scal))
        generator_redshift = get_singlet_PDF_generator(rho)
        # zi=generator_redshift(np.random.rand())
        zi=generator_redshift(rng.random())


        lumi_str = f"{luminosity:.0e}".replace("+", "")
        out_path = Path(f'{icdir}/combined/combined_{i}/unified_gamma{LZGamma:.0f}_modify_rng/{lumi_str}_{rho:.1e}/')
        
        out_path.mkdir(parents=True, exist_ok=False)  # ここは必ず新規作成にする）
        # print('correct')
        inject_list_small = []
        final_number_list_small = []
        zi_list_small = []

        # judge whether conuterpart exist or not based on signalness

        out = int(rng.random()<signalness)
        if out==0:
            no_counterpart = Path(f'{out_path}/No-Counterpart/')
            no_counterpart.mkdir(parents=True, exist_ok=True)
            # ListOfSource.append(ListOfSource_small)
            inject_list.append(inject_list_small)
            final_number_list.append(final_number_list_small)
            # total_list.append(total_list_small)
            # ListOfFlux.append(ListOfFlux_small)
            zi_list.append(zi_list_small)
            # written.append(written_small)

            continue


        for j, path in enumerate(paths):
            base = Path(path).stem
            try:
                target = targets[j]
                realFU_hdul = fits.open(f'{target}')
                realFU_RA_PNT = realFU_hdul['EVENTS'].header["RA_PNT"]
                realFU_DEC_PNT = realFU_hdul['EVENTS'].header["DEC_PNT"]
                realFU_TCRPX2 = realFU_hdul['EVENTS'].header["TCRPX2"]  
                realFU_TCDLT2 = realFU_hdul['EVENTS'].header["TCDLT2"]    
                realFU_TCRPX3 = realFU_hdul['EVENTS'].header["TCRPX3"]    
                realFU_TCDLT3 = realFU_hdul['EVENTS'].header["TCDLT3"]
                p = Path(path)
            
                with fits.open(p, mode="readonly", memmap=True) as original:

                    EVENTS = original["EVENTS"].data
                    skyx = EVENTS["X"]; skyy = EVENTS["Y"]
                    rawx = EVENTS["RAWX"]; rawy = EVENTS["RAWY"]
                    detx = EVENTS["DETX"]; dety = EVENTS["DETY"]
                    GTI = original["GTI"].data
                    start = GTI["START"][0]; stop = GTI["STOP"][0]
                    time = EVENTS['TIME']

                    # BADPIX
                    if "BADPIX" in original:
                        BADPIX = original["BADPIX"].data
                        badrawx = BADPIX["RAWX"]; badrawy = BADPIX["RAWY"]
                    else:
                        badrawx = np.array([], dtype=int)
                        badrawy = np.array([], dtype=int)

                    # Extended BAD
                    ext_badx, ext_bady = extended(path)
                    badrawx = ext_badx
                    badrawy = ext_bady

                    # RAW→SKY mapping
                    badskyx, badskyy = BADRAWtoBADSKY(skyx, skyy, rawx, rawy, badrawx, badrawy, start, stop, time)
                    badset_expanded = expand(badskyx, badskyy)  # set((x,y)) を期待

                    fov_center_skyx, fov_center_skyy = DETtoSKY(skyx, skyy, detx, dety, start, stop, time, fov_center_detx=307.0, fov_center_dety=296.0)
                  
                    
                    x_source, y_source = conv_RADEC2local(ra_source, dec_source, realFU_RA_PNT, realFU_DEC_PNT, realFU_TCDLT2, realFU_TCDLT3, realFU_TCRPX2, realFU_TCRPX3)
                    
                    x_source = round(x_source.item())
                    y_source = round(y_source.item())

                    # skip if the position is out of FOV
                    # FOV in sky coordinate is from 200 to 800 for both X and Y 
                    # considering PSF, if the source position is out of 150 to 850, we skip injection.
                    iy = y_source - 150
                    ix = x_source - 150 
                    if not (0 <= iy <= 700 and 0 <= ix <= 700):
                        inject_list_small.append(0)
                        final_number_list_small.append(0)
                        zi_list_small.append(zi)
                        continue
                    
                    # once source position is determined, we calculate expected number of photons from the source
                    offaxis=np.sqrt((500.5-ix)**2+(500.5-iy)**2)*(2.36/60)
                    crate = countrate(offaxis, luminosity, zi)
                    exposure_time = 200.0  # s
                    total_counts = (crate * exposure_time) * psfratio # since we limit the injection area, we need psf ration
                    inject_counts = rng.poisson(lam=total_counts)
                    if inject_counts==0:
                        inject_list_small.append(int(inject_counts))
                        final_number_list_small.append(0)
                        zi_list_small.append(zi)
                        continue
                    # 多項分布でピクセルに振り分け
                    pixel_counts_flat = rng.multinomial(inject_counts, norm_prob_map_flat)

                    pixel_counts = pixel_counts_flat.reshape(psf_vals.shape).astype(float)

                    pixel_counts[nan_mask] = np.nan
    
                    # extract not Nan not 0
                    mask = (~np.isnan(pixel_counts)) & (pixel_counts != 0)
                    coords = np.argwhere(mask)
                    vals = pixel_counts[mask]
                    if vals.size == 0:
                        inject_list_small.append(int(inject_counts))
                        final_number_list_small.append(0)
                        zi_list_small.append(zi)

                        continue
                    # PSF coordinate grid
                    x_range = np.linspace(-limit, limit, pixel_counts.shape[1])
                    y_range = np.linspace(-limit, limit, pixel_counts.shape[0])

                    # append each photon's coordinate (if it is in extended bad, it is not used)
                    x_list = []
                    y_list = []

                    fov_radi=314
                    for (y_idx, x_idx), c in zip(coords, vals):
                        x = x_range[x_idx] + x_source
                        y = y_range[y_idx] + y_source
                        xi = int(x); yi = int(y)
                        if (xi, yi) in badset_expanded: # bad pixel selection
                            continue
                        if (xi-fov_center_skyx)**2+(yi-fov_center_skyy)**2 > fov_radi**2: # FOV selection (previous FOV selection was for source position, this selection is for photons)
                            continue
                        x_list.extend([xi] * int(c))
                        y_list.extend([yi] * int(c))

                    x_array = np.array(x_list, dtype=">i2")
                    y_array = np.array(y_list, dtype=">i2")
                    
                    if len(x_array) == 0:
                        inject_list_small.append(int(inject_counts))
                        final_number_list_small.append(0)
                        zi_list_small.append(zi)
                        continue                        
                    
                    # det and raw coordinates are not used in analysis but need some values for them to make hdu. So this value is not important.
                    # same for time, grade and other columns except sky coordinates 
                    detx_new = (x_array - 200).astype(">i2")
                    dety_new = (y_array - 200).astype(">i2")
                    rawx_new = (detx_new - 1).astype(">i2")
                    rawy_new = (dety_new - 1).astype(">i2")

                    total = int(len(x_array))

                    # time
                    time_array = rng.uniform(start + 0.1, stop, total).astype(">f8")

                    # grade（ここでは全部0をシャッフル）
                    grade = np.zeros(total, dtype=">i2")
                    rng.shuffle(grade)

                    # STATUS（16bitをpackbits→ 2byte/evt）
                    status = np.zeros((total, 16), dtype="u1")
                    status = np.packbits(status + 1, axis=1)

                    # PHA/PI
                    PI = np.full(total, 100)
                    PHA = (3.27 * PI + 1.87).astype(">i4")
                    
                    # create EVENTS table
                    event_table = Table(
                        [time_array, x_array, y_array, rawx_new, rawy_new, detx_new, dety_new, PHA, PI, grade, status],
                        names=("TIME", "X", "Y", "RAWX", "RAWY", "DETX", "DETY", "PHA", "PI", "GRADE", "STATUS"),
                    )
                    created_events = fits.BinTableHDU(data=event_table, name="EVENTS").data

                    # combine the oiginal EVENTS and created one and sort by time (but time sorting is not neccesary)
                    combined = fits.FITS_rec(np.concatenate([
                        EVENTS,
                        created_events.astype(EVENTS.dtype)  # dtype/エンディアンを既存に揃える
                    ]))
                    sort_idx = np.argsort(combined["TIME"])
                    combined = combined[sort_idx].astype(EVENTS.dtype)

                    # change EVENTS of original HDU
                    evt_hdu = original["EVENTS"].copy()
                    evt_hdu.data = combined

                    hdus = [hdu for hdu in original]
                    evt_idx = original.index_of("EVENTS")
                    hdus[evt_idx] = evt_hdu
                    new_hdul = fits.HDUList(hdus)

                    outfile = out_path / f"{base}.evt"
                    new_hdul.writeto(outfile, overwrite=True, output_verify="silentfix")
                

                    inject_list_small.append(int(inject_counts))
                    final_number_list_small.append(total)
                    # total_list_small.append(total)
                    # ListOfFlux_small.append(0)
                    zi_list_small.append(zi)

            except Exception as e:
                print(f"[ERR in {p.name}{lumi_str}] {e}")
        
        inject_list.append(inject_list_small)
        final_number_list.append(final_number_list_small)
        zi_list.append(zi_list_small)

        

    result['inject'] = inject_list
    result['z'] = zi_list
    result['finalnumber'] = final_number_list
    return result

###################### for Multiprocessing #########################
def run(icdir_str: str, idx: int, rho: float) -> dict:
    icdir = Path(icdir_str)
    result = process_evt_file(icdir, idx, rho)
    return result

def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) != 4:
        print("Usage: python inject_one_abs.py <ICDIR> <INDEX> <RHO>")
        return 1

    result = run(argv[1], int(argv[2]), float(argv[3]))

    print(result["icname"], result["idx"], result["rho"], result["status"])
    if result.get("note"):
        print("note:", result["note"])

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


# if __name__ == "__main__":
    
    
#     if len(sys.argv) != 3:
#         print("Usage: python 2rxs_remove_one.py <ICDIR> <INDEX>")
#         sys.exit(1)

#     icdir_str= sys.argv[1]
#     idx = int(sys.argv[2])
#     icdir = Path(icdir_str)

#     print(icdir, idx)
#     _icdir, _idx = process_evt_file(icdir, idx)
#     print(f"[DONE] {_icdir}, {_idx}")
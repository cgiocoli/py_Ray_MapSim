import scipy.fftpack as fftengine
import scipy.integrate as integrate
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from astropy.io import fits
import scipy.stats as stats
import pyccl as ccl
import sys
plt.ion()
c_speed = 2.99792458e+10
speed_factor = c_speed*1.e-7
GNewton=4.3011e-9

# ... DEMNUni_cov - Cosmology
h=0.67
om = 0.32
ol = 0.68
Omega_b=0.05
Omega_c=om-Omega_b
A_s= 2.1265e-09
T_CMB=2.7225
n_s=0.96
#.... loading the cosmology of pyccl
cosmo = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        A_s= A_s,
        #sigma8= 0.83,
        T_CMB=T_CMB,
        n_s=n_s)

print('')
print('   >>>   run: ini_params(IDsim,IDlos,pathsim) to initialize the parameters')
print('               IDsim = six digit of the simulation number (example 123456)')
print('               IDlos = a number from 0 to 63')
print('               pathsim = path of the light-cone simulations ')
print('')
def ini_params(IDsim,IDlos,pathsim):
    global ID_realisation
    ID_realisation = IDsim
    global str_ID_los
    if IDlos<10:
        str_ID_los = "0" + str(IDlos)
    else:
        std_id_los = str(IDlos)
    global path_sim
    path_sim = str(pathsim) +'/'+str(ID_realisation)+'/'+str_ID_los+'/'
    file_planes = path_sim + "planes_list.txt"
    global dmax
    zl_i,dmin,dmax = np.loadtxt(file_planes,unpack=True,usecols=[1,2,3])
    #...defining the path of the comoving distance file used in MapSim (for consistency)
    fil_comoving_dist = "comoving-dist_LCDM_DEMNUni_Cov.txt"
    z,dc = np.loadtxt(fil_comoving_dist,unpack=True,usecols=[0,1])
    dc = dc*speed_factor
    intep_z_from_dc = interpolate.interp1d(dc,z,kind='cubic')
    intep_dc_from_z = interpolate.interp1d(z,dc,kind='cubic')
    global zs
    zs = intep_z_from_dc(dmax)
    #...here if you want to plot
    #plt.plot(z,dc)
    #plt.plot(zs,dmax,marker='o',linestyle='')
    #plt.xlabel('z')
    #plt.ylabel('D [Mpc/h]')
    global dlc
    dlc = (dmin + dmax)*0.5
    global zl
    zl = intep_z_from_dc(dlc)
    print('--- source redshift availables (i, zs, zl) ---')
    for i in range(0,len(dmax)):
        print(i,zs[i],zl[i])
    print('----------------------------------')
    print(' ')
    print('   >>>   run: shoot_rays(up_to_plane), defining as up_to_plane the corresponding i of the zs desired ')

def shoot_rays(up_to_plane):
    print('zs = ', zs[up_to_plane])
    dsc = dmax[up_to_plane]
    dls = (dsc - dlc)/(1+zs[up_to_plane])
    ds = dsc/(1+zs[up_to_plane])
    dl = dlc/(1+zl)
    SigmaCrit = (c_speed*1e-5)**2/4.0/np.pi/GNewton*ds/dl/dls
    print('since python start from 0 the fits file ID will have a + 1')
    print(' ')
    print(' ... shooting rays ... ')

    for i in range(0,up_to_plane+1):
        if (i+1) < 10:
            i_str = "00" + str(i+1)
        else:
            i_str = "0" + str(i+1)
        file_fits = path_sim + "l."+i_str+".plane_4096.fits"
        print(i,file_fits)
        hdul = fits.open(file_fits)
        pixel_unit = hdul[0].header['PIXELUNIT']
        fov = hdul[0].header['PHYSICALSIZE']
        n_pixels = hdul[0].header['NAXIS1']
        d1 = hdul[0].header['DlLOW']
        d2 = hdul[0].header['DlUP']
        #print(d1*h, d2*h)
        pixel_unit = pixel_unit*h
        #print(dmin[i],dmax[i])
        #print(fov, pixel_unit,n_pixels)
        pixLMpc = fov/180.0*np.pi*dl[i]/n_pixels
        pixel_unit=pixel_unit/pixLMpc/pixLMpc
        mass_map = fits.getdata(file_fits, ext=0)
        kappa_i = mass_map/SigmaCrit[i]*pixel_unit
        if i==0:
            kappa = np.copy(kappa_i)
        else:
            kappa = kappa + kappa_i
    kappa_mean = np.mean(kappa)
    print(' ')
    print('kappa_mean_before_subtraction = ', kappa_mean)
    print(' ')
    print('saving kappa map in: kappa_'+str(up_to_plane+1)+'_'+str(ID_realisation)+'_'+str_ID_los+'.fits')
    kappa = kappa - kappa_mean
    hdu = fits.PrimaryHDU(kappa)
    hdu.header['zs'] = zs[up_to_plane]
    hdu.header['fov_in_deg'] = fov
    hdu.header['kappa_mean_before_subtraction'] = kappa_mean
    kappa_hdul = fits.HDUList([hdu])
    kappa_hdul.writeto('kappa_'+str(up_to_plane+1)+'_'+str(ID_realisation)+'_'+str_ID_los+'.fits',overwrite=True)
    print(' ')
    print('   >>>   measure_and_plot_PS(fits_file), defining fits_file the kappa_ID.fits file you want to load')

def compute_PS(input_map,FieldSize,nbins=128,lrmin=-2):
    """
    Compute the angular power spectrum of input_map.
    :param input_map: input map (n x n numpy array)
    :param FieldSize: the side-length of the input map in degrees
    :return: l, Pl - the power-spectrum at l
    """
    # set the number of pixels and the unit conversion factor
    npix = input_map.shape[0]
    factor = 2.0*np.pi/(FieldSize*np.pi/180.0)

    # take the Fourier transform of the input map:
    fourier_map = fftengine.fftn(input_map)/npix**2
    # compute the Fourier amplitudes

    fourier_amplitudes = np.abs(fourier_map)**2
    fourier_amplitudes = fourier_amplitudes.flatten()

    # compute the wave vectors
    kfreq = fftengine.fftfreq(input_map.shape[0])*input_map.shape[0]
    kfreq2D = np.meshgrid(kfreq, kfreq)

    # take the norm of the wave vectors
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()

    # set up k bins. The PS will be evaluated in these bins
    half = npix/2
    rbins = int(np.sqrt(2*half**2))+1
    #kbins = np.linspace(0.0,rbins,(rbins+1))
    #print(kbins,rbins)
    kbins = np.linspace(lrmin ,np.log10(rbins),nbins)
    kbins = 10**kbins
    #print(kbins)
    #return 0
    # use the middle points in each bin to define the values of k # where the PS is evaluated
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*factor

    # now compute the PS: calculate the mean of the
    # Fourier amplitudes in each kbin
    Pbins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    # return kvals and PS
    l=kvals[1:]
    Pl=Pbins[1:]/factor**2
    return l, Pl

def measure_and_plot_PS(fits_file,nbins=256,lrmin=-2):
    hdul = fits.open(fits_file)
    print(' ')
    print('reading ', fits_file)
    fov = hdul[0].header['fov_in_deg']
    zs = hdul[0].header['zs']
    print(' source redshift located at z_s = ', zs)
    lens = ccl.CMBLensingTracer(cosmo, zs)
    kappa = fits.getdata(fits_file, ext=0)
    #kappa = kappa.astype(float)
    l,P=compute_PS(kappa,fov,nbins,lrmin)
    print(' ')
    l = l[~np.isnan(P)]
    P = P[~np.isnan(P)]
    l = l [P>0]
    P = P [P>0]
    P = P [l>0]
    l = l [l>0]
    P = P [l<1e4]
    l = l [l<1e4]
    np.savetxt(fits_file + '_powerPS.txt',np.c_[l, P],fmt='%f     %e   ')
    print(' power spectrum saved in: ',fits_file+'_powerPS.txt')
    print(' ')
    print('number of points passing the criteria = ', len(l))
    plt.loglog(l,P*l*l)
    cls = ccl.angular_cl(cosmo, lens, lens, l)
    plt.loglog(l,cls*l*l/(2*np.pi)**2)
    plt.xlabel('$l$')
    plt.ylabel('$l^2 C(l)/4 \pi^2$')

    #...to compare with ...
    #file_in = "/Users/cgiocoli/Desktop/24_kappaBApp.fits"
    #kappa2 = fits.getdata(file_in, ext=0)
    #l,P=compute_PS(kappa2,fov,nbins,lrmin)
    #l = l[~np.isnan(P)]
    #P = P[~np.isnan(P)]
    #l = l [P>0]
    #P = P [P>0]
    #P = P [l>0]
    #l = l [l>0]
    #print(' ')
    #print('number of points passing the criteria = ', len(l))
    #plt.loglog(l,P*l*l,linestyle=":")

    plt.show()

def runAll_shoot_rays():
    for i in range(0,43):
        shoot_rays(i)

def runAll_measure_and_plot_PS(IDsim,IDlos):
    for i in range(0,43):
        measure_and_plot_PS("kappa_"+str(i+1)+"_"+str(IDsim)+"_"+str(IDlos)+".fits",128,-2)

def plot_PS(IDsim,IDlos):
    plt.xlabel('$l$')
    plt.ylabel('$l^2 C(l)/4 \pi^2$')
    snap =([9,16,24,29,33,39,43])
    for i in range(0,len(snap)):
        file_in = "kappa_"+str(snap[i])+"_"+str(IDsim)+"_"+str(IDlos)+".fits"
        hdul = fits.open(file_in)
        zs = hdul[0].header['zs']
        print(' source redshift located at z_s = ', zs)
        lens = ccl.CMBLensingTracer(cosmo, zs)
        l,P = np.loadtxt(file_in+"_powerPS.txt",unpack=True,usecols=[0,1])
        zs = answer = str(round(zs, 2))
        plt.loglog(l,P*l*l,label="$z_s=$"+str(zs))
        cls = ccl.angular_cl(cosmo, lens, lens, l)
        plt.loglog(l,cls*l*l/(2*np.pi)**2)

    plt.legend(loc=2,ncol=2)
    plt.show()

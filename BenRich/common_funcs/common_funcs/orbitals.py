import scipy.special as spe
import numpy as np

def psi_R(r,n,l):

    coeff = np.sqrt((2.0/n)**3 * spe.factorial(n-l-1) /(2.0*n*spe.factorial(n+l)))

    laguerre = spe.assoc_laguerre(2.0*r/n,n-l-1,2*l+1)

    return coeff * np.exp(-r/n) * (2.0*r/n)**l * laguerre

def psi_ang(phi,theta,l,m):

    sphHarm = spe.sph_harm(m,l,phi,theta)

    return sphHarm.real


def cart2sphe(x, y, z):
    '''
    3D Cartesian coordinates to spherical coordinates.

    input:
        x, y, z : numpy arrays
    '''
    xy2     = x**2 + y**2
    r       = np.sqrt(xy2 + z**2)
    theta   = np.arctan2(np.sqrt(xy2), z) # the polar angle in radian angles
    phi     = np.arctan2(y, x)            # the azimuth angle in radian angles
    phi[phi < 0] += np.pi * 2             # np.arctan2 returns the angle in the range [-pi, pi]

    return r, theta, phi

def HFunc_cart(x,y,z,n,l,m):
    r, theta, phi = cart2sphe(x,y,z)
    return HFunc_polar(r,theta,phi,n,l,m)

def HFunc_polar(r,theta,phi,n,l,m):
    '''
    Hydrogen wavefunction // a_0 = 1

    INPUT
        r: Radial coordinate
        theta: Polar coordinate
        phi: Azimuthal coordinate
        n: Principle quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number

    OUTPUT
        Value of wavefunction
    '''


    return psi_R(r,n,l) * psi_ang(phi,theta,l,m)
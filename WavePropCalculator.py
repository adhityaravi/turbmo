import math

PR = 0.72
GAMMA = 1.4
LREF = 1.0
S = 110.4
T0 = 273.15

ma = 0.1
re = 1000.0

a_plus = 30
l_plus = 500
t_plus = 40
c_plus = l_plus/t_plus

t8 = 1.0/(1.0 + (GAMMA-1.0)/2.0*ma*ma)
u8 = ma*math.sqrt(t8)
rho8 = t8**(1.0/(GAMMA-1.0))
mue8 = (t8**(3.0/2.0))*((1.0+(S/T0))/(t8+(S/T0)))
re0 = re*mue8/rho8/u8
p8 = (1.0/GAMMA)*t8**(GAMMA/(GAMMA-1))

# compute cf based on the empirical law by Smits
# (other laws could also be used)
cf = 0.024*re**(-0.25)

# we assume rho_wall = rho_infinity, thus this
# is only valid for nearly incompressible flow
utau = math.sqrt(cf/2)*u8

wavelength = l_plus/re*math.sqrt(2/cf)
period = 2*t_plus*mue8/(cf*u8*u8*re0)
wavespeed = c_plus*utau

print("Wavelength: ",str(wavelength),
      " period: ",str(period),
      " wavespeed1: ",str(wavespeed))

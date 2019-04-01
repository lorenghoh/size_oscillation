# Define conditional cloud field
cp = 1004.    # Heat capacity at constant pressure for dry air [J kg^-1 K^-1]
Rv = 461.      # Gas constant of water vapor [J kg^-1 K^-1]
Rd = 287.      # Gas constant of dry air [J kg^-1 K^-1]
p_0 = 100000.
lam = Rv/Rd - 1.

def theta_v(p, T, qv, qn, qp):
    return T*(1. + lam*qv - qn - qp)*(p_0/p)**(Rd/cp)
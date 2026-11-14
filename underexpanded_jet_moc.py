# Gas dynamics task 2

### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math
import matplotlib.tri as mtri

### INPUTS #######################################################################################################################
pa = 101325     # [Pa], ambient pressure

pe = 2*pa       # [Pa], exit pressure
M_e = 1.5         # [-], exit Mach number
gamma = 1.4     # [-], ratio of specific heat
height_full = 1 # [m], full diameter of nozzle
h_lip = height_full/2
phi_e = 0       # [rad], exit flow angle

n_ray = 3           # [-], rays in the fan INCLUDING the two bounding rays (>= 2)

### FUNCTIONS #######################################################################################################################
def get_p_total(ps, gamma, M):
    """
    Input:
        ps [Pa]: static pressure
        gamma [-]: specific heat ratio
        M [-]: Mach number
    Return:
        p_tot [Pa]: total pressure
    """
    p_tot = ps /( (1+ (gamma-1)/2 * M**2)**(-gamma/(gamma-1)) )
    return p_tot

def get_p_static(pt, gamma, M):
    """
    Input:
        pt [Pa]: total pressure
        gamma [-]: specific heat ratio
        M [-]: Mach number
    Return:
        ps [Pa]: static pressure
    """
    ps = pt * (1 + (gamma-1)/2 * M**2)**(-gamma/(gamma-1))
    return ps

def get_M_isentropic(ps, pt, gamma):
    """ Get Mach number, based on isentropic relations
    Input:
        ps [Pa]: static pressure
        pt [Pa]: total pressure
        gamma [-]: specific heat ratio
    Return:
        M_isentropic [-]: Mach number
    """
    M_isentropic = np.sqrt( ((ps/pt)**(-(gamma-1)/gamma) -1) * 2/(gamma-1) )
    return M_isentropic

def get_nu_PM(gamma, M):
    """ Get Prandtly-Meyer angle based on PM function
    Input: 
        gamma [-]: specific heat ratio
        M [-]: Mach number
    Return:
        nu [rad]: Prandtl-Meyer angle
    """
    nu = np.sqrt( (gamma+1)/(gamma-1) ) * np.arctan(np.sqrt( (gamma-1)/(gamma+1) * (M**2-1) )) - np.arctan(np.sqrt( (M**2-1) ))
    return nu

def get_phi_Gamma_minus(nu_A, phi_A, nu_B):
    """ Get flow angle angle based on Gamma_minus characteristic"""
    phi_B = nu_B + phi_A - nu_A 
    return phi_B

def get_mu_Mach_angle(M):
    """ Get Mach angle based on Mach angle relationship"""
    mu = np.arctan( 1/(np.sqrt(M**2-1)) )
    return mu

def get_alpha(mu, phi):
    """ Get flow deflection angle
    Input: 
        mu [rad]: Mach angle
        phi [rad]: flow angle
    Return:
        alpha [rad]: flow deflection angle
    """
    return phi - mu

def get_M_from_nu(gamma, nu):
    """Find Mach number using Brentq root finding method """
    return brentq(lambda M: get_nu_PM(gamma, M) - nu, 1.0, 50.0)

def intersect(xQ, yQ, mQ, xR, yR, mR):
    """Intersection of line through Q (slope mQ) and line through R (slope mR)"""
    xP = (yR - yQ + mQ*xQ - mR*xR) / (mQ - mR)
    yP = yQ + mQ*(xP - xQ)
    return xP, yP

### CALCULATIONS #######################################################################################################################
# Region A to B
p_tot = get_p_total(pe, gamma, M_e) # [Pa], total pressure. Stays cst everywhere due to isentropic
M_B = get_M_isentropic(pa, p_tot, gamma) # [-], Mach number at region B
nu_A = get_nu_PM(gamma, M_e) # [rad], Prandtl-Meyer angle at region A
nu_B = get_nu_PM(gamma, M_B) # [rad], Prandtl-Meyer angle at region B
phi_B = nu_B + phi_e - nu_A  # [rad], flow angle in region B
mu_A = get_mu_Mach_angle(M_e) # [rad], Mach angle at region A
mu_B = get_mu_Mach_angle(M_B) # [rad], Mach angle at region B
alpha_A = get_alpha(mu_A, phi_e) # [rad]: expansion flow deflection at region A
alpha_B = get_alpha(mu_B, phi_B) # [rad]: expansion flow deflection at region B


### FAN: intermediate rays #######################################################################################################################
alpha_fan_list = np.linspace(alpha_A, alpha_B, n_ray)      # equal steps in alpha

def f(M_i, alpha_i):    # Residual: function for which we try to find M as nu_i and M_i are unknown
    return get_nu_PM(gamma, M_i) - get_mu_Mach_angle(M_i) - alpha_i - nu_A + phi_e 

M_fan_list = []
for alpha_i in alpha_fan_list:   # For each ray
    M_fan_list.append(brentq(f, M_e, M_B, args=(alpha_i,)))  # Find M in the function, for a given alpha_i
M_fan_list = np.array(M_fan_list)

mu_fan_list  = get_mu_Mach_angle(M_fan_list)
phi_fan_list = alpha_fan_list + mu_fan_list
nu_fan_list  = get_nu_PM(gamma, M_fan_list)
p_fan_list   = get_p_static(p_tot, gamma, M_fan_list)


### REFLECTION CENTRELINE #######################################################################################################################
V_minus = phi_fan_list + nu_fan_list                 # Gamma- invariant of each incoming ray
V_plus = -V_minus

shape = (n_ray, n_ray)                     # index [j, i], only j <= i is used
x = np.full(shape, np.nan)
y = np.full(shape, np.nan)
nu_intersection = np.full(shape, np.nan)
phi_intersection = np.full(shape, np.nan)
M_intersection = np.full(shape, np.nan)
mu_intersection = np.full(shape, np.nan)

for j in range(n_ray):                         # reflected wave j
    for i in range(j, n_ray):                  # incoming ray i
        # Find flow state at intersections
        V_in = V_minus[i]
        V_out = V_plus[j]
        nu_intersection[j, i]  = 0.5*(V_in - V_out)  
        phi_intersection[j, i] = 0.5*(V_in + V_out)
        M_intersection[j, i]   = get_M_from_nu(gamma, nu_intersection[j, i])
        mu_intersection[j, i]  = get_mu_Mach_angle(M_intersection[j, i])

        # Q: on gamma minus to current point P         
        # R: on gamma plus to current point P             
        # Gamma minus
        if j == 0: # Straight fan ray from lip until 1st intersection
            xQ, yQ = 0.0, h_lip                     # previous point position (before 1st intersection) is the lip
            slope_incoming_char = np.tan(alpha_fan_list[i])     # slope is directly alpha_i0
        else: # After intersection
            xQ, yQ = x[j-1, i], y[j-1, i]           # previous point position find in list
            alpha_Q = phi_intersection[j-1, i] - mu_intersection[j-1, i]
            alpha_P = phi_intersection[j, i] - mu_intersection[j, i]
            slope_incoming_char = np.tan(0.5*(alpha_Q + alpha_P))

        # Gamma plus
        if i == j: # Centreline point
            x[j, j], y[j, j] = xQ - yQ/slope_incoming_char, 0.0
        else: # After intersection
            xR, yR = x[j, i-1], y[j, i-1]
            alpha_R = phi_intersection[j, i-1] + mu_intersection[j, i-1]
            alpha_P = phi_intersection[j, i] + mu_intersection[j, i]
            slope_outgoing_char = np.tan(0.5*(alpha_R + alpha_P))

            # find xp and yp using intersect of 2 slopes Gamma- and Gamma+
            x[j, i], y[j, i] = intersect(xQ, yQ, slope_incoming_char, xR, yR, slope_outgoing_char)


### REFLECTION THE JET BOUNDARY ####################################################################################
# Gamma+ wave j (reflected waves) arrives at the jet boundary and is reflected as a Gamma- wave 
# Index [k, j] (k <= j): reflected wave k crosses Gamma+ wave j, the diagonal [j, j] are the boundary point
V_boundary = 2*nu_B + V_plus                   # Gamma- invariant of each reflected wave: V- = V+ + 2*nu_B

x_ref = np.full(shape, np.nan)
y_ref = np.full(shape, np.nan)
nu_ref = np.full(shape, np.nan)
phi_ref = np.full(shape, np.nan)
M_ref = np.full(shape, np.nan)
mu_ref = np.full(shape, np.nan)

for j in range(n_ray):                         # Gamma+ wave j
    for k in range(j+1):                       # reflected Gamma- wave k
        # Find flow state (same equations as before)
        V_in  = V_boundary[k]
        V_out = V_plus[j]
        nu_ref[k, j]  = 0.5*(V_in - V_out)
        phi_ref[k, j] = 0.5*(V_in + V_out)
        M_ref[k, j]   = get_M_from_nu(gamma, nu_ref[k, j])
        mu_ref[k, j]  = get_mu_Mach_angle(M_ref[k, j])

        # R: previous point on Gamma+ wave j
        if k == 0:                             # not crossed any reflected wave yet: last point of first reflection
            xR, yR = x[j, n_ray-1], y[j, n_ray-1]
            alpha_R = phi_intersection[j, n_ray-1] + mu_intersection[j, n_ray-1]
        else:
            xR, yR = x_ref[k-1, j], y_ref[k-1, j]
            alpha_R = phi_ref[k-1, j] + mu_ref[k-1, j]
        alpha_P = phi_ref[k, j] + mu_ref[k, j]
        slope_outgoing_char = np.tan(0.5*(alpha_R + alpha_P))

        if k == j:                             # Boundary point: Gamma+ wave j meets the jet boundary
            if j == 0:
                xB, yB, phi_prev = 0.0, h_lip, phi_B                          # previous boundary point = lip
            else:
                xB, yB, phi_prev = x_ref[j-1, j-1], y_ref[j-1, j-1], phi_ref[j-1, j-1]
            slope_boundary = np.tan(0.5*(phi_prev + phi_ref[j, j]))           # boundary is a streamline: slope = tan(phi)
            x_ref[j, j], y_ref[j, j] = intersect(xB, yB, slope_boundary, xR, yR, slope_outgoing_char)
        else:                                  # Interior point: reflected wave k crosses Gamma+ wave j
            xQ, yQ = x_ref[k, j-1], y_ref[k, j-1]                             # previous point on reflected wave k
            alpha_Q = phi_ref[k, j-1] - mu_ref[k, j-1]
            alpha_P = phi_ref[k, j] - mu_ref[k, j]
            slope_reflected_char = np.tan(0.5*(alpha_Q + alpha_P))
            x_ref[k, j], y_ref[k, j] = intersect(xQ, yQ, slope_reflected_char, xR, yR, slope_outgoing_char)

x_boundary = np.r_[0.0, np.diag(x_ref)]        # lip + all boundary points
y_boundary = np.r_[h_lip, np.diag(y_ref)]

### REFLECTION OFF THE CENTRELINE (AGAIN) ##############################################################################
# Reflected Gamma- wave k arrives at the centreline and is reflected as a Gamma+ wave m = k.
# Index [m, k] (m <= k): second-reflection wave m crosses reflected wave k.  The diagonal [k, k] is the centreline point.
V_centre = -V_boundary                         # Gamma+ invariant after the centreline: K+ = -K-  (phi = 0 there)

x_re = np.full(shape, np.nan)
y_re = np.full(shape, np.nan)
nu_re = np.full(shape, np.nan)
phi_re = np.full(shape, np.nan)
M_re = np.full(shape, np.nan)
mu_re = np.full(shape, np.nan)

for m in range(n_ray):                         # DD wave m
    for k in range(m, n_ray):                  # CC wave k
        V_in, V_out = V_boundary[k], V_centre[m]
        nu_re[m, k]  = 0.5*(V_in - V_out)
        phi_re[m, k] = 0.5*(V_in + V_out)
        M_re[m, k]   = get_M_from_nu(gamma, nu_re[m, k])
        mu_re[m, k]  = get_mu_Mach_angle(M_re[m, k])

        # Q: previous point on CC wave k
        if m == 0:
            xQ, yQ = x_ref[k, n_ray-1], y_ref[k, n_ray-1]
            alpha_Q = phi_ref[k, n_ray-1] - mu_ref[k, n_ray-1]
        else:
            xQ, yQ = x_re[m-1, k], y_re[m-1, k]
            alpha_Q = phi_re[m-1, k] - mu_re[m-1, k]
        alpha_P = phi_re[m, k] - mu_re[m, k]
        slope_reflected_char = np.tan(0.5*(alpha_Q + alpha_P))

        if k == m:                             # centreline point
            x_re[m, k], y_re[m, k] = xQ - yQ/slope_reflected_char, 0.0
        else:                                  # R: previous point on DD wave m
            xR, yR = x_re[m, k-1], y_re[m, k-1]
            alpha_R = phi_re[m, k-1] + mu_re[m, k-1]
            alpha_P = phi_re[m, k] + mu_re[m, k]
            slope_second_char = np.tan(0.5*(alpha_R + alpha_P))
            x_re[m, k], y_re[m, k] = intersect(xQ, yQ, slope_reflected_char, xR, yR, slope_second_char)


### DD WAVES UNTIL JET BOUNDARY ###########################################################
# Continue jet boundary after last point (x_boundary and y_boundary are already defined above)
xb_last, yb_last = x_boundary[-1], y_boundary[-1]
m_b_last = np.tan(phi_ref[n_ray-1, n_ray-1])

# Last computed point of every DD wave (after the last CC wave) and direction
xe, ye = x_re[:, n_ray-1], y_re[:, n_ray-1]
alpha_end = phi_re[:, n_ray-1] + mu_re[:, n_ray-1]       # Gamma+ direction, constant beyond the last CC wave


### CHECK ORDER OF ARRIVING ###########################################################
# Where each DD wave hits the extended boundary
x_hit, y_hit = np.zeros(n_ray), np.zeros(n_ray)
for m in range(n_ray):
    x_hit[m], y_hit[m] = intersect(xe[m], ye[m], np.tan(alpha_end[m]), xb_last, yb_last, m_b_last)

# Order they left the centreline vs order they arrive at the boundary
x_dep = np.diag(x_re)                                    # departure points on the centreline
crossed_pairs = [(a, b) for a in range(n_ray) for b in range(n_ray)
                 if x_dep[a] < x_dep[b] and x_hit[a] > x_hit[b]]    # a left first but arrives later

assert crossed_pairs, "No DD waves crossed: no shock found for these inputs"

crossed_DD = set()                  # start with an empty set
for a, b in crossed_pairs:          # crossed_pairs = [(2, 3), (3, 4)] -> {2, 3, 4}
    crossed_DD.add(a)               # add both wave numbers to the set
    crossed_DD.add(b)

# Approximate shock position: earliest intersection of the extended lines of a crossed pair
shock_x, shock_y = np.inf, None     # start value: "nothing found yet"
for a, b in crossed_pairs:
    xs_, ys_ = intersect(xe[a], ye[a], np.tan(alpha_end[a]), xe[b], ye[b], np.tan(alpha_end[b]))
    if xs_ < shock_x:
        shock_x, shock_y = xs_, ys_

print(f"DD waves arrive out of order: shock forms near x = {shock_x:.3f} m, y = {shock_y:.3f} m")


### PLOTTING ###########################################################
def draw_waves(ax, c=('blue', 'green', 'magenta', 'red'), lw=0.8):
    """ Draw the wave pattern (AA, BB, CC, DD), the jet boundary, the shock onset and the centreline
    Input:
        ax [-]: matplotlib axes to draw on
        c [-]: colours of the wave groups (AA, BB, CC, DD)
        lw [-]: line width of the waves
    Use:
        x, y, x_ref, y_ref, x_re, y_re [m]: crossing positions of the BB, CC and DD waves
        x_hit, y_hit [m]: arrival points of the DD waves at the extended jet boundary
        x_boundary, y_boundary [m]: jet boundary points
        crossed_DD [-]: set of DD wave numbers that are part of a crossed pair
        shock_x, shock_y [m]: shock formation point
    Return:
        None:  plot is drawn on ax
    """
    for i in range(n_ray):                     # AA
        ax.plot(np.r_[0.0, x[:i+1, i]], np.r_[h_lip, y[:i+1, i]], color=c[0], lw=lw)
    for j in range(n_ray):                     # BB
        ax.plot(np.r_[x[j, j:], x_ref[:j+1, j]], np.r_[y[j, j:], y_ref[:j+1, j]], color=c[1], lw=lw)
    for k in range(n_ray):                     # CC
        ax.plot(np.r_[x_ref[k, k:], x_re[:k+1, k]], np.r_[y_ref[k, k:], y_re[:k+1, k]], color=c[2], lw=lw)
    for m in range(n_ray):                     # DD, extended to the boundary
        if m in crossed_DD:
            colour = 'orange'                  # orange if this wave crossed another one
        else:
            colour = c[3]                      # normal DD colour
        ax.plot(np.r_[x_re[m, m:], x_hit[m]], np.r_[y_re[m, m:], y_hit[m]], color=colour, lw=lw)

    ax.plot(x_boundary, y_boundary, color='red', label='Jet boundary')
    ax.plot(x_boundary[1:], y_boundary[1:], color='red', marker='o', linestyle='')
    x_ext = x_hit.max()
    ax.plot([xb_last, x_ext], [yb_last, yb_last + m_b_last*(x_ext - xb_last)], color='red', ls='--')
    ax.plot(shock_x, shock_y, 'kx', markersize=10, markeredgewidth=2)
    ax.axhline(0, color='black')
    ax.set_xlabel('x [m]'); ax.set_ylabel('h [m]'); ax.set_aspect('equal')


### NODES WHERE THE FLOW STATE IS KNOWN ###
def valid(xa, ya, Ma):
    """ Keep only the computed crossings that can be used for the colour field
    Input:
        xa [m]: x positions of an array of crossings (NaN if unused)
        ya [m]: y positions of the same array
        Ma [-]: Mach numbers at the same crossings
    Use:
        shock_x [m]: x position of the shock onset, points after it are cut away
    Return:
        xa [m]: x positions of the crossings, 1D array
        ya [m]: y positions of the crossings, as an 1D array
        Ma [-]: Mach numbers of the crossings, 1D array
    """
    ok = np.isfinite(xa) & np.isfinite(ya) & (xa <= shock_x)    # check for valid crossings before shock
    return xa[ok], ya[ok], Ma[ok]

parts = [valid(x, y, M_intersection),          # BB x AA crossings
         valid(x_ref, y_ref, M_ref),           # CC x BB crossings
         valid(x_re, y_re, M_re)]              # DD x CC crossings
xn_list, yn_list, Mn_list = [], [], []
for p in parts:
    xn_list.append(p[0])        # x positions of this group
    yn_list.append(p[1])        # y positions
    Mn_list.append(p[2])        # Mach numbers
xn = np.concatenate(xn_list)    # glue into 1 big list
yn = np.concatenate(yn_list)
Mn = np.concatenate(Mn_list)

# Add 2 points in fan region
eps = 0.02*height_full
xf = np.r_[eps*np.cos(alpha_fan_list), 0.5*x[0, :], 0.0, 0.0]
yf = np.r_[h_lip + eps*np.sin(alpha_fan_list), 0.5*(h_lip + y[0, :]), 0.0, h_lip]
Mf = np.r_[M_fan_list, M_fan_list, M_e, M_B]   # last two: nozzle exit on the axis, and the lip
ok = np.isfinite(xf) & np.isfinite(yf)
xn, yn, Mn = np.r_[xn, xf[ok]], np.r_[yn, yf[ok]], np.r_[Mn, Mf[ok]]

# Remove duplicate points
_, u = np.unique(np.round(np.c_[xn, yn], 9), axis=0, return_index=True)
xn, yn, Mn = xn[u], yn[u], Mn[u]
pn = get_p_static(p_tot, gamma, Mn) / pa       # static pressure ratio p/pa

# triangulate, then cut away triangles outside the jet (above the boundary or beyond the shock)
tri = mtri.Triangulation(xn, yn)
cx = xn[tri.triangles].mean(axis=1)             # cx and cy are the centres of the triangles
cy = yn[tri.triangles].mean(axis=1)
xb_m, yb_m = np.r_[x_boundary, shock_x], np.r_[y_boundary, shock_y]   # jet boundary closed by the shock point

too_far_right = cx > xb_m[-1]                   # triangles that are too far right
boundary_height = np.interp(cx, xb_m, yb_m)     # height of the boundary line at the x of each triangle centre
too_high = cy > boundary_height                 # triangles above that line
outside = too_far_right | too_high              # outside the jet if too far right OR too high
tri.set_mask(outside)                           # hide those triangles so they are not coloured


### PLOTS ###
def plot_field(values, label, title):
    """ Plot a flow property as a coloured field inside the jet, with the waves on top
    Input:
        values [-]: value of the property at every node (Mach number or pressure ratio)
        label [-]: text next to the colour bar
        title [-]: title of the plot
    Use:
        tri [-]: triangulation of the nodes, with the triangles outside the jet cut off
    Return:
        fig [-]: matplotlib figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    cf = ax.tricontourf(tri, values, levels=60, cmap='viridis')
    fig.colorbar(cf, ax=ax, label=label, shrink=0.8)
    draw_waves(ax, c=('white', 'white', 'white', 'white'), lw=0.6)
    ax.set_title(title)
    return fig

plot_field(Mn, 'Mach number [-]', 'Jet stream plot: Mach number')
plot_field(pn, 'Static pressure ratio p/pa [-]', 'Jet stream plot: static pressure ratio')

fig, ax = plt.subplots(figsize=(13, 5))        # original wave plot
draw_waves(ax)
ax.plot(0, h_lip, 'ko', label=f'Fan origin (h = {h_lip:g} m)')
ax.legend(); ax.grid(alpha=0.3)
plt.show()
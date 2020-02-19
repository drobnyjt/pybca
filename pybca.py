import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, Polygon, box
from shapely.ops import nearest_points
from star import ProtonSTARCalculator, ProtonMaterials, AlphaSTARCalculator, AlphaMaterials

#constants
e = 1.602e-19
amu = 1.66e-27
angstrom = 1e-10
eps0 = 8.85e-12
a0 = 0.52918e-10
K = 4.*np.pi*eps0
me = 9.11e-31

#screening
phi_coef = np.array([0.191, 0.474, 0.335])
phi_args = np.array([-0.279, -0.637, -1.919])

class Particle:
    def __init__(self, m, Z, E, dir_cos, pos, incident=False, track_trajectories=False):
        self.m = m
        self.Z = Z

        self.E = E
        self.dir_cos = np.array(dir_cos)
        self.pos = np.array(pos)
        self.pos_old = np.array(pos)
        self.t = 0.0
        self.incident = incident
        self.stopped = False
        self.left = False
        self.first_step = self.incident
        self.trajectory = [[self.x, self.y, self.z]]
        self.track_trajectories = track_trajectories

    def add_trajectory(self):
        if self.track_trajectories: self.trajectory.append([self.x, self.y, self.z])

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def z(self):
        return self.pos[2]

    @x.setter
    def x(self, x_new):
        self.pos[0] = x_new

    @y.setter
    def y(self, y_new):
        self.pos[1] = y_new

    @z.setter
    def z(self, z_new):
        self.pos[2] = z_new

    @property
    def cosx(self):
        return self.dir_cos[0]

    @property
    def cosy(self):
        return self.dir_cos[1]

    @property
    def cosz(self):
        return self.dir_cos[2]

    @cosx.setter
    def cosx(self, cosx_new):
        self.dir_cos[0] = cosx_new

    @cosy.setter
    def cosy(self, cosy_new):
        self.dir_cos[1] = cosy_new

    @cosz.setter
    def cosz(self, cosz_new):
        self.dir_cos[2] = cosz_new

class Material:
    def __init__(self, n, m, Z, Es, Eb=0.0, thickness=2*1e-6, depth=100*1e-6, use_PSTAR=False, use_ASTAR=False, STAR_material=None):
        self.n = n
        self.m = m
        self.Z = Z
        self.Eb = Eb
        self.Es = Es
        self.energy_barrier_position = -2.*self.n**(-1./3.)/np.sqrt(2.*np.pi)
        self.energy_barrier_thickness = 2.*self.n**(-1./3.)/np.sqrt(2.*np.pi)
        self.surface_position = 0.

        self.geometry = Polygon((
            (0.0, -thickness*angstrom),
            (0.0, thickness*angstrom),
            (0.25*depth*angstrom, thickness*angstrom),
            (0.25*depth*angstrom, 4.*thickness*angstrom),
            (0.0, 4.*thickness*angstrom),
            (0.0, 6.*thickness*angstrom),
            (depth*angstrom, 6.*thickness*angstrom),
            (depth*angstrom, 4.*thickness*angstrom),
            (0.75*depth*angstrom, 4.*thickness*angstrom),
            (0.75*depth*angstrom, thickness*angstrom),
            (depth*angstrom, thickness*angstrom),
            (depth*angstrom, -thickness*angstrom)))

        self.geometry = Polygon((
            (0.0, -thickness),
            (0.0, thickness),
            (depth, thickness),
            (depth, -thickness)
        ))

        self.energy_barrier_geometry = self.geometry.buffer(self.energy_barrier_thickness)
        self.simulation_boundary = box(*self.energy_barrier_geometry.bounds)

        self.use_PSTAR = use_PSTAR
        self.use_ASTAR = use_ASTAR

        self.STAR_material = STAR_material
        if use_ASTAR:
            self.AlphaSTARCaclulator = AlphaSTARCalculator(STAR_material)
        if use_PSTAR:
            self.ProtonSTARCalculator = ProtonSTARCalculator(STAR_material)

    def inside(self, pos):
        point = Point(pos[0], pos[1])
        return point.within(self.geometry)

    def inside_energy_barrier(self, pos):
        point = Point(pos[0], pos[1])
        return point.within(self.energy_barrier_geometry)

    def inside_simulation_boundary(self, pos):
        point = Point(pos[0], pos[1])
        return point.within(self.simulation_boundary)

    def mfp(self, pos):
        return self.n**(-1./3.)

    def Z_eff(self, pos):
        return self.Z

    def number_density(self, pos):
        return self.n

    def mass(self, pos):
        return self.m

    def electronic_stopping_factor(self, particle_1):
        Za = particle_1.Z
        Zb = self.Z_eff(particle_1.pos)
        Ma = particle_1.m
        E = particle_1.E
        pos = particle_1.pos
        Mb = self.mass(pos)

        if not self.inside(particle_1.pos):
            stopping_factor = 0.
        else:
            if particle_1.Z == 1 and self.use_PSTAR:
                electronic_stopping_power = self.ProtonSTARCalculator.calculate_electronic_stopping_powers(E/e/1E6) #MeV cm2 / gram
                stopping_factor = electronic_stopping_power*(self.number_density(pos)*Mb)*1E5*e #J/m
            elif particle_1.Z == 2 and self.use_ASTAR:
                electronic_stopping_power = self.ProtonSTARCalculator.calculate_electronic_stopping_powers(E/e/1E6) #MeV cm2 / gram
                stopping_factor = electronic_stopping_power*(self.number_density(pos)*Mb)*1E5*e #J/m
            elif E > 25E3*e:
                v = np.sqrt(2.0*E/Ma) #[m/s]
                print(v)
                v = c*np.sqrt(E*2.*c^2*Ma + E)/(E - Ma*c^2)
                print(v)
                exit()
                I = 10.*e*Zb #[J]
                n = self.number_density(pos)*Zb #[electrons/m3]
                stopping_factor = 4.0*np.pi*n*Za**2/(me*v**2)*(e**2/4.0/np.pi/eps0)**2*np.log(2.0*me*v**2/I) #[Joules/meter]
            else:
                #TRIDYN version of Lindhard-Scharff electronic stopping
                Sel = 1.212*(Za**(7./6.)*Zb)/((Za**(2./3.) + Zb**(2./3.))**(3./2.))*np.sqrt(E/Ma*amu/e)
                stopping_factor = self.number_density(pos)*Sel*angstrom**2*e

        return stopping_factor

def phi(xi):
    return np.sum(phi_coef*np.exp(phi_args*xi))

def dphi(xi):
    return np.sum(phi_args*phi_coef*np.exp(phi_args*xi))

def screening_length(Za, Zb):
    return 0.8853*a0/(np.sqrt(Za) + np.sqrt(Zb))**(2./3.)

def doca_function(x0, beta, reduced_energy):
    return x0 - phi(x0)/reduced_energy - beta**2/x0

def diff_doca_function(x0, beta, reduced_energy):
    return beta**2/x0**2 - dphi(x0)/reduced_energy + 1

def binary_collision(particle_1, particle_2, material, impact_parameter, tol=1e-6, max_iter=100):
    #If recoil outside surface, skip collision
    if not material.inside(particle_2.pos):
        return 0., 0., 0., 0., 0.

    Za = particle_1.Z
    Zb = particle_2.Z
    Ma = particle_1.m
    Mb = particle_2.m
    E0 = particle_1.E
    mu = Mb/(Ma + Mb)

    #Lindhard screening length and reduced energy; nondimensionalized impact parameter
    a = screening_length(Za, Zb)
    reduced_energy = K*a*mu/(Za*Zb*e**2)*E0
    beta = impact_parameter/a

    #Guess from analytic solution to unscreened case
    #M. H. Mendenhall & R. A. Weller, 1991
    x0 = 1./2./reduced_energy + np.sqrt((1./2./reduced_energy)**2 + beta**2)

    #Newton-Raphson method
    err = 1
    for _ in range(max_iter):
        xn = x0 - doca_function(x0, beta, reduced_energy)/diff_doca_function(x0, beta, reduced_energy)
        err = np.abs(xn - x0)/xn
        x0 = xn
        if err < tol:
            break
    else:
        raise ValueError('Newton-Raphson exceeded {max_iter} iterations.')

    #See M. H. Mendenhall & R. A. Weller, 1991 and 2005 for theta calculation
    f = lambda x: (1 - phi(x)/x/reduced_energy - (beta/x)**2)**(-1./2.)
    lambda_0 = (1./2. + (beta/x0)**2/2. - dphi(x0)/2./reduced_energy)**(-1./2.)
    alpha = 1./12.*(1. + lambda_0 + 5.*(0.4206*f(x0/0.9072) + 0.9072*f(x0/0.4206)))
    theta = np.pi*(1. - beta * alpha / x0)

    #See Eckstein, Computer Simulation of Ion-Solid Interactions
    t = x0*a*np.sin(theta/2.)
    psi = np.arctan2(np.sin(theta), (Ma/Mb) + np.cos(theta))
    T = 4.*(Ma*Mb)/(Ma + Mb)**2*E0*(np.sin(theta/2.))**2

    return theta, psi, T, t, x0

def update_coordinates(particle_1, particle_2, material, phi_azimuthal, theta, psi, T, t):
    #update position of moving particle
    mfp = material.mfp(particle_1.pos)
    particle_1.pos_old[:] = particle_1.pos

    #TRIDYN style "atomically rough surface"
    #Scatters first mfp step to avoid spatial correlation
    if particle_1.first_step:
        mfp *= np.random.uniform(0., 1,)
        particle_1.first_step = False

    #Have to subtract previous asymptotic deflection and add next to get correct trajectory
    free_flight_path = mfp - t + particle_1.t
    particle_1.pos[:] = particle_1.pos + free_flight_path*particle_1.dir_cos
    particle_1.t = t
    #breakpoint()

    #update angular coordinates of incident particle
    ca, cb, cg = particle_1.dir_cos
    sa = np.sin(np.arccos(ca))
    cphi = np.cos(phi_azimuthal)
    sphi = np.sin(phi_azimuthal)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    ca_new = cpsi*ca + spsi*cphi*sa
    cb_new = cpsi*cb - spsi/sa*(cphi*ca*cb - sphi*cg)
    cg_new = cpsi*cg - spsi/sa*(cphi*ca*cg + sphi*cb)
    dir_cos_new = [ca_new, cb_new, cg_new]
    dir_cos_new /= np.linalg.norm(dir_cos_new)
    particle_1.dir_cos[:] = dir_cos_new

    #update angular coordinates of secondary particle
    psi_b = np.arctan2(-np.sin(theta), 1. - np.cos(theta))
    cpsi_b = np.cos(psi_b)
    spsi_b = np.sin(psi_b)
    ca_new = cpsi_b*ca + spsi_b*cphi*sa
    cb_new = cpsi_b*cb - spsi_b/sa*(cphi*ca*cb - sphi*cg)
    cg_new = cpsi_b*cg - spsi_b/sa*(cphi*ca*cg + sphi*cb)
    dir_cos_new = [ca_new, cb_new, cg_new]
    dir_cos_new /= np.linalg.norm(dir_cos_new)
    particle_2.dir_cos[:] = dir_cos_new

    #update energy coordinates
    Za = particle_1.Z
    Ma = particle_1.m
    Mb = material.m
    Zb = material.Z_eff(particle_1.pos)
    E = particle_1.E
    a = screening_length(Za, Zb) #Z_eff?

    stopping_factor = material.electronic_stopping_factor(particle_1)
    Enl = free_flight_path*stopping_factor

    #Energy calculation - make sure stopping doesn't reduce energy below zero
    particle_1.E = E - T - Enl
    if particle_1.E < 0: particle_1.E = 0.

    particle_2.E = T - material.Eb
    if particle_2.E < 0: particle_2.E = 0.

def pick_collision_partner(particle_1, material):
    #Pick mfp and impact parameter from distributions
    mfp = material.mfp(particle_1.pos)
    pmax = mfp/np.sqrt(np.pi)
    impact_parameter = pmax * np.sqrt(np.random.uniform(0., 1.))
    phi_azimuthal = np.random.uniform(0.0, 2.0*np.pi)

    sphi = np.sin(phi_azimuthal)
    ca = particle_1.cosx
    cb = particle_1.cosy
    cg = particle_1.cosz
    sa = np.sin(np.arccos(ca))
    cphi = np.cos(phi_azimuthal)

    #Recoil position displaced by impact parameter 1 mfp ahead of particle
    x_recoil = particle_1.x + mfp*ca - impact_parameter*cphi*sa
    y_recoil = particle_1.y + mfp*cb - impact_parameter*(sphi*cg - cphi*cb*ca)/sa
    z_recoil = particle_1.z + mfp*cg + impact_parameter*(sphi*cb - cphi*ca*cg)/sa

    return impact_parameter, phi_azimuthal, Particle(material.m, material.Z, 0., [ca, cb, cg], [x_recoil, y_recoil, z_recoil])

def surface_boundary_condition(particle_1, material, model='planar'):
    #Must overcome surface energy barrier to leave - planar model
    if model == 'planar':
        leaving_energy = particle_1.E*particle_1.cosx**2
    elif model == 'isotropic':
        leaving_energy = particle_1.E

    if (not material.inside_energy_barrier(particle_1.pos)) and (material.inside_energy_barrier(particle_1.pos_old)):
        if leaving_energy < material.Es:
            point = Point(particle_1.x, particle_1.y)
            nearest, _ = nearest_points(material.energy_barrier_geometry, point)
            dx = nearest.x - point.x
            dy = nearest.y - point.y

            particle_1.cosx *= -1 #Reflect back onto surface if not enough energy to leave
            particle_1.cosy *= -1 #This is a 180deg rotation - not correct!

            particle_1.cosx = 2.*(particle_1.cosx*dx + particle_1.cosy*dy)/(dx**2 + dy**2)*dx - particle_1.cosx
            particle_1.cosy = 2.*(particle_1.cosx*dx + particle_1.cosy*dy)/(dx**2 + dy**2)*dy - particle_1.cosy

            particle_1.x = nearest.x
            particle_1.y = nearest.y

            return False
        else:
            surface_refraction(particle_1, material, model, factor=-1)
            return True

def surface_refraction(particle_1, material, model='planar', factor=1):
    if model == 'planar':
        #See Eckstein Eq. 6.2.4
        #Bends particles towards surface by surface binding energy
        Es = material.Es
        E0 = particle_1.E
        cosx0 = particle_1.cosx
        sign = np.sign(particle_1.cosx)
        sinx0 = np.sin(np.arccos(cosx0))

        particle_1.cosx = np.sqrt((E0*cosx0**2 + sign*Es)/(E0 + sign*Es))
        sinx = np.sin(np.arccos(particle_1.cosx))
        particle_1.cosy *= sinx/sinx0
        particle_1.cosz *= sinx/sinx0
        particle_1.E += sign*material.Es
    else:
        particle_1.E += factor*material.Es

def bca(E0, Ec, N, theta, material, particles, num_print=100, track_recoils=False):
    #Surface refraction as first step - don't use for isotropic potential!
    for particle in particles:
        surface_refraction(particle, material, model='isotropic')

    #Empty arrays for plotting
    estimated_num_recoils = np.int(np.ceil(N*E0/Ec))

    print(f'E: {E0/e} eV')
    #Begin particle loop
    particle_index = 0
    while particle_index < len(particles):

        if particle_index%(len(particles)/num_print): print(f'particle {particle_index} of {len(particles)}')

        particle_1 = particles[particle_index]
        #Begin trajectory loop
        while not (particle_1.stopped or particle_1.left):
            #Check particle stop conditions - reflection/sputtering or stopping

            if not material.inside_simulation_boundary(particle_1.pos):
                particle_1.left = True
                continue #Skip binary collision

            if particle_1.E < Ec:
                particle_1.stopped = True
                continue #Skip binary collision

            #Binary collision step
            impact_parameter, phi_azimuthal, particle_2 = pick_collision_partner(particle_1, material)
            theta, psi, T, t, doca = binary_collision(particle_1, particle_2, material, impact_parameter)
            update_coordinates(particle_1, particle_2, material, phi_azimuthal, theta, psi, T, t)
            surface_boundary_condition(particle_1, material, model='isotropic')

            #Store incident particle trajectories
            particle_1.add_trajectory()

            #Add recoil to particle array
            if T > Ec:
                particle_2.track_trajectories = track_recoils
                particles.append(particle_2)

        particle_index += 1
        print(f'R: {particle_1.pos[0]*1e6} um')

    R = sum([1 if particle.left and particle.incident else 0 for particle in particles])
    S = sum([1 if particle.left and not particle.incident else 0 for particle in particles])
    print(f'reflected: {R} sputtered: {S}')

    return particles, material

def main():
    np.random.seed(1)

    angle = 0.0001
    energy = 1e6

    N = 100

    colors = {
        29: 'black',
        2: 'red',
        74: 'blue',
        1: 'red'
    }

    thickness = 1.0
    depth = 25
    material = Material(8.453e28, 63.54*amu, 29, 3.52*e, depth=depth*1e-6, thickness=thickness*1e-6, use_PSTAR=True, STAR_material=ProtonMaterials.COPPER) #Copper
    particles = [Particle(
        1*amu, 1, energy*e,
        [np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.), 0.0],
        [0.99*material.energy_barrier_position, np.random.uniform(-thickness*1e-6, thickness*1e-6), 0.0],
        incident=True, track_trajectories=True) for _ in range(N)]

    plt.figure(1)
    x, y = material.geometry.exterior.xy
    x = np.array(x)
    y = np.array(y)
    plt.plot(x/angstrom, y/angstrom, color='dimgray', linewidth=3)
    x, y = material.simulation_boundary.exterior.xy
    x = np.array(x)
    y = np.array(y)
    plt.plot(x/angstrom, y/angstrom, '--', color='dimgray')

    particles, material = bca(energy*e, 3.*e, N, angle, material, particles, track_recoils=False)

    for particle_index, particle in enumerate(particles):
        #if particle_index: break
        trajectory = np.array(particle.trajectory).transpose()
        plt.plot(trajectory[0, :]/angstrom, trajectory[1, :]/angstrom, color=colors[particle.Z], linewidth=1)
    plt.axis('square')
    plt.show()
    plt.savefig('starchip.png')
    print('Done!')
    breakpoint()

if __name__ == '__main__':
    main()

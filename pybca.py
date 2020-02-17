import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

#constants
e = 1.602e-19
amu = 1.66e-27
angstrom = 1e-10
eps0 = 8.85e-12
a0 = 0.52918e-10
K = 4.*np.pi*eps0

#screening
phi_coef = np.array([0.191, 0.474, 0.335])
phi_args = np.array([-0.279, -0.637, -1.919])

class Particle:
    def __init__(self, m, Z, E, dir_cos, pos, incident=False):
        self.m = m
        self.Z = Z

        self.E = E
        self.dir_cos = np.array(dir_cos)
        self.pos = np.array(pos)
        self.t = 0.0
        self.incident = incident
        self.stopped = False
        self.left = False
        self.first_step = self.incident

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
    def __init__(self, n, m, Z, Es, Eb=0.0):
        self.n = n
        self.m = m
        self.Z = Z
        self.Z_eff = np.mean(Z)
        #self.mfp = n**(-1./3.)
        self.Eb = Eb
        self.Es = Es
        self.energy_barrier_position = -2.*self.n**(-1./3.)/np.sqrt(2.*np.pi)
        self.surface_position = 0.

    def inside(self, x):
        return x > self.surface_position

    def mfp(self, x):
        return self.n**(-1./3.)

def phi(xi):
    #return 0.191*np.exp(-0.279*xi) + 0.474*np.exp(-0.637*xi) + 0.335*np.exp(-1.919*xi)
    return np.sum(phi_coef*np.exp(phi_args*xi))

def dphi(xi):
    #return -0.191*0.279*np.exp(-0.279*xi) - 0.474*0.637*np.exp(-0.637*xi) - 0.335*1.919*np.exp(-1.919*xi)
    return np.sum(phi_args*phi_coef*np.exp(phi_args*xi))

def screening_length(Za, Zb):
    return 0.8853*a0/(np.sqrt(Za) + np.sqrt(Zb))**(2./3.)

def binary_collision(particle_1, particle_2, material, impact_parameter):

    #If recoil outside surface, skip computation
    if not material.inside(particle_2.x):
        return 0., 0., 0., 0., 0.

    Za = particle_1.Z
    Zb = particle_2.Z
    Ma = particle_1.m
    Mb = particle_2.m
    E0 = particle_1.E
    mu = Mb/(Ma + Mb)

    #Lindhard screening and reduced energy
    a = screening_length(Za, Zb)
    reduced_energy = K*a*mu/(Za*Zb*e**2)*E0

    #See M. H. Mendenhall & R. A. Weller, 1991 and 2005
    beta = impact_parameter/a
    doca_function = lambda x0: x0 - phi(x0)/reduced_energy - beta**2/x0
    f = lambda x: (1 - phi(x)/x/reduced_energy - (beta/x)**2)**(-1./2.)
    x0 = newton(doca_function, x0=1, tol=1e-3, maxiter=100)
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
    mfp = material.mfp(particle_1.x)

    if particle_1.first_step:
        mfp *= np.random.uniform(0., 1.) #In TRIDYN, this is the "atomically rough" surface
        particle_1.first_step = False

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
    Zb = material.Z_eff
    E = particle_1.E
    a = screening_length(Za, Zb) #Z_eff?
    Sel = 1.212*(Za**(7./6.)*Zb)/((Za**(2./3.) + Zb**(2./3.))**(3./2.))*np.sqrt(E/Ma*amu/e)
    stopping_factor = material.n*Sel*angstrom**2*e
    Enl = free_flight_path*stopping_factor

    #No electronic stopping out of material
    if not material.inside(particle_1.x): Enl = 0.

    #Energy calculation - make sure stopping doesn't reduce energy below zero
    particle_1.E = E - T - Enl
    if particle_1.E < 0: particle_1.E = 0.

    particle_2.E = T - material.Eb
    if particle_2.E < 0: particle_2.E = 0.

    #Did particle leave material? Check surface binding energy and reflect
    #TODO: Surface Refraction
    if particle_1.x < material.energy_barrier_position and particle_1.cosx < 0.0:

        if particle_1.E*np.abs(particle_1.cosx) < material.Es:
            particle_1.cosx *= -1
            particle_1.x = material.energy_barrier_position

        else:
            particle_1.E - material.Es*particle_1.cosx
            particle_1.left = True

def pick_collision_partner(particle_1, material):
    mfp = material.mfp(particle_1.x)
    pmax = mfp/np.sqrt(np.pi)
    impact_parameter = pmax * np.sqrt(np.random.uniform(0., 1.))
    phi_azimuthal = np.random.uniform(0.0, 2.0*np.pi)
    sphi = np.sin(phi_azimuthal)
    ca = particle_1.cosx
    cb = particle_1.cosy
    cg = particle_1.cosz
    sa = np.sin(np.arccos(ca))
    cphi = np.cos(phi_azimuthal)

    x_recoil = particle_1.x + mfp*ca - impact_parameter*cphi*sa
    y_recoil = particle_1.y + mfp*cb - impact_parameter*(sphi*cg - cphi*cb*ca)/sa
    z_recoil = particle_1.z + mfp*cg + impact_parameter*(sphi*cb - cphi*ca*cg)/sa

    return impact_parameter, phi_azimuthal, Particle(material.m, material.Z, 0.0, [ca, cb, cg], [x_recoil, y_recoil, z_recoil])

def main():
    np.random.seed(1)

    E0 = 1e4*e
    Ec = 3*e
    N = 20
    alpha = 30

    material = Material(8.453e28, 63.54*amu, 29, 0.0) #Copper
    particles = [Particle( 1*amu, 1, E0, [np.cos(alpha*np.pi/180.), np.sin(alpha*np.pi/180.), 0.0], [material.energy_barrier_position, 0.0, 0.0], incident=True) for _ in range(N)]

    trajectories = np.zeros((3, np.int(np.ceil(N*E0/Ec))))
    trajectory_index = 0

    x_final = np.zeros(N)

    particle_index = 0
    while particle_index < len(particles):
        if particle_index%10 == 0: print(f'{np.round(particle_index / len(particles) * 100, 1)}%')

        particle = particles[particle_index]
        while not (particle.stopped or particle.left):

            impact_parameter, phi_azimuthal, particle_2 = pick_collision_partner(particle, material)
            theta, psi, T, t, doca = binary_collision(particle, particle_2, material, impact_parameter)
            update_coordinates(particle, particle_2, material, phi_azimuthal, theta, psi, T, t)

            if particle.x < material.energy_barrier_position and particle.cosx < 0.:
                particle.left = True

            if particle.x > material.surface_position and particle.E < Ec:
                particle.stopped = True

            #Store incident particle trajectories
            if particle_index < N:
                trajectories[:, trajectory_index] = particle.pos
                trajectory_index += 1

            if T > Ec:
                particles.append(particle_2)

        if particle_index < N and particle.x > 0.0:
            x_final[particle_index] = particle.x

        particle_index += 1

    plt.figure(1)
    plt.scatter(trajectories[0, :trajectory_index]/angstrom, trajectories[1, :trajectory_index]/angstrom, color='black', s=1)
    plt.scatter(material.energy_barrier_position/angstrom, 0., color='red', marker='+')
    plt.axis('square')

    plt.figure(2)
    plt.hist(x_final/angstrom, bins=20)
    print(f'R: {np.mean(x_final/angstrom)} sR: {np.std(x_final/angstrom)}')
    R = sum([1 if particle.left and particle.incident else 0 for particle in particles])
    S = sum([1 if particle.left and not particle.incident else 0 for particle in particles])
    print(f'reflected: {R} sputtered: {S}')
    plt.show()

if __name__ == '__main__':
    main()

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

#MAGIC
C1 = 1.0144
C2 = 0.2358
C3 = 0.1260
C4 = 63950
C5 = 83550

class Particle:
    def __init__(self, m, Z, E, dir_cos, pos):
        self.m = m
        self.Z = Z

        self.E = E
        self.dir_cos = np.array(dir_cos)
        self.pos = np.array(pos)
        self.t = 0.0

class Material:
    def __init__(self, n, m, Z, ES, BE=0.0):
        self.n = n
        self.m = m
        self.Z = Z
        self.Z_eff = np.mean(Z)
        self.mfp = n**(-1./3.)
        self.BE = BE
        self.ES = ES

def phi(xi):
    return 0.191*np.exp(-0.279*xi) + 0.474*np.exp(-0.637*xi) + 0.335*np.exp(-1.919*xi)

def binary_collision(particle_1, particle_2, impact_parameter):
    #calculated parameters
    Za = particle_1.Z
    Zb = particle_2.Z
    Ma = particle_1.m
    Mb = particle_2.m
    E0 = particle_1.E
    mu = Mb/(Ma + Mb)
    a = 0.8853*a0/(np.sqrt(Za) + np.sqrt(Zb))**(2/3)
    b = impact_parameter/a
    reduced_energy = K*a*mu/(Za*Zb*e**2)*E0
    sqer = np.sqrt(reduced_energy)
    com_energy = E0*mu

    #potential functions
    potential_depth = Za*Zb*e**2/K
    potential_energy = lambda r: potential_depth*(1./r)*phi(r/a)
    diff_potential_energy = lambda r: potential_depth*np.sum(phi_coef*np.exp(-phi_args*r/a)*(a+phi_args*r)/(a*r**2))

    #distance of closest approach
    doca_function = lambda xi: b**2/xi + phi(xi)/reduced_energy - xi
    if reduced_energy == 0:
        breakpoint()
    try:
        xic = newton(doca_function, x0=0.1, tol=1e-3, maxiter=100)
    except RuntimeError:
        breakpoint()

    distance_of_closest_approach = xic*a

    #MAGIC algorithm
    rho = -2.*(com_energy - potential_energy(distance_of_closest_approach))/(diff_potential_energy(distance_of_closest_approach))/a
    V = phi(xic)
    A = 2.0*(1.0+C1/sqer)*reduced_energy*b**((C2+sqer)/(C3+sqer))
    F = (C5 + reduced_energy)/(C4 + reduced_energy)*(np.sqrt(1 + A**2) - A)
    Delta = A*F/(1. + F)*(xic - b)
    theta = 2.*np.arccos((b + rho + Delta)/(xic + rho))

    t = impact_parameter*np.tan(theta/2.)
    psi = np.arctan2(np.sin(theta), (Ma/Mb) + np.cos(theta))
    T = 4.*(Ma*Mb)/(Ma + Mb)**2*E0*(np.sin(theta/2.))**2
    return theta, psi, T, t

def update_coordinates(particle_1, particle_2, material, phi_azimuthal, theta, psi, T, t):
    #update position of moving particle

    pos_old = particle_1.pos
    particle_1.pos[:] = particle_1.pos + (material.mfp - t + particle_1.t)*particle_1.dir_cos
    free_flight_path = (material.mfp - t + particle_1.t)
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
    particle_1.dir_cos[:] = [ca_new, cb_new, cg_new]

    #update angular coordinates of secondary particle
    psi_b = np.arctan2(-np.sin(theta), 1. - np.cos(theta))
    cpsi_b = np.cos(psi_b)
    spsi_b = np.sin(psi_b)
    ca_new = cpsi_b*ca + spsi_b*cphi*sa
    cb_new = cpsi_b*cb - spsi_b/sa*(cphi*ca*cb - sphi*cg)
    cg_new = cpsi_b*cg - spsi_b/sa*(cphi*ca*cg + sphi*cb)
    particle_2.dir_cos[:] = [ca_new, cb_new, cg_new]

    #update energy coordinates
    Za = particle_1.Z
    Ma = particle_1.m
    Zb = material.Z_eff
    E = particle_1.E
    a = 0.8853*a0/(np.sqrt(Za) + np.sqrt(particle_2.Z))**(2/3)
    Sel = 1.212*(Za**(7./6.)*Zb)/((Za**(2./3.) + Zb**(2./3.))**(3./2.))*np.sqrt(E/Ma*amu/e)
    stopping_factor = material.n*Sel*angstrom**2*e
    Enl = free_flight_path*stopping_factor

    #No electronic stopping out of material
    if particle_1.pos[0] < 0.0:
        Enl = 0.0

    #Energy mathematics - make sure stopping doesn't reduce energy below zero
    if E - T - Enl > 0:
        particle_1.E = E - T - Enl
    else:
        particle_1.E = 0

    if T - material.BE > 0:
        particle_2.E = T - material.BE
    else:
        particle_2.E = 0

    #Did particle leave material? Check surface binding energy and reflect
    if particle_1.pos[0] < -material.mfp and pos_old[0] > -material.mfp:
        print('Huh')
        if particle_1.E * particle_1*particle_1.dir_cos[0] > material.ES:
            particle_1.dir_cos[0] *= -1
            print('Reflected!')
        else:
            particle_1.E - material.ES
            print('Sputtered!')

def pick_collision_partner(particle_1, material):
    mfp = material.n**(-1./3.)
    impact_parameter = np.sqrt(np.random.uniform(0.0, 1.0)) * mfp / np.sqrt(np.pi)
    phi_azimuthal = np.random.uniform(0.0, 2.0*np.pi)
    sphi = np.sin(phi_azimuthal)
    ca = particle_1.dir_cos[0]
    cb = particle_1.dir_cos[1]
    cg = particle_1.dir_cos[2]
    sa = np.sin(np.arccos(ca))
    cphi = np.cos(phi_azimuthal)

    x_recoil = particle_1.pos[0] + mfp*ca - impact_parameter*cphi*sa
    y_recoil = particle_1.pos[1] + mfp*cb - impact_parameter*(sphi*cg - cphi*cb*ca)/sa
    z_recoil = particle_1.pos[2] + mfp*cg + impact_parameter*(sphi*cb - cphi*ca*cg)/sa

    return impact_parameter, phi_azimuthal, Particle(material.m, material.Z, 0.0, [ca, cb, cg], [x_recoil, y_recoil, z_recoil])

def main():
    np.random.seed(5)
    E0 = 1e4*e
    EC = 3*e
    N = 100
    material = Material(8.491e28, 64.*amu, 29, 3.49*e) #Copper
    energy_barrier_position = -material.mfp
    alpha = 0.001
    particle_1 = Particle(64.*amu, 29, E0, [np.cos(alpha), np.sin(alpha), 0.0], [energy_barrier_position, 0.0, 0.0])

    particles = [Particle(64.*amu, 29, E0, [np.cos(alpha), np.sin(alpha), 0.0], [energy_barrier_position, 0.0, 0.0]) for _ in range(N)]

    x, y, z, E = [energy_barrier_position], [0.0], [0.0], [E0]
    particle_index = 0

    while particle_index < len(particles):


        while particles[particle_index].E > EC and particles[particle_index].pos[0] >= energy_barrier_position:

            impact_parameter, phi_azimuthal, particle_2 = pick_collision_partner(particles[particle_index], material)

            if particle_2.pos[0] > 0.0:
                theta, psi, T, t = binary_collision(particles[particle_index], particle_2, impact_parameter)
                if particle_index == 0:
                    print(theta*180/np.pi, psi, T, t)
            else:
                theta, psi, T, t = 0.0, 0.0, 0.0, 0.0

            update_coordinates(particles[particle_index], particle_2, material, phi_azimuthal, theta, psi, T, t)
            if particle_index < N:
                print(particles[particle_index].E/e)
                E.append(particles[particle_index].E)
                x.append(particles[particle_index].pos[0])
                y.append(particles[particle_index].pos[1])
                z.append(particles[particle_index].pos[2])

            if T > material.BE:
                particles.append(particle_2)
        particle_index += 1

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    E = np.array(E)

    plt.figure(1)
    plt.scatter(x/angstrom, z/angstrom)

    plt.figure(2)
    plt.plot(E/e)
    plt.show()


if __name__ == '__main__':
    main()

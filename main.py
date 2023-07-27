import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


G = 6.6743E-11
AU = 1.496E11
M_sol = 2E30
M_earth = 6E24
day = 86400

dt = 3600.  # seconds
vec = np.array([0, 0, 0], dtype=float)


class Body:
    def __init__(self, mass=M_sol, pos=np.copy(vec), vel=np.copy(vec), acc=np.copy(vec)):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.positions = []

    def acceleration(self, bodies: list):
        acc = np.copy(vec)
        for body in bodies:
            if body == self:
                continue
            r = body.pos - self.pos
            r_hat = r / np.linalg.norm(r)
            r2 = np.sum(np.power(r, 2))
            acc += G * body.mass * r_hat / r2
        self.acc = acc

    def verlet_pos(self, t=dt):
        self.pos += self.vel * t + 0.5 * self.acc * t * t
        self.positions.append(np.copy(self.pos))

    def verlet_vel(self, t=dt):
        self.vel += self.acc * t


body1 = Body(0.99*M_sol, np.array([0.501*AU, 0, 0]), np.array([0, 20.86E3, 0]))
body2 = Body(0.99*M_sol, np.array([-0.501*AU, 0, 0]), np.array([0, -20.86E3, 0]))
body3 = Body(0.2*M_sol, np.array([-0.25*AU, 2*AU, 0.]), np.array([5E3, -10E3, 0]))

sun = Body(M_sol, np.array([0., 0., 0.]), np.array([0., 0., 0.]))
mercury = Body(0.055*M_earth, np.array([0.387*AU, 0, 0]), np.array([0, 47360., 0]))
venus = Body(0.815*M_earth, np.array([0.723*AU, 0, 0]), np.array([0, 35020., 0]))
earth = Body(M_earth, np.array([AU, 0, 0]), np.array([0, 29785., 0]))
mars = Body(0.107*M_earth, np.array([227.94E9, 0, 0]), np.array([0, 24070., 0]))
jupiter = Body(317.8*M_earth, np.array([778.479E9, 0, 0]), np.array([0, 13070., 0]))
saturn = Body(95.159*M_earth, np.array([9.5826*AU, 0, 0]), np.array([0, 9689.3, 0]))
uranus = Body(14.536*M_earth, np.array([19.19126*AU, 0, 0]), np.array([0, 6803.3, 0]))
neptune = Body(17.147*M_earth, np.array([30.07*AU, 0, 0]), np.array([0, 5436.5, 0]))

n_bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]


for i in range(60000*int(day/dt)):
    for n_body in n_bodies:
        n_body.verlet_pos()

    for n_body in n_bodies:
        n_body.verlet_vel()

    for n_body in n_bodies:
        n_body.acceleration(n_bodies)


for n_body in n_bodies:
    plt.scatter([p[0] for p in n_body.positions[::int(day/dt)]], [p[1] for p in n_body.positions[::int(day/dt)]])
plt.show()

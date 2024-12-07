import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


G = 6.6743E-11
AU = 1.496E11
M_sol = 2E30
M_earth = 6E24
day = 86400

dt = 60  # seconds
vec = np.array([0, 0, 0], dtype=float)


class Body:
    def __init__(self, mass=M_sol, pos=np.copy(vec), vel=np.copy(vec), acc=np.copy(vec)):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc_old = 0
        self.acc = acc
        self.positions = []
        self.L = []

    def angular_momentum(self, bodies: list):
        com = np.sum([body.pos * body.mass for body in bodies], axis=0) / np.sum([body.mass for body in bodies])
        self.L.append(self.mass * np.linalg.norm(np.cross(self.vel, self.pos - com)))

    def acceleration(self, bodies: list):
        self.acc_old = self.acc
        acc = np.copy(vec)
        for body in bodies:
            if body == self:
                continue
            r = body.pos - self.pos
            r_hat = r / np.linalg.norm(r)
            r2 = np.sum(np.power(r, 2))
            acc += G * body.mass * r_hat / r2
        self.acc = acc
        return self.acc

    def verlet_pos(self, t=dt):
        self.pos += self.vel * t
        self.positions.append(np.copy(self.pos))

    def verlet_vel(self, bodies: list, t=dt):
        self.vel += self.acceleration(bodies) * t


body1 = Body(0.99*M_sol, np.array([0.501*AU, 0, 0]), np.array([0, 20.86E3, 0]))
body2 = Body(0.99*M_sol, np.array([-0.501*AU, 0, 0]), np.array([0, -20.86E3, 0]))
body3 = Body(0.2*M_sol, np.array([-0.25*AU, 2*AU, 0.]), np.array([5E3, -10E3, 0]))

body4 = Body(1, np.array([1, 0, 0.]), np.array([0, 0.9, 0]))
body5 = Body(1, np.array([0, 0, 0.]), np.array([0, 0.0, 0]))

sun = Body(M_sol, np.array([0., 0., 0.]), np.array([0., 0., 0.]))
mercury = Body(0.055*M_earth, np.array([0.387*AU, 0, 0]), np.array([0, 47360., 0]))
venus = Body(0.815*M_earth, np.array([0.723*AU, 0, 0]), np.array([0, 35020., 0]))
earth = Body(M_earth, np.array([AU, 0, 0]), np.array([0, 29785., 0]))
mars = Body(0.107*M_earth, np.array([227.94E9, 0, 0]), np.array([0, 24070., 0]))
jupiter = Body(317.8*M_earth, np.array([778.479E9, 0, 0]), np.array([0, 13070., 0]))
saturn = Body(95.159*M_earth, np.array([9.5826*AU, 0, 0]), np.array([0, 9689.3, 0]))
uranus = Body(14.536*M_earth, np.array([19.19126*AU, 0, 0]), np.array([0, 6803.3, 0]))
neptune = Body(17.147*M_earth, np.array([30.07*AU, 0, 0]), np.array([0, 5436.5, 0]))

n_bodies = [body4, body5]

# for body in n_bodies:
#     body.acceleration(n_bodies)


for i in range(10*int(day/dt)):
    for body in n_bodies:
        body.angular_momentum(n_bodies)

    for body in n_bodies:
        body.verlet_vel(n_bodies, t=dt/2)

    for body in n_bodies:
        body.verlet_pos(t=dt)

    for body in n_bodies:
        body.verlet_vel(n_bodies, t=dt/2)


for body in n_bodies:
    plt.scatter([p[0] for p in body.positions[::int(day/dt)]], [p[1] for p in body.positions[::int(day/dt)]])
plt.show()

plt.cla()

# for body in n_bodies:
#     plt.plot([i for i in range(len(body.L))], [body.L[i] for i in range(len(body.L))])
# plt.show()

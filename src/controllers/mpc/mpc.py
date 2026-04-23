import casadi as ca
import numpy as np


class MPC:

    def __init__(self, x_d, u_d, dynamics_func, term_constraint_func, constraint_func, ipopt_settings, step=0.02*2, n=20):
        """ 
        
        """
        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.constraint = constraint_func
        self.term_constraint = term_constraint_func

        self.opti = ca.Opti()
        
        self.x0 = self.opti.parameter(x_d)  # Params for running MPC opt
        self.x_d = x_d
        self.u_d = u_d
        self.n = n

        self.step = step
        self.dynamics = dynamics_func()  # ca.Function for efficiency
        self.u_sym = self.setup_mpc()

        self.opti.solver("ipopt", ipopt_settings)


    def euler_step(self, x, u):
        return x + self.dynamics(x, u) * self.step

    def setup_mpc(self):
        x = self.x0
        u = self.opti.variable(self.n, self.u_d)
        J = 0

        for i in range(self.n):
            x = self.euler_step(x, u[i])

            J += x[2] ** 2 + x[0] ** 2
            self.constraint(self.opti, x, u[i])

        self.opti.minimize(J)

        if self.term_constraint:
            self.term_constraint(self.opti, x, u[0], u[-1])

        return u

    def run_mpc(self, x, verbose=True, warm_start=True):

        if self.last_trajectory is not None and warm_start:
            self.opti.set_initial(self.u_sym,  ca.vertcat(self.last_trajectory[:-1], self.last_trajectory[-1]))
        
        self.opti.set_value(self.x0, x)

        sol = self.opti.solve()
        
        if verbose:
            stats = sol.stats()
            print(f"Solve iteration succeeded in {stats['iter_count']} iterations in {stats['t_wall_total']} s")

        u = sol.value(self.u_sym)[0]

        if warm_start:
            self.last_trajectory = sol.value(self.u_sym)
        
        return u


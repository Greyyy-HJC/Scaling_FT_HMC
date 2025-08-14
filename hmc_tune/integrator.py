# %%
import torch
import math
import numpy as np

def plaq_from_field(theta):
    """
    Calculate the plaquette value for a given field configuration.
    """
    theta0, theta1 = theta[0], theta[1]
    thetaP = theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0)

    return thetaP


def regularize(theta):
    """
    Regularize the plaquette value to the range [-pi, pi).
    """
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - torch.floor(theta_wrapped) - 0.5)



class HMC_U1:
    def __init__(
        self,
        lattice_size,
        beta,
        n_steps,
        step_size,
        device="cpu",
    ):
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_steps = n_steps
        self.dt = step_size
        self.device = torch.device(device)

        # Set default data type and device
        torch.set_default_dtype(torch.float64)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def action(self, theta):
        theta_P = plaq_from_field(theta)
        thetaP_wrapped = regularize(theta_P)
        action_value = (-self.beta) * torch.sum(torch.cos(thetaP_wrapped))
        
        assert action_value.dim() == 0, "Action value is not a scalar."
        
        return action_value

    def force(self, theta):
        theta_copy = theta.detach().clone().requires_grad_(True)
        action_value = self.action(theta_copy)
        force = torch.autograd.grad(action_value, theta_copy)[0]
        return force.detach()  # *: break the gradient chain

    def omelyan(self, theta, pi):
        """
        Use 2nd-order minimum-norm (Omelyan) integrator.
        """
        lam = 0.1931833 # the best λ for 2nd-order minimum-norm (Omelyan) integrator #? NOTE: think of tuning this parameter
        #TODO: ~ 0.21 for 4D SU(3)
        dt = self.dt
        theta_ = theta
        pi_ = pi - lam * dt * self.force(theta_)
        for step_index in range(self.n_steps):
            theta_ = theta_ + 0.5 * dt * pi_
            pi_ = pi_ - (1 - 2 * lam) * dt * self.force(theta_)
            theta_ = theta_ + 0.5 * dt * pi_
            # Interior momentum update that merges the trailing P(lam*dt) of the
            # current substep with the leading P(lam*dt) of the next substep.
            if step_index != self.n_steps - 1:
                pi_ = pi_ - 2 * lam * dt * self.force(theta_)
        pi_ = pi_ - lam * dt * self.force(theta_)
        theta_ = regularize(theta_)
        return theta_, pi_
    
    def leapfrog(self, theta, pi):
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        theta_ = regularize(theta_)
        return theta_, pi_

    def metropolis_step_leapfrog(self, theta):
        pi = torch.randn_like(theta, device=self.device)
        action_value = self.action(theta)
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        new_action_value = self.action(new_theta)
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        
        dH = delta_H.detach().cpu().numpy().item()
        return dH
    
    def metropolis_step_omelyan(self, theta):
        pi = torch.randn_like(theta, device=self.device)
        action_value = self.action(theta)
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.omelyan(theta.clone(), pi.clone())
        new_action_value = self.action(new_theta)
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)
        
        delta_H = H_new - H_old
        
        dH = delta_H.detach().cpu().numpy().item()
        return dH
    
#TODO: add test with FT, FT after training
#TODO: check the integrator



# theta = initial field
# pi = initial momentum
# theta_new, pi_new = omelyan(theta, pi)
# theta_reversed, pi_reversed = omelyan(theta_new, - pi_new)





# %%
import numpy as np
def compute_dH_leapfrog(dt, steps, n_trials=32):
    lattice_size = 32
    beta = 3.0
    n_steps = steps
    step_size = dt
    hmc = HMC_U1(lattice_size, beta, n_steps, step_size)
    
    dHs = []
    for _ in range(n_trials):
        x = torch.zeros([2, lattice_size, lattice_size])
        dH = hmc.metropolis_step_leapfrog(x)
        dHs.append(dH)
    return np.mean(dHs), np.std(dHs)

def compute_dH_omelyan(dt, steps, n_trials=32):
    lattice_size = 32
    beta = 3.0
    n_steps = steps
    step_size = dt
    hmc = HMC_U1(lattice_size, beta, n_steps, step_size)
    
    dHs = []
    for _ in range(n_trials):
        x = torch.zeros([2, lattice_size, lattice_size])
        dH = hmc.metropolis_step_omelyan(x)
        dHs.append(dH)
    return np.mean(dHs), np.std(dHs)

def test_reversed_omelyan(dt, steps, n_trials=8):
    lattice_size = 32
    beta = 3.0
    n_steps = steps
    step_size = dt
    hmc = HMC_U1(lattice_size, beta, n_steps, step_size)
    
    for _ in range(n_trials):
        theta = torch.zeros([2, lattice_size, lattice_size])
        pi = torch.randn_like(theta, device=hmc.device)
        theta_new, pi_new = hmc.omelyan(theta.clone(), pi.clone())
        theta_reversed, pi_reversed = hmc.omelyan(theta_new.clone(), -pi_new.clone())
        
        diff_theta = theta_reversed - theta
        diff_pi = pi_reversed + pi
        
        print(f"diff_theta: {diff_theta.norm()}")
        print(f"diff_pi: {diff_pi.norm()}")


steps_list = [5, 10, 20, 40]
tau_leapfrog = 0.1
tau_omelyan = 0.1

print('-' * 60)

for steps in steps_list:
    dt = tau_leapfrog / steps
    mean_dH, std_dH = compute_dH_leapfrog(dt, steps)
    print(f"Leapfrog: steps={steps:2d}, dt={dt:.5f}, ⟨ΔH⟩={mean_dH:.5e}, std={std_dH:.2e}")
    
print('-' * 60)

for steps in steps_list:
    dt = tau_omelyan / steps
    mean_dH, std_dH = compute_dH_omelyan(dt, steps)
    print(f"Omelyan: steps={steps:2d}, dt={dt:.5f}, ⟨ΔH⟩={mean_dH:.5e}, std={std_dH:.2e}")
    
print('-' * 60)

for steps in steps_list:
    dt = tau_omelyan / steps
    test_reversed_omelyan(dt, steps)


# %%
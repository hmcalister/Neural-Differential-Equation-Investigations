# This example comes from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
# I have adapted it slightly and annotated each part for my own understanding and as a jumping off point for NDE exploration

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

TORCH_DEVICE = "cuda"

DATA_STEPS = 1000   # The number of time steps to simulate when creating the data. In practice this is defined by collection of data, not simulation.

USE_ADJOINT = True # Defines how to solve the ODE. Adjoint is typically more stable.
BATCH_SIZE = 20     # The number of samples to take in each batch. Defines the number of time intervals that are sampled for each batch of learning.
BATCH_TIME = 10     # Defines the length of each time interval for learning.
NUM_EPOCHS = 2000   # Number of epochs to train over
TEST_FREQUENCY = 20 # How many epochs between testing the network during training (mainly for visualization)
HIDDEN_LAYER_SIZE = 50  # Defines the (very simple) neural network architecture

if USE_ADJOINT:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# --------------------------------------------------------------------------------
# SET UP AND DIFFERENTIAL EQUATION DEF

true_y0 = torch.tensor([[2., 0.]]).to(TORCH_DEVICE) # Define the true initial condition for the simulation. In practice, defined by data collection.
t = torch.linspace(0., 25., DATA_STEPS).to(TORCH_DEVICE)    # Define the time steps starting from the true_y0
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(TORCH_DEVICE) # Define the true matrix determining the differential equation. THIS IS WHAT WE ARE MODELLING!!!

# Define the differential equation of the system as a nn.Module for future integration in torch.
# 
# Interpret Lambda as dy/dt = f(t ,y) such that forward implements f(t, y)
class Lambda(nn.Module):
    
    def forward(self, t, y):
        return torch.mm(y**3, true_A)
    
# --------------------------------------------------------------------------------
# SIMULATE DATA

# EULER METHOD OF ITERATION. Sanity check for finding next values.
# with torch.no_grad():
#     f = Lambda()
#     iterated_y_val = [true_y0]
#     for i in range(DATA_STEPS-1):
#         t_i = t[i]
#         t_f = t[i+1]
#         y_i = iterated_y_val[i]
#         y_f = y_i + f.forward(t_i, y_i) * (t_f-t_i)
#         iterated_y_val.append(y_f)

# Simulate the true y values. In practice this would be collected data.
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method="dopri5")

# --------------------------------------------------------------------------------
# NEURAL NETWORK SETUP

# Get a batch of data to train the neural network on
def get_batch():
    # First get a bunch of random start time indices. Note that we allow enough room for a full batch (defined by BATCH_TIME) at each choice.
    # s.shape = (BATCH_SIZE).
    s = torch.from_numpy(np.random.choice(np.arange(DATA_STEPS - BATCH_TIME, dtype=np.int64), BATCH_SIZE, replace=False))
    
    # Define the initial conditions for each start time choice.
    # batch_y0.shape = (BATCH_SIZE, *dimension)
    batch_y0 = true_y[s]

    # Get the time for the batch. Note that we use the same time frame for each initial condition, 
    # i.e. we are reframing the problem to start from t = 0. This is fine, just a shift of the time variable (?)
    # batch_t.shape = (BATCH_TIME)
    batch_t = t[:BATCH_TIME]

    # Get the true value of y for the batch. Collect the right intervals from true_y based on start time and BATCH_TIME.
    # batch_y.shape = (BATCH_TIME, BATCH_SIZE, *dimension)
    batch_y = torch.stack([true_y[s + i] for i in range(BATCH_TIME)], dim=0)  # (T, M, D)
    return batch_y0.to(TORCH_DEVICE), batch_t.to(TORCH_DEVICE), batch_y.to(TORCH_DEVICE)

# Neural network that will learn the equation and dynamics
class ODEFunc(nn.Module):

    # Set up architecture just like any "normal" network
    def __init__(self):
        super(ODEFunc, self).__init__()

        # Input is defined by the number of measurements fed in 
        # e.g. [y**1, y**2, y**3, sin(y)] would be (4, *dimension) for y.shape = dimension
        # 
        # Output is simply y.shape = dimension
        # 
        # Hidden layer (may be many, but only one is required, and easy to interpret) can be small
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN_LAYER_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_SIZE, 2),
        )

        # Initialization... is this useful? Required?
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    # Define forward based on the differential equation. If we *know* dy/dt only depends on the cube, this is fine... but how robust?
    def forward(self, t, y):
        return self.net(y**3)

# --------------------------------------------------------------------------------
# Training and visualization of Neural Network output

if __name__ == '__main__':
    func = ODEFunc().to(TORCH_DEVICE)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3) # Strange Optimizer... any reason for this choice?

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(TORCH_DEVICE) # See that we solve the ODE in the same way as simulation... but now use NN not simulation.
        loss = torch.mean(torch.abs(pred_y - batch_y)) # Loss is MAE? What about MSE?
        loss.backward()
        optimizer.step()

        if epoch % TEST_FREQUENCY == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print(f"Epoch {epoch:04d} | Total Loss {loss.item():.6f}")

    with torch.no_grad():
        pred_A = func.forward(0, torch.eye(2).to(TORCH_DEVICE))
    print(f"TRUE A:\n{true_A}")
    print(f"PRED A:\n{pred_A}")

    # The following visualization is specific to the given example, and would be hard to generalize
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', label="True")
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--', "Pred")
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)

    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    dydt = func(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(TORCH_DEVICE)).cpu().detach().numpy()
    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()
    plt.show()

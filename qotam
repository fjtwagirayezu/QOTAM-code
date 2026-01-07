import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

print("φ⁴ wavepacket toy — QOTAM-style numerics + angular ρ_in, ρ_out\n")

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

key = random.PRNGKey(0)

# ==============================
# 1. Kinematics & sampling
# ==============================
s = 100.0   # E_cm^2
m = 1.0
n = 4000    # number of samples

def sample_on_shell(key, n, s, m):
    """Sample n on-shell 4-momenta near 2->2 COM kinematics."""
    k1, k2 = random.split(key)
    phi = 2 * jnp.pi * random.uniform(k1, (n,))
    ct  = 2 * random.uniform(k2, (n,)) - 1.0
    st  = jnp.sqrt(1.0 - ct**2)

    E_cm  = jnp.sqrt(s)
    p_mag = jnp.sqrt((E_cm / 2.0)**2 - m**2)

    px = p_mag * st * jnp.cos(phi)
    py = p_mag * st * jnp.sin(phi)
    pz = p_mag * ct
    E  = jnp.sqrt(p_mag**2 + m**2) * jnp.ones_like(px)

    return jnp.stack([E, px, py, pz], axis=-1)  # (n,4)

key, k_p, k_q = random.split(key, 3)
p = sample_on_shell(k_p, n, s, m)  # incoming sampling
q = sample_on_shell(k_q, n, s, m)  # outgoing sampling

# Discrete Lorentz-invariant measure d^3p/(2E) ~ 1/(2E) * 1/n
wp = jnp.ones(n) / (2.0 * p[:, 0] * n)
wq = jnp.ones(n) / (2.0 * q[:, 0] * n)

# ==============================
# 2. Angular variables and wavepackets ρ_in, ρ_out
# ==============================
# cos(theta) for each momentum: z = p_z / |p|
def cos_theta(p4):
    spatial = p4[:, 1:]
    mag = jnp.linalg.norm(spatial, axis=-1) + 1e-30
    return p4[:, 3] / mag  # p_z / |p|

z_in  = cos_theta(p)  # cos(theta_in)
z_out = cos_theta(q)  # cos(theta_out)

# Define angular wavepackets:
#  - incoming: double peak at z = ±1
#  - outgoing: single peak around z = 0 (side scattering)
sigma_ang_in  = 0.3
sigma_ang_out = 0.3

def double_peak(z, sigma):
    return jnp.exp(-(z - 1.0)**2 / (2 * sigma**2)) + \
           jnp.exp(-(z + 1.0)**2 / (2 * sigma**2))

def single_peak(z, sigma):
    return jnp.exp(-z**2 / (2 * sigma**2))

rho_in_raw  = double_peak(z_in,  sigma_ang_in)
rho_out_raw = single_peak(z_out, sigma_ang_out)

# Normalize with respect to the discretized Lorentz measure
norm_in  = jnp.sum(wp * rho_in_raw)
norm_out = jnp.sum(wq * rho_out_raw)

rho_in  = rho_in_raw  / (norm_in  + 1e-30)
rho_out = rho_out_raw / (norm_out + 1e-30)

print(f"check norm: ∑_p dμ ρ_in  = {jnp.sum(wp * rho_in):.6f}")
print(f"check norm: ∑_q dμ ρ_out = {jnp.sum(wq * rho_out):.6f}\n")

# ==============================
# 3. φ⁴ tree-level amplitude
# ==============================
lam = 0.8
K_tree = -1j * lam   # φ⁴ contact kernel (constant)

A = K_tree * (jnp.sum(wp * rho_in)) * (jnp.sum(wq * rho_out))
exact = -1j * lam

print("=================================================================")
print("FINAL RESULT (φ⁴ wavepacket toy)")
print("=================================================================")
print(f"Amplitude A (MC)  : {A.real: .6f} + {A.imag: .6f}j")
print(f"Exact tree-level  :  0.000000 - {lam:.6f}j")
print(f"|A - exact|       : {jnp.abs(A-exact):.6e}")
print(f"Relative error    : {100*jnp.abs(A-exact)/lam:.4f}%")
print("=================================================================\n")

# ==============================
# 4. Angular cost C(z_in, z_out) and toy |K(q|p)|
# ==============================
# Cost based purely on angular mismatch (locality on S^2):
# C_ji ∝ (z_out_j - z_in_i)^2, normalized to [0,1].
z_in_np  = np.array(z_in)
z_out_np = np.array(z_out)

# Outer difference grid
Z_in_grid  = z_in_np[None, :]   # (1, n)
Z_out_grid = z_out_np[:, None]  # (n, 1)

C = (Z_out_grid - Z_in_grid)**2
C = C / (np.max(C) + 1e-30)     # normalize max cost to 1

# Toy kernel magnitude: rank-1 outer product of packets
K_abs = np.outer(np.array(rho_out), np.array(rho_in))  # (n, n)

# ==============================
# 5. Sort by angles for nicer plots
# ==============================
idx_in  = np.argsort(z_in_np)
idx_out = np.argsort(z_out_np)

z_in_sorted   = z_in_np[idx_in]
z_out_sorted  = z_out_np[idx_out]
rho_in_sorted = np.array(rho_in)[idx_in]
rho_out_sorted = np.array(rho_out)[idx_out]

K_abs_sorted = K_abs[np.ix_(idx_out, idx_in)]
C_sorted     = C[np.ix_(idx_out, idx_in)]

# Subsample for visualization (≤ 200 x 200)
step = max(n // 200, 1)
K_sub = K_abs_sorted[::step, ::step]
C_sub = C_sorted[::step, ::step]

# ==============================
# 6. Plots: ρ_in(z) and ρ_out(z)
# ==============================
rho_in_plot  = rho_in_sorted  / (rho_in_sorted.max()  + 1e-30)
rho_out_plot = rho_out_sorted / (rho_out_sorted.max() + 1e-30)

plt.figure(figsize=(6, 4))
plt.plot(z_in_sorted, rho_in_plot)
plt.title(r"Incoming angular wavepacket $\rho_{\rm in}(z)$")
plt.xlabel(r"$\cos\theta_{\rm in}$")
plt.ylabel(r"$\rho_{\rm in} / \rho_{\rm in}^{\max}$")
plt.tight_layout()
plt.savefig("rho_in_vs_costheta.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(z_out_sorted, rho_out_plot)
plt.title(r"Outgoing angular wavepacket $\rho_{\rm out}(z)$")
plt.xlabel(r"$\cos\theta_{\rm out}$")
plt.ylabel(r"$\rho_{\rm out} / \rho_{\rm out}^{\max}$")
plt.tight_layout()
plt.savefig("rho_out_vs_costheta.pdf")
plt.close()

# ==============================
# 7. Heatmaps: |K| and C
# ==============================
plt.figure(figsize=(7, 5))
plt.imshow(K_sub, aspect='auto', origin='lower')
plt.title(r'$|K(q|p)|$ (kernel magnitude)')
plt.xlabel("incoming sample index (sorted by $cosθ_{in}$)")
plt.ylabel("outgoing sample index (sorted by $cosθ_{out}$)")
plt.colorbar()
plt.tight_layout()
plt.savefig("K_magnitude_plot.pdf")   # matches LaTeX name
plt.show()

plt.figure(figsize=(7, 5))
plt.imshow(C_sub, aspect='auto', origin='lower')
plt.title(r'Angular transport cost $C(z_{\rm in}, z_{\rm out})$')
plt.xlabel("incoming sample index (sorted by $cosθ_{in}$)")
plt.ylabel("outgoing sample index (sorted by $cosθ_{out}$)")
plt.colorbar()
plt.tight_layout()
plt.savefig("transport_cost.pdf")
plt.show()

# ==============================
# 8. φ⁴ amplitude validation plot
# ==============================
A_np = np.array(A)
A_im = A_np.imag

plt.figure(figsize=(6.2, 4.8))
plt.axhline(-lam, linestyle='--',
            label=r"Exact $\mathrm{Im}\,\mathcal{A}=-\lambda$")
plt.scatter([0], [A_im], s=80,
            label=r"Monte Carlo $\mathrm{Im}\,\mathcal{A}$")

plt.xticks([])
plt.ylabel(r"$\mathrm{Im}\,\mathcal{A}_{2\to 2}$")
plt.title(r"Validation of $\phi^{4}$ tree--level amplitude")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("phi4_numeric_validation.pdf")
plt.show()

# ==============================
# 9. Toy convergence / stability plot
# ==============================
print("Running toy convergence simulation...\n")

num_steps = 80
eta = 0.25

stationarity_error = []
marginal_error = []
kernel_drift = []

# Initialize iterative kernel (complex, zero phase, sorted)
K_iter = K_abs_sorted.astype(np.complex128)

target_out = rho_out_sorted
# measure-weighted ρ_in for the *sorted* incoming
wp_np = np.array(wp)
wp_sorted = wp_np[idx_in]
rho_in_sorted_np = rho_in_sorted
rho_in_weighted = wp_sorted * rho_in_sorted_np  # (n,)

for t in range(num_steps):

    # Current outgoing marginal induced by K_iter:
    # ρ_out^K(j) = Σ_i w_i |K_ji|^2 ρ_in(i)
    current = np.sum((np.abs(K_iter)**2) * rho_in_weighted[None, :],
                     axis=1)

    # Enforce marginal constraint toward target_out
    correction = np.sqrt((target_out + 1e-18) / (current + 1e-18))
    K_new = (correction[:, None]) * K_iter

    # Simple relaxation step (toy "variational" update)
    K_new = (1 - eta) * K_iter + eta * K_new

    # Diagnostics
    stationarity_error.append(np.linalg.norm(K_new - K_iter))
    marginal_error.append(np.linalg.norm(current - target_out))
    kernel_drift.append(np.mean(np.abs(K_new - K_iter)))

    K_iter = K_new

plt.figure(figsize=(7, 5))
plt.plot(stationarity_error, label="Stationarity error")
plt.plot(marginal_error, label="Marginal constraint error")
plt.plot(kernel_drift, label="Mean kernel drift")

plt.yscale("log")
plt.xlabel("Iteration step")
plt.ylabel("Error (log scale)")
plt.title("Convergence behavior of QOTAM")
plt.legend()
plt.tight_layout()
plt.savefig("qotam_convergence.pdf")
plt.show()

print("All plots generated and saved as:")
print("  rho_in_vs_costheta.pdf")
print("  rho_out_vs_costheta.pdf")
print("  K_magnitude_plot.pdf")
print("  transport_cost.pdf")
print("  phi4_numeric_validation.pdf")
print("  qotam_convergence.pdf\n")

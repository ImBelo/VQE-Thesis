import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pennylane.qchem import excitations, excitations_to_wires

# -------------------------------------------------------------------
# Molecule and Hamiltonian
# -------------------------------------------------------------------
def create_h2_molecule(bond_length=0.6614, coordinates=None):
    """
    Create an H2 molecule in Bohr units.
    If coordinates is None, use default bond length along z-axis.
    """
    symbols = ["H", "H"]
    if coordinates is None:
        coordinates = np.array([0.0, 0.0, -bond_length, 0.0, 0.0, bond_length])
    mol = qml.qchem.Molecule(symbols, coordinates)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(mol)
    hf_state = qml.qchem.hf_state(2, qubits, basis="occupation_number")
    return mol, hamiltonian, qubits, hf_state

# -------------------------------------------------------------------
# UCCSD Ansatz
# -------------------------------------------------------------------
def get_uccsd_excitations(qubits, electrons=2):
    """Compute singles and doubles for UCCSD and convert to wire format."""
    singles, doubles = excitations(electrons, qubits)
    s_wires, d_wires = excitations_to_wires(singles, doubles)
    return singles, doubles, s_wires, d_wires

def uccsd_ansatz_h2_reduced(hamiltonian, hf_state, qubits, s_wires, d_wires, dev=None):
    """
    Return a QNode that computes the expectation value of the Hamiltonian 
    of the H2 molecule using the UCCSD ansatz with reduced parameter space
    """
    if dev is None:
        dev = qml.device("default.qubit", wires = qubits)

    @qml.qnode(dev)
    def circuit_reduced(params):

        mapped_params = qml.math.stack([params[0], params[0], params[1]])
        qml.UCCSD(mapped_params, wires=range(qubits), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(hamiltonian)

    return circuit_reduced
def uccsd_ansatz(hamiltonian, hf_state, qubits, s_wires, d_wires, dev=None):
    """
    Return a QNode that computes the expectation value of the Hamiltonian
    using the UCCSD ansatz.
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(params):
        qml.UCCSD(params, wires=range(qubits), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(hamiltonian)

    return circuit

# -------------------------------------------------------------------
# Optimization with history tracking
# -------------------------------------------------------------------
def run_vqe(circuit, n_params, optimizer=None, max_iterations=1000,
            convergence_threshold=1e-5, plateau_window=10,
            plateau_tolerance=1e-5, initial_params=None, print_every=10):
    """
    Optimize the given VQE circuit.

    Args:
        circuit: QNode returning energy.
        n_params: number of variational parameters.
        optimizer: PennyLane optimizer instance (default: AdamOptimizer 0.01).
        max_iterations: maximum number of steps.
        convergence_threshold: max energy difference in plateau window.
        plateau_window: number of steps to consider for plateau detection.
        plateau_tolerance: max allowed linear slope in plateau window.
        initial_params: initial parameters (random if None).
        print_every: print progress every N steps.

    Returns:
        history: dict with 'params', 'energies', 'steps'.
        final_params: optimized parameters.
        converged: bool.
    """
    if optimizer is None:
        optimizer = qml.AdamOptimizer(stepsize=0.01)

    if initial_params is None:
        params = pnp.random.random(n_params) * 0.1
    else:
        params = pnp.array(initial_params, requires_grad=True)

    history = {
        'params': [],
        'energies': [],
        'steps': []
    }

    iteration = 0
    converged = False

    while not converged and iteration < max_iterations:
        params, energy = optimizer.step_and_cost(circuit, params)

        history['params'].append(params.copy())
        history['energies'].append(energy)
        history['steps'].append(iteration)

        if iteration % print_every == 0:
            if iteration > 0 and not converged:
                # Optional: reduce stepsize gradually
                optimizer.stepsize /= 1.5
                print(f"   (Reduced stepsize to {optimizer.stepsize:.6f})")
            print(f"Step {iteration:4d}: Energy = {energy:.8f} Ha")
            print(f"   Params: {params}")

        # Plateau detection
        if iteration > plateau_window:
            recent = history['energies'][-plateau_window:]
            max_diff = max(recent) - min(recent)
            std_dev = np.std(recent)
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]

            if (max_diff < convergence_threshold and
                std_dev < convergence_threshold / 2 and
                abs(slope) < plateau_tolerance):
                converged = True
                print(f"\n CONVERGED at step {iteration}!")
                print(f"   Final energy: {energy:.8f} Ha")
                print(f"   Window max diff: {max_diff:.2e}")
                print(f"   Window std dev: {std_dev:.2e}")
                print(f"   Window slope: {slope:.2e} Ha/step")

        iteration += 1

    # Convert history to numpy arrays for convenience
    for key in history:
        history[key] = np.array(history[key])

    return history, params, converged

# -------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------
def plot_convergence(history, target_energy=None, convergence_threshold=1e-5):
    """Plot energy vs step and energy change."""
    steps = history['steps']
    energies = history['energies']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy convergence
    ax1.plot(steps, energies, 'b-', linewidth=2)
    if target_energy is not None:
        ax1.axhline(y=target_energy, color='r', linestyle='--',
                    label=f'Target ({target_energy} Ha)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Energy (Ha)')
    ax1.set_title('Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy change
    energy_changes = np.diff(energies)
    ax2.semilogy(steps[1:], np.abs(energy_changes), 'g-', linewidth=2)
    ax2.axhline(y=convergence_threshold, color='r', linestyle='--',
                label=f'Threshold ({convergence_threshold})')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('|ΔE| (Ha)')
    ax2.set_title('Energy Change per Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_parameter_trajectory(history):
    """
    Create 2D/3D plots of parameter evolution.
    Assumes at least 3 parameters.
    """
    params = history['params']
    steps = history['steps']
    energies = history['energies']

    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory (first three parameters)
    ax1 = fig.add_subplot(221, projection='3d')
    sc1 = ax1.scatter(params[:, 0], params[:, 1], params[:, 2],
                      c=steps, cmap='viridis', s=30, alpha=0.8)
    ax1.plot(params[:, 0], params[:, 1], params[:, 2],
             'gray', alpha=0.3, linewidth=1)
    ax1.scatter(params[0, 0], params[0, 1], params[0, 2],
                color='red', s=100, label='Start')
    ax1.scatter(params[-1, 0], params[-1, 1], params[-1, 2],
                color='green', s=100, label='End')
    ax1.set_xlabel('Param 1')
    ax1.set_ylabel('Param 2')
    ax1.set_zlabel('Param 3')
    ax1.set_title('Parameter Trajectory (3D)')
    ax1.legend()
    plt.colorbar(sc1, ax=ax1, label='Step')

    # Energy vs step
    ax2 = fig.add_subplot(222)
    ax2.plot(steps, energies, 'b-', linewidth=2)
    ax2.scatter(steps[::10], energies[::10], color='red', s=30)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Energy (Ha)')
    ax2.set_title('Energy Convergence')
    ax2.grid(True, alpha=0.3)

    # 2D projections
    ax3 = fig.add_subplot(223)
    sc3 = ax3.scatter(params[:, 0], params[:, 1], c=steps, cmap='viridis', s=30)
    ax3.plot(params[:, 0], params[:, 1], 'gray', alpha=0.3)
    ax3.set_xlabel('Param 1')
    ax3.set_ylabel('Param 2')
    ax3.set_title('Param 1 vs Param 2')
    plt.colorbar(sc3, ax=ax3, label='Step')

    ax4 = fig.add_subplot(224)
    sc4 = ax4.scatter(params[:, 1], params[:, 2], c=steps, cmap='viridis', s=30)
    ax4.plot(params[:, 1], params[:, 2], 'gray', alpha=0.3)
    ax4.set_xlabel('Param 2')
    ax4.set_ylabel('Param 3')
    ax4.set_title('Param 2 vs Param 3')
    plt.colorbar(sc4, ax=ax4, label='Step')

    plt.tight_layout()
    return fig

def plot_energy_colored_trajectory(history):
    """3D trajectory colored by energy."""
    params = history['params']
    energies = history['energies']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(params[:, 0], params[:, 1], params[:, 2],
                    c=energies, cmap='coolwarm', s=50, alpha=0.8)

    # Connect points with gradient lines
    for i in range(len(params)-1):
        color = plt.cm.coolwarm((energies[i] - energies.min()) /
                                (energies.max() - energies.min()))
        ax.plot(params[i:i+2, 0], params[i:i+2, 1], params[i:i+2, 2],
                color=color, linewidth=2, alpha=0.7)

    ax.set_xlabel('Param 1')
    ax.set_ylabel('Param 2')
    ax.set_zlabel('Param 3')
    ax.set_title('Parameter Trajectory Colored by Energy')

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label('Energy (Ha)')

    plt.tight_layout()
    return fig

def plot_interactive_trajectory(history):
    """Plotly interactive 3D trajectory (if plotly available)."""
    try:
        import plotly.graph_objects as go
        params = history['params']
        steps = history['steps']
        energies = history['energies']

        fig = go.Figure(data=[go.Scatter3d(
            x=params[:, 0], y=params[:, 1], z=params[:, 2],
            mode='markers+lines',
            marker=dict(size=5, color=energies, colorscale='Viridis',
                        showscale=True, colorbar=dict(title="Energy (Ha)")),
            line=dict(color='gray', width=2),
            text=[f'Step {s}<br>Energy: {e:.6f} Ha'
                  for s, e in zip(steps, energies)],
            hoverinfo='text'
        )])
        fig.update_layout(title='Interactive Parameter Trajectory',
                          scene=dict(xaxis_title='Param 1',
                                     yaxis_title='Param 2',
                                     zaxis_title='Param 3'),
                          width=900, height=700)
        fig.show()
    except ImportError:
        print("Plotly not installed. Skipping interactive plot.")

# -------------------------------------------------------------------
# Analysis functions
# -------------------------------------------------------------------
def analyze_parameters(history):
    """Print correlation and basic statistics."""
    params = history['params']
    if len(params) < 2:
        print("Not enough steps for analysis.")
        return

    print("\n" + "="*50)
    print("OPTIMIZATION STATISTICS")
    print("="*50)
    print(f"Initial parameters: {params[0]}")
    print(f"Final parameters: {params[-1]}")
    print(f"Initial energy: {history['energies'][0]:.6f} Ha")
    print(f"Final energy: {history['energies'][-1]:.6f} Ha")
    print(f"Energy improvement: {history['energies'][0] - history['energies'][-1]:.6f} Ha")
    print(f"Total steps: {len(params)}")

    print("\n" + "="*60)
    print("PARAMETER RELATIONSHIP ANALYSIS")
    print("="*60)

    if params.shape[1] >= 2:
        corr_12 = np.corrcoef(params[:, 0], params[:, 1])[0, 1]
        print(f"Correlation (p1 vs p2): {corr_12:.6f}")

    if params.shape[1] >= 3:
        if np.any(params[:, 1] != 0):
            corr_23_quad = np.corrcoef(params[:, 1]**2, params[:, 2])[0, 1]
            print(f"Correlation (p2² vs p3): {corr_23_quad:.6f}")
        print(f"Final p1: {params[-1, 0]:.6f}")
        print(f"Final p2: {params[-1, 1]:.6f}")
        print(f"Final p3: {params[-1, 2]:.6f}")

# -------------------------------------------------------------------
# Energy landscape for 2-parameter reduced model
# -------------------------------------------------------------------
def create_energy_landscape(circuit_2param, x_range, y_range, resolution=50):
    """
    Compute energy on a 2D grid for a circuit that takes exactly 2 parameters.
    Useful for visualizing landscape when two parameters are constrained equal.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    print("Computing energy landscape...")
    for i in range(resolution):
        for j in range(resolution):
            # circuit_2param expects [x, y] (here x may be reused if needed)
            Z[i, j] = circuit_2param(np.array([X[i, j], Y[i, j]]))
        if i % 10 == 0:
            print(f"  Row {i}/{resolution} complete")

    return X, Y, Z

def plot_energy_landscape(X, Y, Z, opt_point=None, trajectory=None):
    """Create surface, contour, and heatmap plots of the energy landscape."""
    fig = plt.figure(figsize=(20, 12))

    # 3D Surface
    ax1 = fig.add_subplot(231, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0)
    if opt_point:
        ax1.scatter(*opt_point, color='red', s=200, marker='*', label='Optimum')
    ax1.set_xlabel('Parameter 1')
    ax1.set_ylabel('Parameter 2')
    ax1.set_zlabel('Energy (Ha)')
    ax1.set_title('3D Energy Landscape')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Energy (Ha)')

    # Contour
    ax2 = fig.add_subplot(232)
    contour = ax2.contour(X, Y, Z, levels=20, cmap=cm.viridis)
    ax2.clabel(contour, inline=True, fontsize=8)
    if opt_point:
        ax2.scatter(opt_point[0], opt_point[1], color='red', s=200, marker='*',
                    label='Optimum')
    if trajectory is not None:
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2,
                 alpha=0.7, label='Path')
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', s=100,
                    marker='o', label='Start')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', s=100,
                    marker='s', label='End')
    ax2.set_xlabel('Parameter 1')
    ax2.set_ylabel('Parameter 2')
    ax2.set_title('Energy Contours')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Heatmap
    ax3 = fig.add_subplot(233)
    im = ax3.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                    origin='lower', cmap=cm.viridis, aspect='auto', alpha=0.8)
    if opt_point:
        ax3.scatter(opt_point[0], opt_point[1], color='red', s=200, marker='*',
                    label='Optimum')
    ax3.set_xlabel('Parameter 1')
    ax3.set_ylabel('Parameter 2')
    ax3.set_title('Energy Heatmap')
    plt.colorbar(im, ax=ax3, label='Energy (Ha)')

    plt.tight_layout()
    return fig

def analyze_landscape(X, Y, Z, opt_point):
    """Print curvature information at the optimum."""
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_idx]
    min_y = Y[min_idx]
    min_energy = Z[min_idx]

    print("\n" + "="*60)
    print("ENERGY LANDSCAPE ANALYSIS")
    print("="*60)
    print(f"Grid minimum: p1={min_x:.6f}, p2={min_y:.6f}, E={min_energy:.8f} Ha")
    print(f"Provided optimum: p1={opt_point[0]:.6f}, p2={opt_point[1]:.6f}, E={opt_point[2]:.8f} Ha")

    # Approximate Hessian via finite differences
    hessian_xx = np.gradient(np.gradient(Z, axis=0), axis=0)[min_idx]
    hessian_yy = np.gradient(np.gradient(Z, axis=1), axis=1)[min_idx]
    hessian_xy = np.gradient(np.gradient(Z, axis=0), axis=1)[min_idx]

    if hessian_xx > 0 and hessian_yy > 0:
        print("\nMinimum is a true convex minimum (both curvatures positive)")
    else:
        print("\nMinimum might be a saddle point or plateau")

    if hessian_yy != 0:
        ratio = hessian_xx / hessian_yy
        print(f"\nCurvature ratio (xx/yy): {ratio:.2f}")
        if ratio < 0.1:
            print("  Landscape is much steeper in the y-direction (parameter 2)")
        elif ratio > 10:
            print("  Landscape is much steeper in the x-direction (parameter 1)")
        else:
            print("  Similar curvature in both directions")
    print("="*60)

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

# Import the modular functions from your utils file
from vqe_utils import (
    get_uccsd_excitations,
    uccsd_ansatz,
    run_vqe,
    plot_convergence,
    plot_parameter_trajectory,
    plot_energy_colored_trajectory,
    analyze_parameters
)

def create_h2o_molecule(basis="sto-3g", bond_length_oh=0.958, bond_angle=104.5):
    """
    Create H2O molecule.
    
    Args:
        basis: Basis set (sto-3g, 6-31g, etc.)
        bond_length_oh: O-H bond length in Angstroms (converted to Bohr)
        bond_angle: H-O-H bond angle in degrees
    
    Returns:
        mol, hamiltonian, qubits, hf_state
    """
    # Convert to Bohr (1 Angstrom = 1.8897261245 Bohr)
    bohr_per_angstrom = 1.8897261245
    bond_length_bohr = bond_length_oh * bohr_per_angstrom
    
    # Convert angle to radians
    angle_rad = np.radians(bond_angle / 2)  # Half-angle for each H
    
    # Place O at origin, H's in xz-plane
    # H1: at angle/2 from z-axis
    # H2: at -angle/2 from z-axis
    x1 = bond_length_bohr * np.sin(angle_rad)
    z1 = bond_length_bohr * np.cos(angle_rad)
    x2 = -bond_length_bohr * np.sin(angle_rad)
    z2 = bond_length_bohr * np.cos(angle_rad)
    
    # Coordinates in Bohr: [O_x, O_y, O_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z]
    coordinates = np.array([0.0, 0.0, 0.0, x1, 0.0, z1, x2, 0.0, z2])
    
    symbols = ["O", "H", "H"]
    electrons = 10  # 8 from O + 1 from each H
    
    print(f"H2O Geometry (Bohr):")
    print(f"  O:  (0.000, 0.000, 0.000)")
    print(f"  H1: ({x1:.4f}, 0.000, {z1:.4f})")
    print(f"  H2: ({x2:.4f}, 0.000, {z2:.4f})")
    print(f"  Bond length: {bond_length_oh:.3f} Å ({bond_length_bohr:.3f} Bohr)")
    print(f"  Bond angle: {bond_angle:.1f}°")
    
    # Create molecule
    mol = qml.qchem.Molecule(symbols, coordinates, basis_name=basis)
    
    # Get Hamiltonian (this may take a while for larger basis sets)
    print(f"\nComputing Hamiltonian with {basis} basis...")
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(mol)
    
    # Get HF state
    hf_state = qml.qchem.hf_state(electrons, qubits, basis="occupation_number")
    
    return mol, hamiltonian, qubits, hf_state, electrons

def filter_excitations_by_qubits(singles, doubles, qubits):
    """
    Filter excitations to ensure they're valid for the number of qubits.
    Some excitations might exceed the number of orbitals/qubits.
    """
    valid_singles = []
    valid_doubles = []
    
    for s in singles:
        if max(s) < qubits:
            valid_singles.append(s)
    
    for d in doubles:
        # doubles are stored as [wires for excitation]
        if max(d) < qubits:
            valid_doubles.append(d)
    
    return valid_singles, valid_doubles

def analyze_h2o_results(history, qubits, electrons):
    """Print H2O-specific analysis."""
    print("\n" + "="*60)
    print("H2O VQE RESULTS")
    print("="*60)
    print(f"System: H2O")
    print(f"Electrons: {electrons}")
    print(f"Qubits: {qubits}")
    print(f"Number of parameters: {history['params'].shape[1]}")
    print(f"Final energy: {history['energies'][-1]:.8f} Ha")
    
    # Reference energies (approximate - will vary with basis set)
    ref_energies = {
        "sto-3g": -74.96,  # Approximate STO-3G energy
        "6-31g": -75.98,   # Approximate 6-31G energy
        "cc-pvdz": -76.24,  # Approximate cc-pVDZ energy
    }
    
    print("\nReference energies (approximate):")
    for basis, e in ref_energies.items():
        print(f"  {basis}: {e} Ha")

def main():
    # -------------------------------------------------------------------
    # 1. Set up H2O molecule
    # -------------------------------------------------------------------
    # Choose basis set (smaller = faster but less accurate)
    basis = "sto-3g"  # Start with minimal basis for testing
    # basis = "6-31g"   # More accurate, more qubits
    # basis = "cc-pvdz" # Even more accurate, even more qubits
    
    mol, hamiltonian, qubits, hf_state, electrons = create_h2o_molecule(
        basis=basis,
        bond_length_oh=0.958,  # Equilibrium bond length in Å
        bond_angle=104.5        # Equilibrium bond angle in degrees
    )
    
    print(f"\nHamiltonian: {hamiltonian}")
    print(f"\nNumber of qubits: {qubits}")
    print(f"Hartree-Fock state: {hf_state}")
    
    # -------------------------------------------------------------------
    # 2. Get UCCSD excitations
    # -------------------------------------------------------------------
    print("\nGenerating UCCSD excitations...")
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    
    # Filter excitations to ensure they're valid
    singles, doubles = filter_excitations_by_qubits(singles, doubles, qubits)
    
    print(f"Number of singles: {len(singles)}")
    print(f"Number of doubles: {len(doubles)}")
    
    if len(singles) + len(doubles) == 0:
        print("WARNING: No excitations found! Trying with different parameters...")
        # For minimal basis, we might need to adjust
        singles = [[0, 4], [0, 5], [1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]]
        doubles = []
        print(f"Using manual singles: {len(singles)}")
    
    # Convert to wire format
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    
    n_params = len(singles) + len(doubles)
    print(f"Total UCCSD parameters: {n_params}")
    
    # -------------------------------------------------------------------
    # 3. Create device and circuit
    # -------------------------------------------------------------------
    # For larger systems, you might want to use a faster simulator
    dev = qml.device("lighting.qubit", wires=qubits)
    
    uccsd_circuit = uccsd_ansatz(
        hamiltonian, 
        hf_state, 
        qubits, 
        s_wires, 
        d_wires, 
        dev
    )
    
    # -------------------------------------------------------------------
    # 4. Run optimization
    # -------------------------------------------------------------------
    print("\nStarting VQE optimization...")
    
    # Use a smaller stepsize for stability
    opt = qml.AdamOptimizer(stepsize=0.005)
    
    # Initialize parameters close to zero (near HF)
    initial_params = pnp.random.normal(0, 0.01, n_params)
    initial_params.requires_grad = True
    
    history, final_params, converged = run_vqe(
        uccsd_circuit,
        n_params,
        optimizer=opt,
        max_iterations=2000,  # More iterations for H2O
        convergence_threshold=1e-4,  # Slightly looser convergence
        plateau_window=20,  # Longer window for plateau detection
        plateau_tolerance=1e-4,
        initial_params=initial_params,
        print_every=50  # Print less frequently
    )
    
    # -------------------------------------------------------------------
    # 5. Analyze results
    # -------------------------------------------------------------------
    analyze_h2o_results(history, qubits, electrons)
    
    # -------------------------------------------------------------------
    # 6. Visualize
    # -------------------------------------------------------------------
    plot_convergence(history, convergence_threshold=1e-4)
    plt.show()
    
    if qubits >= 3 and n_params >= 3:
        plot_parameter_trajectory(history)
        plt.show()
        
        plot_energy_colored_trajectory(history)
        plt.show()
    
    analyze_parameters(history)
    

if __name__ == "__main__":
    main()

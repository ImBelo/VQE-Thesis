import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from vqe_utils import (
    create_h2_molecule,
    get_uccsd_excitations,
    uccsd_ansatz,
    run_vqe,
    plot_convergence,
    plot_parameter_trajectory,
    plot_energy_colored_trajectory,
    plot_interactive_trajectory,
    analyze_parameters,
    create_energy_landscape,
    plot_energy_landscape,
    analyze_landscape,
    uccsd_ansatz_h2_reduced,
)

def main():
    # -------------------------------------------------------------------
    # 1. Set up molecule and Hamiltonian
    # -------------------------------------------------------------------
    mol, hamiltonian, qubits, hf_state = create_h2_molecule(bond_length=0.6614)
    print(f"Number of qubits: {qubits}")
    print(f"Hamiltonian: {hamiltonian}")
    print(f"Hartree-Fock state: {hf_state}")

    # -------------------------------------------------------------------
    # 2. Define UCCSD ansatz
    # -------------------------------------------------------------------
    singles, doubles, s_wires, d_wires = get_uccsd_excitations(qubits, electrons=2)
    print(f"Singles (endpoints): {singles}")
    print(f"Doubles (flat): {doubles}")

    dev = qml.device("default.qubit", wires=qubits)
    uccsd_circuit = uccsd_ansatz(hamiltonian, hf_state, qubits, s_wires, d_wires, dev)
    uccsd_circuit_h2_reduced = uccsd_ansatz_h2_reduced(hamiltonian, hf_state, qubits, s_wires, d_wires, dev)

    n_params = len(singles) + len(doubles)
    print(f"Number of UCCSD parameters: {n_params}")

    # -------------------------------------------------------------------
    # 3. Run VQE optimization
    # -------------------------------------------------------------------
    opt = qml.AdamOptimizer(stepsize=0.01)
    history, final_params, converged = run_vqe(
        uccsd_circuit,
        n_params,
        optimizer=opt,
        max_iterations=1000,
        convergence_threshold=1e-5,
        plateau_window=10,
        plateau_tolerance=1e-5,
        print_every=10
    )
    history_reduced, final_params_reduced, converged_reduced = run_vqe(
        uccsd_circuit_h2_reduced,
        n_params,
        optimizer=opt,
        max_iterations=1000,
        convergence_threshold=1e-5,
        plateau_window=10,
        plateau_tolerance=1e-5,
        print_every=10
    )


    # -------------------------------------------------------------------
    # 4. Print final results
    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Total steps: {len(history['steps'])}")
    print(f"Final energy: {history['energies'][-1]:.8f} Ha")
    print(f"Final parameters: {final_params}")
    target_energy = -1.136189  # known exact energy for H2 at this bond length
    print(f"Expected energy: {target_energy} Ha")
    print(f"Difference: {abs(history['energies'][-1] - target_energy):.2e} Ha")
    print(f"Status: {'CONVERGED' if converged else 'Max iterations reached'}")
    cov_matrix = np.corrcoef(history['params'], rowvar=False)
    print(history['params'])
    print(f"Covariance: {cov_matrix}")

    labels = ['Param 1', 'Param 2', 'Param 3']
    sns.heatmap(cov_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
    plt.title("Parameter Covariance")
    plt.show()



    # -------------------------------------------------------------------
    # 5. Visualize convergence and parameter trajectories
    # -------------------------------------------------------------------
    plot_convergence(history, target_energy=target_energy, convergence_threshold=1e-5)
    plt.show()
    plot_convergence(history_reduced, target_energy=target_energy, convergence_threshold=1e-5)

    params_array = history['params']

    if params_array.shape[1] >= 3:
        plot_parameter_trajectory(history)
        plt.show()

        plot_energy_colored_trajectory(history)
        plt.show()

        plot_interactive_trajectory(history)

    # -------------------------------------------------------------------
    # 6. Parameter analysis
    # -------------------------------------------------------------------
    analyze_parameters(history)

    # -------------------------------------------------------------------
    # 7. (Optional) Energy landscape for reduced 2-parameter model
    #    Assumes p1 = p2 (single excitations) and p3 (double excitations)
    # -------------------------------------------------------------------
    # Define a circuit that takes [x, y] where x is the single excitation
    # amplitude (used for both p1 and p2) and y is the double amplitude.
    def uccsd_2param(params_2d):
        # params_2d[0] -> p1 and p2, params_2d[1] -> p3
        full_params = pnp.array([params_2d[0], params_2d[0], params_2d[1]], requires_grad=False)
        return uccsd_circuit(full_params)

    # Ranges around the optimal values from the full optimization
    opt_p1 = final_params[0]  # or average of p1 and p2
    opt_p2 = final_params[2]  # p3
    x_range = [opt_p1 - 0.1, opt_p1 + 0.1]
    y_range = [opt_p2 - 0.2, opt_p2 + 0.2]

    X, Y, Z = create_energy_landscape(uccsd_2param, x_range, y_range, resolution=50)

    # Extract optimization path in reduced space
    reduced_path = np.column_stack([
        (history['params'][:, 0] + history['params'][:, 1]) / 2,  # average p1,p2
        history['params'][:, 2]                                    # p3
    ])

    plot_energy_landscape(X, Y, Z,
                          opt_point=(opt_p1, opt_p2, history['energies'][-1]),
                          trajectory=reduced_path)
    plt.show()

    analyze_landscape(X, Y, Z, (opt_p1, opt_p2, history['energies'][-1]))

if __name__ == "__main__":
    main()

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from analysis.vqe_pipeline import VQEPipeline

# Set publication-quality figure formatting
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11

def draw_circuit_from_config(mol_config: dict, ansatz_config: dict, save_filename: str):
    """
    Instantiates the pipeline using the provided configs and draws the expanded circuit.
    """
    # 1. Dummy config blocks for optimizer and noise (required by pipeline class builder)
    opt_config = {"type": "adam", "lr": 0.1}
    noise_config = {"model": "none","strength": "low" }

    # 2. Build pipeline using your orchestrator method
    #    (Assumes VQERunner / Pipeline class has the from_config method)
    pipeline = VQEPipeline.from_config(
        mol_config=mol_config,
        ansatz_config=ansatz_config,
        opt_config=opt_config,
        noise_config=noise_config
    )

    # 3. Create execution device and define drawing QNode
    dev = qml.device("default.qubit", wires=pipeline.n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        # Prepare initial Hartree-Fock state in the chosen qubit basis
        qml.BasisState(pipeline.h_ref, wires=range(pipeline.n_qubits))
        # Call ansatz instance
        pipeline.ansatz(params, h_ref=pipeline.h_ref)
        return qml.state()

    # 4. Generate random parameters matching the ansatz size
    num_params = pipeline.ansatz.get_num_params()
    params = np.random.randn(num_params)

    # 5. Draw and render matplotlib diagram at device gate level
    fig, ax = qml.draw_mpl(circuit, level="device")(params)
    fig.set_size_inches(26, 6)
    
    mapping_title = mol_config['mapping'].replace('_', ' ').title()
    ax.set_title(
        f"UCCSD Circuit | Molecule: {mol_config['name']} | Mapping: {mapping_title} | HF State: {pipeline.h_ref}",
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved circuit diagram to {save_filename}")


# ==========================================
# Execution Configuration Example
# ==========================================
if __name__ == "__main__":
    
    # --- Config 1: Jordan-Wigner UCCSD ---
    jw_mol_config = {"name": "H2", "basis": "sto-3g", "mapping": "jordan_wigner"}
    jw_ansatz_config = {"type": "uccsd", "mapping": "jordan_wigner", "n_layers": 1}
    
    draw_circuit_from_config(
        mol_config=jw_mol_config,
        ansatz_config=jw_ansatz_config,
        save_filename="uccsd_jordan_wigner.pdf"
    )

    # --- Config 2: Bravyi-Kitaev UCCSD ---
    bk_mol_config = {"name": "H2", "basis": "sto-3g", "mapping": "bravyi_kitaev"}
    bk_ansatz_config = {"type": "uccsd", "mapping": "bravyi_kitaev", "n_layers": 1}
    
    draw_circuit_from_config(
        mol_config=bk_mol_config,
        ansatz_config=bk_ansatz_config,
        save_filename="uccsd_bravyi_kitaev.pdf"
    )

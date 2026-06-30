from utils.config.config import DB_PATH, apply_publication_theme, OUTPUT_DIR
from visualization import convergence, efficiency, mapping_comparison
 
if __name__ == "__main__":
    print("=== Initializing Modular Thesis Visualizations Generator ===")
    print(f"Target DB Path: {DB_PATH.resolve()}\n")
    
    if not DB_PATH.exists():
        print("[CRITICAL ERROR]: SQLite database file was not detected at path location.")
        print("Verify your configuration paths and check your dataset pipeline infrastructure.")
        exit(1)
        
    apply_publication_theme()
    
    MOL = "H2"
    BASIS = "sto-3g"

    print(f"[1/6] Plotting Convergence Trajectories for {MOL}...")
    convergence.plot_trajectories(DB_PATH, mol_name=MOL, basis=BASIS)
    
    print(f"[2/6] Plotting Multi-Noise Layer Rollercoasters...")
    convergence.plot_multi_noise_rollercoasters(DB_PATH)
    
    print(f"[3/6] Plotting Optimizer Efficiency Matrix maps...")
    efficiency.plot_optimizer_efficiency(DB_PATH, mol_name=MOL)
    
    print(f"[4/6] Plotting Resource vs Accuracy Faceted Multi-grids...")
    efficiency.plot_resource_vs_accuracy_faceted(DB_PATH, mol_name=MOL, basis=BASIS)
    
    print(f"[5/6] Plotting Noise Mapping Resilience Curves...")
    mapping_comparison.plot_noise_resilience(DB_PATH, mol_name=MOL, basis=BASIS)
    
    print(f"[6/6] Plotting Fermionic Mapping Delta Lines (JW vs. BK)...")
    mapping_comparison.plot_mapping_delta(DB_PATH, mol_name=MOL, basis=BASIS)
    
    print(f"\n=== Visual Analytics Pipeline Completed Successfully ===")
    print(f"All rendering files outputs saved directly inside directory: {OUTPUT_DIR.resolve()}/")

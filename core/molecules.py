from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pennylane as qml

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pennylane as qml
from pennylane import qchem
from pennylane.qchem.structure import hf_state


@dataclass
class MoleculeConfig:
    """Configuration for creating a molecular system"""
    name: str                       # e.g., "H2", "H2O", "Custom"
    basis: str                      # e.g., "sto-3g", "6-31g"
    mapping: str = "jordan_wigner"  # "jordan_wigner", "parity", "bravyi_kitaev"
    charge: int = 0
    multiplicity: int = 1
    geometry: Optional[List[Tuple[str, List[float]]]] = None


class PhysicalMolecule:
    """ 
    Real physical molecule described in fermionic Fock space
    This class is independent of any qubit mapping
    """
    def __init__(self, symbols, coordinates,basis: str, charge=0):  
        self.charge: int = charge
        self.basis: str = basis
        self.symbols = symbols
        self.coordinates = np.array(coordinates)
        self.charge = charge
        # Build the PennyLane Molecule object (fermionic)
        self._mol = qchem.Molecule(symbols, self.coordinates, charge=charge,basis_name = basis)

    @property
    def n_electrons(self) -> int:
        return self._mol.n_electrons

    @property
    def n_orbitals(self) -> int:
        return self._mol.n_orbitals

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    def apply_mapping(self, mapping: str) -> 'QubitMappedSystem':
        return QubitMappedSystem(self, mapping)
    def __repr__(self) -> str:
        return (f"PhysicalMolecule({self.symbols}, charge={self.charge}, "
                f"basis={self.basis}, e⁻={self.n_electrons}, orbitals={self.n_orbitals})")


class QubitMappedSystem:
    """Virtual quantum computer representation after applying a mapping"""
    def __init__(self, physical: PhysicalMolecule, mapping: str):
        self.physical = physical


        # Build the qubit Hamiltonian made with pauli gates by using mapping
        raw_hamiltonian, self.n_qubits = qchem.molecular_hamiltonian(
            physical._mol,
            mapping=mapping
        )
        ham = qml.simplify(raw_hamiltonian)

        # Mapping to int instead of numpy because a lighting.qubit device bug
        wire_map = {w: int(w) for w in ham.wires}
        self.hamiltonian = qml.map_wires(ham, wire_map)
            
        # Hartree‑Fock state in the qubit basis
        self.hf_state = qchem.hf_state(
            electrons=physical.n_electrons, 
            orbitals=self.n_qubits, 
            basis=mapping  
        )

    def __repr__(self) -> str:
        return (f"QubitMappedSystem(mapping={self.mapping}, n_qubits={self.n_qubits}, "
                f"physical={self.physical.symbols})")


class MoleculeFactory:
    """
    Factory for creating physical molecules and their qubit‑mapped counterparts
    Pre‑defined geometries and basis set validation are included
    """
    # Pre‑defined geometries (in Bohr)
    GEOMETRIES: Dict[str, List[Tuple[str, List[float]]]] = {
        "H2": [
            ("H", [0.0, 0.0, -0.7]),
            ("H", [0.0, 0.0,  0.7])
        ],

        "H2O": [
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.0, 0.757, 0.587]),
            ("H", [0.0, -0.757, 0.587])
        ],
    }

    # Valid basis sets per molecule (for validation)
    VALID_BASIS: Dict[str, List[str]] = {
        "H2": ["sto-3g", "6-31g"],
        "H2O": ["sto-3g", "6-31g"],
        
    }

    @classmethod
    def create_physical(cls, name, basis, geometry = None) -> PhysicalMolecule:
        """
        Create a PhysicalMolecule from a configuration dictionary.

        Expected keys:
            name (str): Molecule name (must be in GEOMETRIES if geometry not given)
            basis (str): Basis set name
            charge (int, optional): Molecular charge
            geometry (list, optional): Override geometry

        Returns:
            PhysicalMolecule
        """

        if geometry is None:
            geometry = cls.GEOMETRIES.get(name)
            if geometry is None:
                raise ValueError(f"Unknown molecule '{name}' and no custom geometry provided.")
        else:
            pass

        symbols = [atom[0] for atom in geometry]
        coordinates = [atom[1] for atom in geometry]

        return PhysicalMolecule(symbols, coordinates , basis=basis)

    @classmethod
    def create_mapped(cls, physical: PhysicalMolecule, mapping) -> QubitMappedSystem:
        """
        Create a QubitMappedSystem (quantum computer representation) directly.

        Config may contain:
            name, basis, charge, geometry (as in create_physical)
            mapping (str): fermion‑to‑qubit mapping, default "jordan_wigner"

        """
        return physical.apply_mapping(mapping)

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate required fields and basis compatibility."""
        required = ["name", "basis"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required field: {key}")

        name = config["name"]
        basis = config["basis"]

        # Only validate basis if using a predefined molecule (name known)
        if name in cls.VALID_BASIS and basis not in cls.VALID_BASIS[name]:

            raise ValueError(f"Invalid basis '{basis}' for molecule '{name}'. "
                             f"Valid: {cls.VALID_BASIS[name]}")

   


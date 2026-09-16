from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import qchem


Geometry = List[Tuple[str, List[float]]]


@dataclass
class MoleculeConfig:
    """Configuration for creating a molecular system."""

    name: str
    basis: str
    mapping: str = "jordan_wigner"
    charge: int = 0
    multiplicity: int = 1
    geometry: Optional[Geometry] = None


class PhysicalMolecule:
    """
    Real physical molecule described in fermionic Fock space.

    Independent of any qubit mapping.
    """

    symbols: List[str]
    coordinates: np.ndarray
    basis: str
    charge: int
    _mol: qchem.Molecule

    def __init__(
        self,
        symbols: List[str],
        coordinates: List[List[float]] | np.ndarray,
        basis: str,
        charge: int = 0,
    ) -> None:
        self.charge: int = charge
        self.basis: str = basis
        self.symbols: List[str] = list(symbols)
        self.coordinates: np.ndarray = np.array(coordinates)
        self._mol: qchem.Molecule = qchem.Molecule(
            self.symbols,
            self.coordinates,
            charge=charge,
            basis_name=basis,
        )

    @property
    def n_electrons(self) -> int:
        return self._mol.n_electrons

    @property
    def n_orbitals(self) -> int:
        return self._mol.n_orbitals

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_orbitals

    def apply_mapping(self, mapping: str) -> QubitMappedSystem:
        return QubitMappedSystem(self, mapping)

    def __repr__(self) -> str:
        return (
            f"PhysicalMolecule({self.symbols}, charge={self.charge}, "
            f"basis={self.basis}, e⁻={self.n_electrons}, orbitals={self.n_orbitals})"
        )


class QubitMappedSystem:
    """Virtual quantum computer representation after applying a mapping."""

    physical: PhysicalMolecule
    hamiltonian: qml.Hamiltonian
    hf_state: np.ndarray
    n_qubits: int
    mapping: str

    def __init__(self, physical: PhysicalMolecule, mapping: str) -> None:
        self.physical: PhysicalMolecule = physical
        self.mapping: str = mapping

        raw_hamiltonian, self.n_qubits = qchem.molecular_hamiltonian(
            physical._mol,
            mapping=mapping,
        )
        ham = qml.simplify(raw_hamiltonian)

        # Map wires to plain ints — workaround for a lightning.qubit bug.
        wire_map = {w: int(w) for w in ham.wires}
        self.hamiltonian: qml.Hamiltonian = qml.map_wires(ham, wire_map)

        self.hf_state: np.ndarray = qchem.hf_state(
            electrons=physical.n_electrons,
            orbitals=self.n_qubits,
            basis=mapping,
        )

    def __repr__(self) -> str:
        return (
            f"QubitMappedSystem(mapping={self.mapping}, n_qubits={self.n_qubits}, "
            f"physical={self.physical.symbols})"
        )


class MoleculeFactory:
    """
    Factory for creating physical molecules and their qubit-mapped counterparts.

    Predefined geometries and basis-set validation are included.
    """

    GEOMETRIES: Dict[str, Geometry] = {
        "H2": [
            ("H", [0.0, 0.0, -0.7]),
            ("H", [0.0, 0.0, 0.7]),
        ],
        "H2O": [
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.0, 0.757, 0.587]),
            ("H", [0.0, -0.757, 0.587]),
        ],
    }

    VALID_BASIS: Dict[str, List[str]] = {
        "H2": ["sto-3g", "6-31g"],
        "H2O": ["sto-3g", "6-31g"],
    }

    @classmethod
    def create_physical(
        cls,
        name: str,
        basis: str,
        geometry: Optional[Geometry] = None,
    ) -> PhysicalMolecule:
        """Create a PhysicalMolecule from name/basis (and optional geometry)."""
        if geometry is None:
            geometry = cls.GEOMETRIES.get(name)
            if geometry is None:
                raise ValueError(
                    f"Unknown molecule '{name}' and no custom geometry provided."
                )

        symbols: List[str] = [atom[0] for atom in geometry]
        coordinates: List[List[float]] = [atom[1] for atom in geometry]

        return PhysicalMolecule(symbols, coordinates, basis=basis)

    @classmethod
    def create_mapped(
        cls,
        physical: PhysicalMolecule,
        mapping: str,
    ) -> QubitMappedSystem:
        """Apply a fermion-to-qubit mapping to a PhysicalMolecule."""
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

        if name in cls.VALID_BASIS and basis not in cls.VALID_BASIS[name]:
            raise ValueError(
                f"Invalid basis '{basis}' for molecule '{name}'. "
                f"Valid: {cls.VALID_BASIS[name]}"
            )

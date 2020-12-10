"""
This module provides a ASE calculator class [#ase1]_ for SchNetPack models, as
well as a general Interface to all ASE calculation methods, such as geometry
optimisation, normal mode computation and molecular dynamics simulations.

References
----------
.. [#ase1] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
    Groves, Hammer, Hargus: The atomic simulation environment -- a Python
    library for working with atoms.
    Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import os

from schnetpack.data.atoms import AtomsConverter

from ase import units
from ase.calculators.calculator import Calculator, all_changes

from schnetpack.md.utils import MDUnits

from schnetpack.environment import SimpleEnvironmentProvider


class SpkCalculatorError(Exception):
    pass


class SpkCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        ml_model (schnetpack.AtomisticModel): Trained model for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
        collect_triples (bool): Set to True if angular features are needed,
            for example, while using 'wascf' models
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase calculator class
    """

    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(
        self,
        model,
        device="cpu",
        collect_triples=False,
        environment_provider=SimpleEnvironmentProvider(),
        energy=None,
        forces=None,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model = model

        self.atoms_converter = AtomsConverter(
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            device=device,
        )

        self.model_energy = energy
        self.model_forces = forces

        # Convert to ASE internal units
        self.energy_units = 1/units.Ha
        self.forces_units = 1/(units.Ha / units.Bohr)
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        Calculator.calculate(self, atoms)

        # Convert to schnetpack input format
        model_inputs = self.atoms_converter(atoms)
        # Call model
        model_results = self.model(model_inputs)

        results = {}
        # Convert outputs to calculator format
        if self.model_energy is not None:
            if self.model_energy not in model_results.keys():
                raise SpkCalculatorError(
                    "'{}' is not a property of your model. Please "
                    "check the model "
                    "properties!".format(self.model_energy)
                )
            energy = model_results[self.model_energy].cpu().data.numpy()
            results[self.energy] = energy.reshape(-1) * self.energy_units

        if self.model_forces is not None:
            if self.model_forces not in model_results.keys():
                raise SpkCalculatorError(
                    "'{}' is not a property of your model. Please "
                    "check the model"
                    "properties!".format(self.model_forces)
                )
            forces = -1. * model_results[self.model_forces].cpu().data.numpy()
            results[self.forces] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = results

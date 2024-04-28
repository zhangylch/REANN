import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings

import numpy as np

from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin


class CalculatorError(RuntimeError):
    """Base class of error types related to ASE calculators."""


class CalculatorSetupError(CalculatorError):
    """Calculation cannot be performed with the given parameters.

    Reasons to raise this errors are:
      * The calculator is not properly configured
        (missing executable, environment variables, ...)
      * The given atoms object is not supported
      * Calculator parameters are unsupported

    Typically raised before a calculation."""


class EnvironmentError(CalculatorSetupError):
    """Raised if calculator is not properly set up with ASE.

    May be missing an executable or environment variables."""


class InputError(CalculatorSetupError):
    """Raised if inputs given to the calculator were incorrect.

    Bad input keywords or values, or missing pseudopotentials.

    This may be raised before or during calculation, depending on
    when the problem is detected."""


class CalculationFailed(CalculatorError):
    """Calculation failed unexpectedly.

    Reasons to raise this error are:
      * Calculation did not converge
      * Calculation ran out of memory
      * Segmentation fault or other abnormal termination
      * Arithmetic trouble (singular matrices, NaN, ...)

    Typically raised during calculation."""


class SCFError(CalculationFailed):
    """SCF loop did not converge."""


class ReadError(CalculatorError):
    """Unexpected irrecoverable error while reading calculation results."""


class PropertyNotImplementedError(NotImplementedError):
    """Raised if a calculator does not implement the requested property."""


class PropertyNotPresent(CalculatorError):
    """Requested property is missing.

    Maybe it was never calculated, or for some reason was not extracted
    with the rest of the results, without being a fatal ReadError."""


def compare_atoms(atoms1, atoms2, tol=1e-15, excluded_properties=None):
    """Check for system changes since last calculation.  Properties in
    ``excluded_properties`` are not checked."""
    if atoms1 is None:
        system_changes = all_changes[:]
    else:
        system_changes = []

        properties_to_check = set(all_changes)
        if excluded_properties:
            properties_to_check -= set(excluded_properties)

        # Check properties that aren't in Atoms.arrays but are attributes of
        # Atoms objects
        for prop in ['cell', 'pbc']:
            if prop in properties_to_check:
                properties_to_check.remove(prop)
                if not equal(getattr(atoms1, prop), getattr(atoms2, prop),
                             atol=tol):
                    system_changes.append(prop)

        arrays1 = set(atoms1.arrays)
        arrays2 = set(atoms2.arrays)

        # Add any properties that are only in atoms1.arrays or only in
        # atoms2.arrays (and aren't excluded).  Note that if, e.g. arrays1 has
        # `initial_charges` which is merely zeros and arrays2 does not have
        # this array, we'll still assume that the system has changed.  However,
        # this should only occur rarely.
        system_changes += properties_to_check & (arrays1 ^ arrays2)

        # Finally, check all of the non-excluded properties shared by the atoms
        # arrays.
        for prop in properties_to_check & arrays1 & arrays2:
            if not equal(atoms1.arrays[prop], atoms2.arrays[prop], atol=tol):
                system_changes.append(prop)

    return system_changes


all_properties = ['energy', 'forces', 'stress', 'stresses', 'dipole',
                  'charges', 'magmom', 'magmoms', 'free_energy', 'energies']


all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']


# Recognized names of calculators sorted alphabetically:
names = ['abinit', 'ace', 'aims', 'amber', 'asap', 'castep', 'cp2k',
         'crystal', 'demon', 'demonnano', 'dftb', 'dftd3', 'dmol', 'eam',
         'elk', 'emt', 'espresso', 'exciting', 'ff', 'fleur', 'gamess_us',
         'gaussian', 'gpaw', 'gromacs', 'gulp', 'hotbit', 'kim',
         'lammpslib', 'lammpsrun', 'lj', 'mopac', 'morse', 'nwchem',
         'octopus', 'onetep', 'openmx', 'orca', 'psi4', 'qchem', 'siesta',
         'tip3p', 'tip4p', 'turbomole', 'vasp', 'reann','fireann','fireann_ef',"Equi_MPNN"]


special = {'cp2k': 'CP2K',
           'demonnano': 'DemonNano',
           'dftd3': 'DFTD3',
           'dmol': 'DMol3',
           'eam': 'EAM',
           'elk': 'ELK',
           'emt': 'EMT',
           'crystal': 'CRYSTAL',
           'ff': 'ForceField',
           'fleur': 'FLEUR',
           'gamess_us': 'GAMESSUS',
           'gulp': 'GULP',
           'kim': 'KIM',
           'lammpsrun': 'LAMMPS',
           'lammpslib': 'LAMMPSlib',
           'lj': 'LennardJones',
           'mopac': 'MOPAC',
           'morse': 'MorsePotential',
           'nwchem': 'NWChem',
           'openmx': 'OpenMX',
           'orca': 'ORCA',
           'qchem': 'QChem',
           'tip3p': 'TIP3P',
           'tip4p': 'TIP4P',
           'fireann': 'FIREANN',
           'fireann_ef': 'FIREANN_EF',
           'Equi_MPNN': 'Equi_MPNN',
           'reann': 'REANN'}


external_calculators = {}


def register_calculator_class(name, cls):
    """ Add the class into the database. """
    assert name not in external_calculators
    external_calculators[name] = cls
    names.append(name)
    names.sort()


def get_calculator_class(name):
    """Return calculator class."""
    if name == 'asap':
        from asap3 import EMT as Calculator
    elif name == 'gpaw':
        from gpaw import GPAW as Calculator
    elif name == 'hotbit':
        from hotbit import Calculator
    elif name == 'vasp2':
        from ase.calculators.vasp import Vasp2 as Calculator
    elif name == 'ace':
        from ase.calculators.acemolecule import ACE as Calculator
    elif name == 'Psi4':
        from ase.calculators.psi4 import Psi4 as Calculator
    elif name in external_calculators:
        Calculator = external_calculators[name]
    else:
        classname = special.get(name, name.title())
        module = __import__('ase.calculators.' + name, {}, None, [classname])
        Calculator = getattr(module, classname)
    return Calculator


def equal(a, b, tol=None, rtol=None, atol=None):
    """ndarray-enabled comparison function."""
    # XXX Known bugs:
    #  * Comparing cell objects (pbc not part of array representation)
    #  * Infinite recursion for cyclic dicts
    #  * Can of worms is open
    if tol is not None:
        msg = 'Use `equal(a, b, rtol=..., atol=...)` instead of `tol=...`'
        warnings.warn(msg, DeprecationWarning)
        assert rtol is None and atol is None, \
            'Do not use deprecated `tol` with `atol` and/or `rtol`'
        rtol = tol
        atol = tol

    a_is_dict = isinstance(a, dict)
    b_is_dict = isinstance(b, dict)
    if a_is_dict or b_is_dict:
        # Check that both a and b are dicts
        if not (a_is_dict and b_is_dict):
            return False
        if a.keys() != b.keys():
            return False
        return all(equal(a[key], b[key], rtol=rtol, atol=atol) for key in a)

    if np.shape(a) != np.shape(b):
        return False

    if rtol is None and atol is None:
        return np.array_equal(a, b)

    if rtol is None:
        rtol = 0
    if atol is None:
        atol = 0

    return np.allclose(a, b, rtol=rtol, atol=atol)


def kptdensity2monkhorstpack(atoms, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    atoms: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.
    """

    recipcell = atoms.cell.reciprocal()
    kpts = []
    for i in range(3):
        if atoms.pbc[i]:
            k = 2 * pi * sqrt((recipcell[i]**2).sum()) * kptdensity
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return np.array(kpts)


def kpts2mp(atoms, kpts, even=False):
    if kpts is None:
        return np.array([1, 1, 1])
    if isinstance(kpts, (float, int)):
        return kptdensity2monkhorstpack(atoms, kpts, even)
    else:
        return kpts


def kpts2sizeandoffsets(size=None, density=None, gamma=None, even=None,
                        atoms=None):
    """Helper function for selecting k-points.

    Use either size or density.

    size: 3 ints
        Number of k-points.
    density: float
        K-point density in units of k-points per Ang^-1.
    gamma: None or bool
        Should the Gamma-point be included?  Yes / no / don't care:
        True / False / None.
    even: None or bool
        Should the number of k-points be even?  Yes / no / don't care:
        True / False / None.
    atoms: Atoms object
        Needed for calculating k-point density.

    """

    if size is not None and density is not None:
        raise ValueError('Cannot specify k-point mesh size and '
                         'density simultaneously')
    elif density is not None and atoms is None:
        raise ValueError('Cannot set k-points from "density" unless '
                         'Atoms are provided (need BZ dimensions).')

    if size is None:
        if density is None:
            size = [1, 1, 1]
        else:
            size = kptdensity2monkhorstpack(atoms, density, None)

    # Not using the rounding from kptdensity2monkhorstpack as it doesn't do
    # rounding to odd numbers
    if even is not None:
        size = np.array(size)
        remainder = size % 2
        if even:
            size += remainder
        else:  # Round up to odd numbers
            size += (1 - remainder)

    offsets = [0, 0, 0]
    if atoms is None:
        pbc = [True, True, True]
    else:
        pbc = atoms.pbc

    if gamma is not None:
        for i, s in enumerate(size):
            if pbc[i] and s % 2 != bool(gamma):
                offsets[i] = 0.5 / s

    return size, offsets


@jsonable('kpoints')
class KPoints:
    def __init__(self, kpts=None):
        if kpts is None:
            kpts = np.zeros((1, 3))
        self.kpts = kpts

    def todict(self):
        return vars(self)


def kpts2kpts(kpts, atoms=None):
    from ase.dft.kpoints import monkhorst_pack

    if kpts is None:
        return KPoints()

    if hasattr(kpts, 'kpts'):
        return kpts

    if isinstance(kpts, dict):
        if 'kpts' in kpts:
            return KPoints(kpts['kpts'])
        if 'path' in kpts:
            cell = Cell.ascell(atoms.cell)
            return cell.bandpath(pbc=atoms.pbc, **kpts)
        size, offsets = kpts2sizeandoffsets(atoms=atoms, **kpts)
        return KPoints(monkhorst_pack(size) + offsets)

    if isinstance(kpts[0], int):
        return KPoints(monkhorst_pack(kpts))

    return KPoints(np.array(kpts))


def kpts2ndarray(kpts, atoms=None):
    """Convert kpts keyword to 2-d ndarray of scaled k-points."""
    return kpts2kpts(kpts, atoms=atoms).kpts


class EigenvalOccupationMixin:
    """Define 'eigenvalues' and 'occupations' properties on class.

    eigenvalues and occupations will be arrays of shape (spin, kpts, nbands).

    Classes must implement the old-fashioned get_eigenvalues and
    get_occupations methods."""

    @property
    def eigenvalues(self):
        return self.build_eig_occ_array(self.get_eigenvalues)

    @property
    def occupations(self):
        return self.build_eig_occ_array(self.get_occupation_numbers)

    def build_eig_occ_array(self, getter):
        nspins = self.get_number_of_spins()
        nkpts = len(self.get_ibz_k_points())
        nbands = self.get_number_of_bands()
        arr = np.zeros((nspins, nkpts, nbands))
        for s in range(nspins):
            for k in range(nkpts):
                arr[s, k, :] = getter(spin=s, kpt=k)
        return arr


class Parameters(dict):
    """Dictionary for parameters.

    Special feature: If param is a Parameters instance, then param.xc
    is a shorthand for param['xc'].
    """

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def read(cls, filename):
        """Read parameters from file."""
        # We use ast to evaluate literals, avoiding eval()
        # for security reasons.
        import ast
        with open(filename) as fd:
            txt = fd.read().strip()
        assert txt.startswith('dict(')
        assert txt.endswith(')')
        txt = txt[5:-1]

        # The tostring() representation "dict(...)" is not actually
        # a literal, so we manually parse that along with the other
        # formatting that we did manually:
        dct = {}
        for line in txt.splitlines():
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            if val[-1] == ',':
                val = val[:-1]
            dct[key] = ast.literal_eval(val)

        parameters = cls(dct)
        return parameters

    def tostring(self):
        keys = sorted(self)
        return 'dict(' + ',\n     '.join(
            '{}={!r}'.format(key, self[key]) for key in keys) + ')\n'

    def write(self, filename):
        pathlib.Path(filename).write_text(self.tostring())


class Calculator(GetPropertiesMixin):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

    implemented_properties: List[str] = []
    'Properties calculator can handle (energy, forces, ...)'

    default_parameters: Dict[str, Any] = {}
    'Default parameters'

    ignored_changes: Set[str] = set()
    'Properties of Atoms which we ignore for the purposes of cache '
    'invalidation with check_state().'

    discard_results_on_any_change = False
    'Whether we purge the results following any change in the set() method.  '
    'Most (file I/O) calculators will probably want this.'

    _deprecated = object()

    def __init__(self, restart=None, ignore_bad_restart_file=_deprecated,
                 label=None, atoms=None, directory='.',
                 **kwargs):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Deprecated, please do not use.
            Passing more than one positional argument to Calculator()
            is deprecated and will stop working in the future.
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters
        self._directory = None  # Initialize

        if ignore_bad_restart_file is self._deprecated:
            ignore_bad_restart_file = False
        else:
            warnings.warn(FutureWarning(
                'The keyword "ignore_bad_restart_file" is deprecated and '
                'will be removed in a future version of ASE.  Passing more '
                'than one positional argument to Calculator is also '
                'deprecated and will stop functioning in the future.  '
                'Please pass arguments by keyword (key=value) except '
                'optionally the "restart" keyword.'
            ))

        if restart is not None:
            try:
                self.read(restart)  # read parameters, atoms and results
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

        self.directory = directory
        self.prefix = None
        if label is not None:
            if self.directory == '.' and '/' in label:
                # We specified directory in label, and nothing in the diretory key
                self.label = label
            elif '/' not in label:
                # We specified our directory in the directory keyword
                # or not at all
                self.label = '/'.join((self.directory, label))
            else:
                raise ValueError('Directory redundantly specified though '
                                 'directory="{}" and label="{}".  '
                                 'Please omit "/" in label.'
                                 .format(self.directory, label))

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                        (atoms.pbc == self.atoms.pbc).all()):
                    raise CalculatorError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell

        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

        if not hasattr(self, 'get_spin_polarized'):
            self.get_spin_polarized = self._deprecated_get_spin_polarized

    @property
    def directory(self) -> str:
        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, pathlib.PurePath]):
        self._directory = str(pathlib.Path(directory))  # Normalize path.

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        # Generally, label ~ directory/prefix
        #
        # We use '/' rather than os.pathsep because
        #   1) directory/prefix does not represent any actual path
        #   2) We want the same string to work the same on all platforms
        if self.prefix is None:
            return self.directory + '/'

        return '{}/{}'.format(self.directory, self.prefix)

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix

    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')
        * label=None: (directory='.', prefix=None)
        """
        self.label = label

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}

    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of succes, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)

    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        if self.discard_results_on_any_change and changed_parameters:
            self.reset()
        return changed_parameters

    def check_state(self, atoms, tol=1e-15):
        """Check for any system changes since last calculation."""
        return compare_atoms(self.atoms, atoms, tol=tol,
                             excluded_properties=set(self.ignored_changes))

    def get_potential_energy(self, atoms=None, force_consistent=False):
        energy = self.get_property('energy', atoms)
        if force_consistent:
            if 'free_energy' not in self.results:
                name = self.__class__.__name__
                # XXX but we don't know why the energy is not there.
                # We should raise PropertyNotPresent.  Discuss
                raise PropertyNotImplementedError(
                    'Force consistent/free energy ("free_energy") '
                    'not provided by {0} calculator'.format(name))
            return self.results['free_energy']
        else:
            return energy

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError('{} property not implemented'
                                              .format(name))

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms, [name], system_changes)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError('{} not present in this '
                                              'calculation'.format(name))

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def calculation_required(self, atoms, properties):
        assert not isinstance(properties, str)
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute and create any missing
        directories.
        """

        if atoms is not None:
            self.atoms = atoms.copy()
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

    def calculate_numerical_forces(self, atoms, d=0.001):
        """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""

        from ase.calculators.test import numeric_force
        return np.array([[numeric_force(atoms, a, i, d)
                          for i in range(3)] for a in range(len(atoms))])

    def calculate_numerical_stress(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)

        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress

    def _deprecated_get_spin_polarized(self):
        msg = ('This calculator does not implement get_spin_polarized().  '
               'In the future, calc.get_spin_polarized() will work only on '
               'calculator classes that explicitly implement this method or '
               'inherit the method via specialized subclasses.')
        warnings.warn(msg, FutureWarning)
        return False

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure
        # XXX This calculator is supposed to just have done a band structure
        # calculation, but the calculator may not have the correct Fermi level
        # if it updated the Fermi level after changing k-points.
        # This will be a problem with some calculators (currently GPAW), and
        # the user would have to override this by providing the Fermi level
        # from the selfconsistent calculation.
        return get_band_structure(calc=self)

    def calculate_properties(self, atoms, properties):
        """This method is experimental; currently for internal use."""
        for name in properties:
            if name not in all_outputs:
                raise ValueError(f'No such property: {name}')

        # We ignore system changes for now.
        self.calculate(atoms, properties, system_changes=all_changes)

        props = self.export_properties()

        for name in properties:
            if name not in props:
                raise PropertyNotPresent(name)
        return props

    def export_properties(self):
        return Properties(self.results)


class FileIOCalculator(Calculator):
    """Base class for calculators that write/read input/output files."""

    command: Optional[str] = None
    'Command used to start calculation'

    def __init__(self, restart=None,
                 ignore_bad_restart_file=Calculator._deprecated,
                 label=None, atoms=None, command=None, **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise CalculatorSetupError(
                'Please set ${} environment variable '
                .format('ASE_' + self.name.upper() + '_COMMAND') +
                'or supply the command keyword')
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', self.prefix)

        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            # Actually this may never happen with shell=True, since
            # probably the shell launches successfully.  But we soon want
            # to allow calling the subprocess directly, and then this
            # distinction (failed to launch vs failed to run) is useful.
            msg = 'Failed to execute "{}"'.format(command)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Calculator "{}" failed with command "{}" failed in '
                   '{} with error code {}'.format(self.name, command,
                                                  path, errorcode))
            raise CalculationFailed(msg)

        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        absdir = os.path.abspath(self.directory)
        if absdir != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
        pass

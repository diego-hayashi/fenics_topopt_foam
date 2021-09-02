"""
<!-- The Markdown files are imported here to serve as the main documentation for pdoc to use -->
.. include:: ../README.md

<details>
<summary>License</summary>
```text
.. include:: ../COPYING
```
</details>

.. include:: ../scripts/docs/aux_files/DOC_VERSION.md

<details>
<summary>Module description</summary>

```text
 ════════════════════════════ FEniCS TopOpt Foam ══════════════════════════════

 ──────────────────────────────── Description ─────────────────────────────────

Topology optimization combining OpenFOAM and FEniCS/dolfin-adjoint.

 ─────────────────────────────────── Usage ────────────────────────────────────

This interface can be included by:

.. code-block:: python

  from fenics_topopt_foam import *

  or

.. code-block:: python

  import fenics_topopt_foam

 ────────────────────────────────── License ───────────────────────────────────

FEniCS TopOpt Foam is licensed under the GNU General Public License (GPL), version 3.

FEniCS TopOpt Foam is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or 
(at your option) any later version.

FEniCS TopOpt Foam is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
for more details.

You should have received a copy of the GNU General Public License 
along with FEniCS TopOpt Foam. If not, see <https://www.gnu.org/licenses/>.

 ──────────────────────────────────────────────────────────────────────────────
```
</details>
"""

#
# Copyright (C) 2020-2021 Diego Hayashi Alonso
#
# This file is part of FEniCS TopOpt Foam.
# 
# FEniCS TopOpt Foam is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FEniCS TopOpt Foam is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FEniCS TopOpt Foam. If not, see <https://www.gnu.org/licenses/>.
#

############################## About information ###############################

from .__about__ import __name__, __version__, __description__ , __author__, __maintainer__

########################### Additional information #############################

# Path to the folder in which the code is located
import os
__fenics_topopt_foam_folder__ = os.path.dirname(__file__)

####################### Default available functionality ########################

# Some useful functions/classes to import when using:
 # "from fenics_topopt_foam import * " and "import fenics_topopt_foam"

__all__ = [
	'FoamSolver',				# FoamSolver
	'FEniCSFoamSolver',			# FEniCS interface for the FoamSolver
	'convertToFoamUnitSpecification',	# Conversion to the OpenFOAM unit specification
	'assignSubFunctionsToFunction',		# Some useful function one may want to use together with the FEniCS interface
	'setPrintLevel',			# Set the print level for this code
	'createFEniCSMeshFromOpenFOAM',		# If you want to use a mesh from OpenFOAM in FEniCS
	'createProblemFolderFromMesh',		# If you want to create a problem folder with a mesh from a 'polyMesh' folder
	'compileLibraryFoldersIfNeeded',	# If you want are using user-defined libraries, and want FEniCS TopOpt Foam to check if the 
						  # you are using the latest compilation of your source files.
	'SuccessiveSimulationParameter',	# If you want to set successive simulations
	'dolfin_adjoint_extensions',		# Some extensions for dolfin-adjoint
]

from .solver.FoamSolver import FoamSolver
from .solver.FEniCSFoamSolver import FEniCSFoamSolver
from .utils.utils import convertToFoamUnitSpecification, setPrintLevel, compileLibraryFoldersIfNeeded
from .utils.utils_fenics import assignSubFunctionsToFunction
from .utils.SuccessiveSimulationParameter import SuccessiveSimulationParameter
from .mesh.createFEniCSMeshFromOpenFOAM import createFEniCSMeshFromOpenFOAM
from .mesh.createProblemFolderFromMesh import createProblemFolderFromMesh
from . import dolfin_adjoint_extensions

###################### Adjustment for the documentation ########################
# Let's leave only the "really basic" information in the main page. The "advanced"
# information can still be viewed through the other pages, though.

__pdoc__ = {}
__to_hide_in_the_main_page_of_the_docs__ = [
	{
		'class' : SuccessiveSimulationParameter,
		'hide' : 'all',
	},
	{
		'class' : FEniCSFoamSolver,
		'hide' : [
			'optimizeMeshForOpenFOAM',
			'prepareFunctionSpacesAndMaps',
			'postProcessFEniCSVariableFromFoam',
			'postProcessFEniCSVariableToFoam',
			'getFoamVectorBoundaryValuesInFEniCS',
			'getFoamVectorFromName',
			'getFoamMeasurementUnit',
			'setFoamConfiguration',
			'setFoamProperty',
			'unset*',
		],
	},
	{
		'class' : FoamSolver,
		'hide' : [
			'prepareForEditing',
			'removeTimeStepFolders',
			'plotResidualsFromLog',
			'setToSaveResidualsToFile',
			'saveResidualsFromLog',
			'getFoamConfigurations',
			'getFoamProperties',
			'getFoamVectors',
			'prepareNewFoamVector',
			'createFoamVectorFromFile',
			'computeTestUtility',
			'plotAllResidualsFromLog',
			'unsetFromRunInParallel',
			'computeField',
			'computeCustomUtility',
			'exportMesh',
			'unset*',
		],
	},

]

from .utils.utils import __remove_some_methods_from_doc
__remove_some_methods_from_doc(__pdoc__, __to_hide_in_the_main_page_of_the_docs__)



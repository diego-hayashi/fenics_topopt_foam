################################################################################
#                         UncoupledLinearVariationalSolver                     #
################################################################################

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

############################## FEniCS libraries ################################

from fenics import *
from dolfin_adjoint import *

#### UFL
def checkLibrary(lib_name): # Check if a library exists.
	import importlib.util
	return type(importlib.util.find_spec(lib_name)).__name__ != 'NoneType'
if checkLibrary('ufl'):
	assert not checkLibrary('ufl_legacy'), " ❌️ ERROR: Sorry, you can only have one of them installed: ufl or ufl_legacy"
	import ufl
else:
	assert checkLibrary('ufl_legacy'), " ❌️ ERROR: Sorry, you need to have one of them installed: ufl or ufl_legacy"
	import ufl_legacy as ufl

import fenics
import dolfin_adjoint

from pyadjoint.tape import annotate_tape, get_working_tape, no_annotations

###################### UncoupledLinearVariationalSolveBlock ####################

if dolfin_adjoint.__version__ == '2019.1.0':
	dolfin_adjoint_SolveBlock = dolfin_adjoint.solving.SolveBlock
else: # > 2019.1.0
	dolfin_adjoint_SolveBlock = dolfin_adjoint.blocks.SolveVarFormBlock

class UncoupledLinearVariationalSolveBlock(dolfin_adjoint_SolveBlock):
	"""
	Overload for the UncoupledLinearVariationalSolver to work with dolfin-adjoint. [dolfin-adjoint 2019.1.0]
	Check out: /opt/Fenics/2019.1.0/lib/python3.6/site-packages/fenics_adjoint/solving.py
	"""

	def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
		"""
		Solve the forward model (i.e., the simulation).
		"""

		if 'simulation_solver' not in self.__dict__ or type(self.simulation_solver).__name__ == 'NoneType':

			# Default behavior
			fenics.solve(lhs == rhs, func, bcs, **kwargs)

		else:

			# [🚩] Use the external simulation solver
			self.simulation_solver.solve(replace_map = self.BAK_replace_map)

		return func

	def _replace_form(self, form, func=None):
		"""
		This function is used by dolfin-adjoint to create a copy of the 
		weak form with copied dependencies. Therefore, the solver
		does not receive the original dependencies of the equations,
		which would make it hard for us to update the current 
		values of the design variable in the weak form. Therefore,
		we should save the replace map.
		"""

		self.BAK_func_state = {} # [🚩] Some auxiliary dictionary we may need to use below

		replace_map = {}
		for block_variable in self.get_dependencies():
			c = block_variable.output
			if c in form.coefficients():
				c_rep = block_variable.saved_output
				if c != c_rep:
					replace_map[c] = c_rep
					if func is not None and c == self.func: # It substitutes everything EXCEPT for the state variable!
						backend.Function.assign(func, c_rep)
						replace_map[c] = func
						self.BAK_func_state = {c : func} # [🚩] Let's save the state variable

		self.BAK_replace_map = {key:replace_map[key] for key in replace_map} # [🚩] Create a copy of the replace map for using later
		self.BAK_replace_map.update(self.BAK_func_state) # [🚩] Adding the state variable to the copy of the replace_map

		return ufl.replace(form, replace_map)

	def setTheSimulationSolver(self, simulation_solver):
		"""
		[🚩] Save the simulation solver for using later.
		"""
		self.simulation_solver = simulation_solver

###################### UncoupledLinearVariationalSolver ########################

class UncoupledLinearVariationalSolver(LinearVariationalSolver):
	"""
	LinearVariationalSolver for uncoupling simulation and optimization. [dolfin-adjoint 2019.1.0]
	Based on: 
		https://bitbucket.org/dolfin-adjoint/pyadjoint/src/master/fenics_adjoint/variational_solver.py
	"""

	@no_annotations
	def __init__(self, *args, **kwargs):

		# [🚩] Simulation solver
		self.simulation_solver = kwargs.pop('simulation_solver', None)

		# Original initialization
		super().__init__(*args, **kwargs)

	def solve(self, *args, **kwargs):

		annotate = annotate_tape(kwargs)
		if annotate:
			tape = get_working_tape()
			problem = self._ad_problem
			sb_kwargs = LinearVariationalSolveBlock.pop_kwargs(kwargs)
			sb_kwargs.update(kwargs)
			block = UncoupledLinearVariationalSolveBlock(problem._ad_a == problem._ad_L,
								problem._ad_u,
								problem._ad_bcs,
								problem_args=problem._ad_args,
								problem_kwargs=problem._ad_kwargs,
								solver_params=self.parameters,
								solver_args=self._ad_args,
								solver_kwargs=self._ad_kwargs,
								solve_args=args,
								solve_kwargs=kwargs,
								**sb_kwargs)

			# [🚩] Setting the simulation solver
			block.setTheSimulationSolver(self.simulation_solver) 

			tape.add_block(block)

		with stop_annotating():

			# [🚩] Use the external simulation solver
			out = self.simulation_solver.solve()

			#out = super(LinearVariationalSolver, self).solve(*args, **kwargs)

		if annotate:
			block.add_output(self._ad_problem._ad_u.create_block_variable())

		return out

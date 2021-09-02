################################################################################
#                  getWallDistanceAndNormalVectorFromDolfinAdjoint             #
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
import ufl

############### getWallDistanceAndNormalVectorFromDolfinAdjoint ################

def getWallDistanceAndNormalVectorFromDolfinAdjoint(l_wall, function_space, function_space_vec, replace_map = None, domain_type = '2D', tol_var = 1.E-15):
	"""
	Get the wall distance computed in dolfin-adjoint (from the checkpoint 
	value), and compute the corresponding normal vector.
	"""

	############################# Dependencies #############################

	# Get the current dependencies
	if type(replace_map).__name__ == 'NoneType' or len(replace_map) == 0:
		pass
	else:

		def get_all_ufl_operands_recursive(l_wall):
			if type(l_wall).__module__ == 'ufl.algebra':
				l_wall_ufl_operands = l_wall.ufl_operands

				coefs_l_wall = []
				for coef_wall in l_wall_ufl_operands:
					coefs_l_wall += get_all_ufl_operands_recursive(coef_wall) # Recursão

			elif type(l_wall).__module__ in ['fenics_adjoint.types.function', 'dolfin.function.function']:
				coefs_l_wall = [l_wall]
			else:
				coefs_l_wall = []

			return coefs_l_wall

		coefs_l_wall = get_all_ufl_operands_recursive(l_wall)

		l_wall_replace_map = {}
		for i in range(len(coefs_l_wall)):
			if coefs_l_wall[i] in replace_map:
				l_wall_replace_map[coefs_l_wall[i]] = replace_map[coefs_l_wall[i]]

		assert len(l_wall_replace_map) > 0

		l_wall_dep_orig_arrays = {}

		for l_wall_dep in l_wall_replace_map:

			l_wall_dep_orig_array = l_wall_dep.vector().get_local()
			l_wall_dep_orig_arrays[l_wall_dep] = l_wall_dep_orig_array

			checkpoint_l_wall_dep = l_wall_dep.block_variable.checkpoint
			if type(checkpoint_l_wall_dep).__name__ == 'NoneType':
				pass # No value computed by dolfin-adjoint
			else:
				l_wall_dep.vector().set_local(checkpoint_l_wall_dep.vector().get_local())
				l_wall_dep.vector().apply('insert')

	######################### Current wall distance ########################

	l_wall_projected = project(l_wall + tol_var, function_space, form_compiler_parameters = {'quadrature_degree' : None})

	############################ Normal to walls ###########################

	# Domain multiplier and differential operator
	if domain_type == '2D axisymmetric':
		(r_, z_) = SpatialCoordinate(mesh); nr = 0; nz = 1
		grad_of_scalar = lambda a: as_tensor([Dx(a, nr), 0, Dx(a, nz)])
	else:
		grad_of_scalar = lambda a: grad(a)

	# Magnitude operator
	mag = lambda a: (inner(a, a))**.5

	# ∇ℓ_wall
	grad_l_wall = grad_of_scalar(l_wall_projected) 

	# |∇ℓ_wall|
	mag_grad_l_wall = mag(grad_l_wall)

	# Small tolerance for when |∇ℓ_wall| == 0
	tol_vec = as_vector([tol_var for i in range(len(grad_l_wall))])

	# n_vetor = -∇ℓ_wall/|∇ℓ_wall|
	normal_to_walls_vec = -(grad_l_wall + tol_vec)/(mag_grad_l_wall + tol_var)
	normal_to_walls = project(normal_to_walls_vec, function_space_vec, form_compiler_parameters = {'quadrature_degree' : None})

	############################# Dependencies #############################

	# Restore the previous dependencies
	if type(replace_map).__name__ == 'NoneType' or len(replace_map) == 0:
		pass
	else:
		for l_wall_dep in l_wall_dep_orig_arrays:

			l_wall_dep_orig_array = l_wall_dep_orig_arrays[l_wall_dep]

			checkpoint_l_wall_dep = l_wall_dep.block_variable.checkpoint
			if type(checkpoint_l_wall_dep).__name__ == 'NoneType':
				pass # No value computed by dolfin-adjoint
			else:
				l_wall_dep.vector().set_local(l_wall_dep_orig_array)
				l_wall_dep.vector().apply('insert')

	######################### Final adjustments ############################

	# Avoid negative values that may arise from numerical precision
	l_wall_projected_array = l_wall_projected.vector().get_local()
	l_wall_projected_array[l_wall_projected_array < tol_var] = tol_var
	l_wall_projected.vector().set_local(l_wall_projected_array)
	l_wall_projected.vector().apply('insert')

	return l_wall_projected, normal_to_walls



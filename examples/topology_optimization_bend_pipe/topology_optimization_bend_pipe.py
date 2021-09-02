################################################################################
#           Bend pipe topology optimization with FEniCS TopOpt Foam            #
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

# Necessary imports
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from fenics import * 
import ufl
from dolfin_adjoint import * 
import pyadjoint
from pyadjoint.tape import no_annotations
import mpi4py

# Some flags for FEniCS
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters['allow_extrapolation'] = True # Allow small numerical differences in the boundary definition.

# Quadrature degree in FEniCS  (sometimes, the "automatic" determination of the quadrature degree becomes excessively high, meaning that it should be manually reduced)
parameters['form_compiler']['quadrature_degree'] = 5 

# FEniCS TopOpt Foam imports
import fenics_topopt_foam
from fenics_topopt_foam.dolfin_adjoint_extensions import UncoupledNonlinearVariationalSolver, getWallDistanceAndNormalVectorFromDolfinAdjoint

# Run OpenFOAM in parallel?
run_openfoam_in_parallel = False

# Fluid flow setup
rho_ = 1.0; mu_ = 0.1 # Density and dynamic viscosity
width_inlet_outlet = 1.0/5.0 # Inlet/outlet width
x_min = 0.0; x_max = 1.0 # x dimensions
y_min = 0.0; y_max = 1.0 # y dimensions
v_max_inlet = 1.0       # Inlet velocity
nu_T_aux_inlet = 0.0001 # Inlet turbulent variable value
flow_regime = 'laminar' # Flow regime: 'laminar' or 'turbulent (Spalart-Allmaras)'

# Topology optimization setup
k_max = 1.E4*mu_; k_min = 0.0; q = 0.1
k = lambda alpha : k_max + (k_min - k_max) * alpha * (1. + q) / (alpha + q)
gamma_max = 1.E3; gamma_min = 0.0
gamma = lambda alpha : gamma_max + (gamma_min - gamma_max) * alpha * (1. + q) / (alpha + q)
lambda_kappa_v = 1.0
f_V = 0.3 # Volume fraction

# Output folders
output_folder = 'output'
problem_folder = "%s/foam_problem" %(output_folder)
if (MPI.comm_world.Get_size() == 1) or (MPI.comm_world.Get_rank() == 0):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder) # Create the output folder if it still does not exist

# Create the 2D mesh and plot it
N_mesh = 50
delta_x = x_max - x_min; delta_y = y_max - y_min
mesh = RectangleMesh(Point(x_min, y_min), Point(x_max, y_max), int(N_mesh*delta_x/delta_y), N_mesh, diagonal = "crossed")
File('%s/mesh.pvd' %(output_folder)) << mesh

# Function spaces -> MINI element (2D)
V1_element =  FiniteElement('Lagrange', mesh.ufl_cell(), 1)
B_element = FiniteElement('Bubble', mesh.ufl_cell(), 3)
V_element = VectorElement(NodalEnrichedElement(V1_element, B_element)) # Velocity
P_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1) # Pressure
if flow_regime == 'turbulent (Spalart-Allmaras)':
	NU_T_AUX_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1) # Turbulent variable
if flow_regime == 'laminar':
	U_element = MixedElement([V_element, P_element])
elif flow_regime == 'turbulent (Spalart-Allmaras)':
	U_element = MixedElement([V_element, P_element, NU_T_AUX_element])
U = FunctionSpace(mesh, U_element) # Mixed function space
A_element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
A = FunctionSpace(mesh, A_element) # Design variable function space (nodal)

# Prepare the boundary definition
class Inlet(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and x[0] == x_min and ((y_min + 4.0/5*delta_y - width_inlet_outlet/2) < x[1] < (y_min + 4.0/5*delta_y + width_inlet_outlet/2))
class Outlet(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and x[1] == y_min and ((x_min + 4.0/5*delta_x - width_inlet_outlet/2) < x[0] < (x_min + 4.0/5*delta_x + width_inlet_outlet/2))
class Walls(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary # * It will be set before the other boundaries
marker_numbers = {'unset' : 0, 'wall' : 1, 'inlet' : 2, 'outlet' : 3}
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(marker_numbers['unset'])
Walls().mark(boundary_markers, marker_numbers['wall'])
Inlet().mark(boundary_markers, marker_numbers['inlet'])
Outlet().mark(boundary_markers, marker_numbers['outlet'])
File("%s/markers.pvd" %(output_folder)) << boundary_markers

# Boundary values (for Dirichlet Boundary conditions)
class InletVelocity(UserExpression):
	def eval(self, values, x):
		for i in range(len(values)):
			values[i] = 0.0 # Initialize all values with zeros
		if x[0] == x_min and (4.0/5*delta_y - width_inlet_outlet/2) < x[1] < (4.0/5*delta_y + width_inlet_outlet/2):
			y_local = x[1] - 4.0/5*delta_y; values[0] = v_max_inlet*(1 - (2*y_local/width_inlet_outlet)**2)
	def value_shape(self):
		return (2,)
inlet_velocity_expression = InletVelocity(element = V_element)
wall_velocity_value = Constant((0,0))
if flow_regime == 'turbulent (Spalart-Allmaras)':
	inlet_nu_T_aux_value = Constant(nu_T_aux_inlet)

# Function to set FEniCS TopOpt Foam
@no_annotations
def prepareFEniCSFoamSolverWithUpdate(u, alpha, mesh, boundary_markers, marker_numbers, bcs):

	# Gather the boundary data
	boundary_data = {
		'mesh_function' : boundary_markers,
		'mesh_function_tag_to_boundary_name' : {value : key for key, value in marker_numbers.items()}, # Invert dictionary
		'boundaries' : {
			'wall'   : {'type' : 'wall', 'inGroups' : ['wall']},
			'inlet'  : {'type' : 'patch',},
			'outlet' : {'type' : 'patch',},
		},
	}

	# Parameters for fenics_topopt_foam.FoamSolver
	foam_parameters = {
		'domain type' : '2D',  # Domain type according to what the model implemented in FEniCS
		'error_on_nonconvergence' : False,
		'problem_folder' : problem_folder, 
		'solver' : {
			'type' : 'custom',
			'openfoam' : {'name' : 'simpleFoam'},
			'custom' : {
				'name' : 'CustomSimpleFoam', # simpleFoam with material model
				'location' : '%s/cpp_openfoam_modules/foam_cpp_solvers' %(os.path.dirname(fenics_topopt_foam.__file__)), 
			},
		},
		'compile_modules_if_needed' : True,
	}

	# Configurations for OpenFOAM
	foam_configurations_dictionary = {
		'controlDict' : {
			'application' :         foam_parameters['solver'][foam_parameters['solver']['type']]['name'],
			'startFrom' :           'startTime',
			'startTime' :           0,
			'stopAt' :              'endTime',
			'endTime' :             2000,
			'deltaT' :              1, 
			'writeControl' :        'timeStep',
			'writeInterval' :       100,
			'purgeWrite' :           2,
			'writeFormat' :         'ascii',
			'writePrecision' :      12,
			'writeCompression' :    'off',
			'timeFormat' :          'general',
			'timePrecision' :       6, 
			'graphFormat' :         'raw',  
			'runTimeModifiable' :   'true', 
		},
		'fvSchemes' : {
			'ddtSchemes' : {
				'default' : 'steadyState',
			},
			'gradSchemes' : {
				'default' : ('Gauss', 'linear'),
				'grad(nuTilda)' : ('cellLimited', 'Gauss', 'linear', '1.0'),
			},
			'divSchemes' : {
				'default' : 'none',
				'div(phi,U)' : ('bounded', 'Gauss', 'linearUpwind', 'grad(U)'),
				'div(phi,nuTilda)' : ('bounded', 'Gauss', 'limitedLinear', 1),
				'div((nuEff*dev2(T(grad(U)))))' : ('Gauss', 'linear'),
				'div(nonlinearStress)' : ('Gauss', 'linear'),
			},
			'laplacianSchemes' : {
				'default' : ('Gauss', 'linear', 'corrected'),
			},
			'interpolationSchemes' : {
				'default' : 'linear',
			},
			'snGradSchemes' : {
				'default' : 'corrected',
			},
			'wallDist' : {
				'method' : 'Custom_externalImport',
			},
		},
		'fvSolution' : {
			'solvers' : {
				'p' : {
					'solver' :         'GAMG', 
					'tolerance' :       1.E-06, 
					'relTol':           0.1, 
					'maxIter' :         1000, 
					'preconditioner' : 'none', 
					'smoother' : 	    'GaussSeidel', 
				},
				'U' : {

					'type' :           'segregated',
					'solver' :         'smoothSolver',
					'tolerance' :       1e-05,
					'relTol' :          0.1,
					'maxIter' :         1000, 
					'preconditioner' : 'none', 
					'smoother' :       'symGaussSeidel',
					'nSweeps' :         2,
				},
				'nuTilda' : {
					'solver' : 	    'smoothSolver', 
					'tolerance' : 	     1e-05,
					'relTol' :	     0.1,
					'maxIter' :         1000, 
					'preconditioner' : 'none',
					'smoother' :       'symGaussSeidel', 
					'nSweeps' :         2,
				},
			},
			'SIMPLE' : { 
				'nNonOrthogonalCorrectors' : 3,
				'consistent' : 'yes',
				'residualControl' : {
					'p' : 	    1e-2,
					'U' : 	    1e-3,
					'nuTilda' : 1e-3,
				},

			},

			'relaxationFactors' : {
				'fields' : {
					'p' :       0.9, 
					'U' :       0.9,
					'nuTilda' : 0.9,
					'".*"' :    0.9, 
				},
				'equations' : {
					'p' :       0.9, 
					'U' :       0.9,
					'nuTilda' : 0.9,
					'".*"' :    0.9, 
				},
			},
		},
	}

	# Set to use the libs provided in FEniCS TopOpt Foam in OpenFOAM
	libs_folder = '%s/cpp_openfoam_modules/foam_cpp_libs' %(os.path.dirname(fenics_topopt_foam.__file__))
	lib_names = fenics_topopt_foam.compileLibraryFoldersIfNeeded(libs_folder)
	lib_names_string = '('
	for i in range(len(lib_names)):
		if i != 0: lib_names_string += ' '
		lib_names_string += "\"%s\"" %(lib_names[i])
	lib_names_string += ')'
	foam_configurations_dictionary['controlDict']['libs'] = lib_names_string

	# Solver that interacts with FEniCS/dolfin-adjoint and OpenFOAM
	class FEniCSFoamSolverWithUpdate():
		def __init__(self, u, alpha):

			# Setups and initializations
			self.u = u; self.u.vector().apply('insert'); self.u_array_copy = self.u.vector().get_local()
			self.alpha = alpha
			self.fenics_foam_solver = fenics_topopt_foam.FEniCSFoamSolver(
				mesh, boundary_data, 
				foam_parameters, self.getPropertiesDictionary(), 
				foam_configurations_dictionary, 
				use_mesh_from_foam_solver = False,
				python_write_precision = foam_configurations_dictionary['controlDict']['writePrecision'],
				configuration_of_openfoam_measurement_units = {'pressure' : 'rho-normalized pressure'},
				)
			if flow_regime == 'turbulent (Spalart-Allmaras)':
				self.fenics_foam_solver.initFoamVector('nut', 'volScalarField', skip_if_exists = True)
				self.fenics_foam_solver.initFoamVector('nuTilda', 'volScalarField', skip_if_exists = True) 
				self.fenics_foam_solver.initFoamVector('yWall_to_load', 'volScalarField', skip_if_exists = True)
				self.fenics_foam_solver.initFoamVector('nWall_to_load', 'volVectorField', skip_if_exists = True)
				self.flag_set_boundary_values = True
			self.setBoundaryConditionsForOpenFOAM()

			# Set to run OpenFOAM in parallel
			if run_openfoam_in_parallel == True:
				parallel_data = { 
					'numberOfSubdomains' : 2,
					'method' : 'simple',
					'simpleCoeffs' : { 
						'n' : '(2 1 1)',
						'delta' : 0.001,
					},
				}
				self.fenics_foam_solver.foam_solver.setToRunInParallel(parallel_data)

		def setBoundaryConditionsForOpenFOAM(self):
			self.fenics_foam_solver.setAdditionalProperty('rho', rho_) # rho is necessary to convert 'rho-normalized pressure' (used in OpenFOAM) to 'pressure' (used here)
			self.fenics_foam_solver.setFoamBoundaryCondition('U', 'outlet', 'zeroGradient', None)
			self.fenics_foam_solver.setFoamBoundaryCondition('U', 'inlet', 'fixedValue', inlet_velocity_expression)
			self.fenics_foam_solver.setFoamBoundaryCondition('U', 'wall', 'noSlip', None) 
			self.fenics_foam_solver.setFoamBoundaryCondition('p', 'outlet', 'fixedValue', np.array([0.0], dtype = 'float'))
			self.fenics_foam_solver.setFoamBoundaryCondition('p', 'inlet', 'zeroGradient', None)
			self.fenics_foam_solver.setFoamBoundaryCondition('p', 'wall', 'zeroGradient', None)
			if flow_regime == 'turbulent (Spalart-Allmaras)':
				self.fenics_foam_solver.setFoamBoundaryCondition('nuTilda', 'inlet', 'fixedValue', inlet_nu_T_aux_value)
				self.fenics_foam_solver.setFoamBoundaryCondition('nuTilda', 'wall', 'fixedValue', np.array([0]))
				self.fenics_foam_solver.setFoamBoundaryCondition('nuTilda', 'outlet', 'zeroGradient', None)
				self.fenics_foam_solver.setFoamBoundaryCondition('nut', 'inlet', 'calculated', np.array([0]))
				self.fenics_foam_solver.setFoamBoundaryCondition('nut', 'outlet', 'calculated', np.array([0]))
				self.fenics_foam_solver.setFoamBoundaryCondition('nut', 'wall', 'calculated', np.array([0]))

		def getPropertiesDictionary(self):
			if flow_regime == 'laminar':
				simulationType = 'laminar'
				turbulence_switch = 'off'
			elif flow_regime == 'turbulent (Spalart-Allmaras)':
				simulationType = 'RAS'
				turbulence_switch = 'on'
			foam_properties_dictionary = {
				'materialmodelProperties' : {
					'q_penalization' : (fenics_topopt_foam.convertToFoamUnitSpecification('dimensionless'), q),
					'k_max' : (fenics_topopt_foam.convertToFoamUnitSpecification('inverse permeability'), k_max),
					'k_min' : (fenics_topopt_foam.convertToFoamUnitSpecification('inverse permeability'), k_min),
					'rho_density' : (fenics_topopt_foam.convertToFoamUnitSpecification('density'), rho_), # * This is because we will need to divide k(alpha) by the density inside CustomSimpleFoam!
				},
				'turbulenceProperties' : {
					'simulationType' : simulationType, 
					'RAS' : {
						'turbulence' : turbulence_switch,
						'RASModel' : 'CustomSpalartAllmaras',
						'CustomSpalartAllmaras' : {
							'lambda_v_design' : lambda_kappa_v,
						},
						'printCoeffs' : 'on',
					},
				},
				'transportProperties' : {
					'transportModel' : 'Newtonian',
					'nu' : (fenics_topopt_foam.convertToFoamUnitSpecification('kinematic viscosity'), mu_/rho_),
				},
			}
			return foam_properties_dictionary

		@no_annotations
		def plotResults(self, *args, **kwargs):
			self.fenics_foam_solver.plotResults(*args, **kwargs)

		@no_annotations
		def solve(self, replace_map = {}):

			# Get variables from replace_map and load the initial guess for the simulation
			if type(replace_map).__name__ == 'NoneType' or len(replace_map) == 0:
				u = self.u; alpha = self.alpha
			else:
				u = replace_map[self.u]; alpha = replace_map[self.alpha]
			u.vector().set_local(self.u_array_copy); u.vector().apply('insert')

			# Wall distance in FEniCS
			if flow_regime == 'turbulent (Spalart-Allmaras)':

				global l_wall
				L_WALL = FunctionSpace(mesh, 'Lagrange', 1)
				N_WALL = VectorFunctionSpace(mesh, 'Lagrange', 1)
				(l_wall_projected, normal_a_paredes) = getWallDistanceAndNormalVectorFromDolfinAdjoint(l_wall, L_WALL, N_WALL, domain_type = '2D', replace_map = replace_map)
				self.fenics_foam_solver.setFEniCSFunctionToFoamVector(l_wall_projected, foam_variable_name = 'yWall_to_load')
				self.fenics_foam_solver.setFEniCSFunctionToFoamVector(normal_a_paredes, foam_variable_name = 'nWall_to_load')

				if self.flag_set_boundary_values == True:
					self.fenics_foam_solver.setFoamBoundaryCondition('yWall_to_load', 'wall', 'fixedValue', l_wall_projected)
					self.fenics_foam_solver.setFoamBoundaryCondition('nWall_to_load', 'wall', 'fixedValue', normal_a_paredes)
					self.fenics_foam_solver.setFoamBoundaryCondition('yWall_to_load', 'inlet', 'zeroGradient', None)
					self.fenics_foam_solver.setFoamBoundaryCondition('nWall_to_load', 'inlet', 'zeroGradient', None)
					self.fenics_foam_solver.setFoamBoundaryCondition('yWall_to_load', 'outlet', 'zeroGradient', None)
					self.fenics_foam_solver.setFoamBoundaryCondition('nWall_to_load', 'outlet', 'zeroGradient', None)
					self.flag_set_boundary_values = False

			# Update properties
			foam_properties_dictionary = self.getPropertiesDictionary()
			if type(foam_properties_dictionary).__name__ != 'NoneType':
				for key in foam_properties_dictionary:
					self.fenics_foam_solver.setFoamProperty(key, foam_properties_dictionary[key])

			# Set all variables to fenics_foam_solver
			u_split_deepcopy = u.split(deepcopy = True)
			self.fenics_foam_solver.setFEniCSFunctionToFoamVector(u_split_deepcopy[0], foam_variable_name = 'U')
			self.fenics_foam_solver.setFEniCSFunctionToFoamVector(u_split_deepcopy[1], foam_variable_name = 'p')
			if flow_regime == 'turbulent (Spalart-Allmaras)':
				self.fenics_foam_solver.setFEniCSFunctionToFoamVector(u_split_deepcopy[2], foam_variable_name = 'nuTilda')
			self.fenics_foam_solver.setFEniCSFunctionToFoamVector(alpha, foam_variable_name = 'alpha_design', set_calculated_foam_boundaries = True, ensure_maximum_minimum_values_after_projection = True)

			# Solve the problem with OpenFOAM and plot residuals
			self.fenics_foam_solver.solve(
				silent_run_mode = False,
				num_logfile_lines_to_print_in_silent_mode = 0,
				continuously_plot_residuals_from_log = True,
				continuously_plot_residuals_from_log_time_interval = 5,
				continuously_plot_residuals_from_log_x_axis_label = 'Iteration',
				continuously_plot_residuals_from_log_y_axis_scale = 'log/symlog',
				)

			# Set the state variables from the fenics_foam_solver
			self.fenics_foam_solver.setFoamVectorToFEniCSFunction(u_split_deepcopy[0], foam_variable_name = 'U')
			self.fenics_foam_solver.setFoamVectorToFEniCSFunction(u_split_deepcopy[1], foam_variable_name = 'p')
			if flow_regime == 'turbulent (Spalart-Allmaras)':
				self.fenics_foam_solver.setFoamVectorToFEniCSFunction(u_split_deepcopy[2], foam_variable_name = 'nuTilda')
			fenics_topopt_foam.assignSubFunctionsToFunction(to_u_mixed = u, from_u_separated_array = list(u_split_deepcopy))
			[bc.apply(u.vector()) for bc in bcs]
			self.u_array_copy = u.vector().get_local()
			return u

	# Create fenics_foam_solver_with_update
	fenics_foam_solver_with_update = FEniCSFoamSolverWithUpdate(u, alpha)
	return fenics_foam_solver_with_update

# Function to solve the forward problem
global fenics_foam_solver_with_update, l_wall, mu_T
fenics_foam_solver_with_update = None; l_wall = None; mu_T = 0
def solve_forward_problem(alpha):
	global fenics_foam_solver_with_update, l_wall, mu_T

	# Set the state vector and test functions
	u = Function(U); u.rename("StateVariable", "StateVariable")
	u_split = split(u); v = u_split[0]; p = u_split[1]
	if flow_regime == 'turbulent (Spalart-Allmaras)': nu_T_aux = u_split[2]
	w = TestFunction(U)
	w_split = split(w); w_v = w_split[0]; w_p = w_split[1]
	if flow_regime == 'turbulent (Spalart-Allmaras)': w_nu_T_aux = w_split[2]

	# Additional definitions
	n = FacetNormal(mesh)
	nu_ = mu_/rho_

	# Wall distance computation (modified Eikonal equation)
	if flow_regime == 'turbulent (Spalart-Allmaras)':
		G_space = FunctionSpace(mesh, 'Lagrange', 1)
		G = Function(G_space); w_G = TestFunction(G_space)
		mesh_hmax = MPI.comm_world.allreduce(mesh.hmax(), op = mpi4py.MPI.MAX)
		G_initial = interpolate(Constant(1./mesh_hmax), G_space); G.assign(G_initial)

		sigma_wall = 0.1
		G_ref = 1./mesh_hmax
		F_G = inner(grad(G), grad(G))*w_G*dx - inner(grad(G), grad(sigma_wall*G*w_G))*dx - (1. + 2*sigma_wall)*G**4 * w_G*dx - gamma(alpha)*(G - G_ref) * w_G*dx
		bcs_G = [DirichletBC(G_space, G_ref, boundary_markers, marker_numbers['wall'])]

		dF_G = derivative(F_G, G); problem_G = NonlinearVariationalProblem(F_G, G, bcs_G, dF_G)
		solver_G = NonlinearVariationalSolver(problem_G)
		solver_G.solve(annotate = True)
		l_wall = 1./G - 1./G_ref

	# Weak form of the turbulent equations (Spalart-Allmaras model)
	if flow_regime == 'turbulent (Spalart-Allmaras)':
		k_von_Karman = 0.41; c_v1 = 7.1; c_b1 = 0.1355; c_b2 = 0.6220; c_w2 = 0.3; c_w3 = 2.0; sigma = 2./3
		adjustment = DOLFIN_EPS_LARGE # Adjustment for numerical precision (1.E-14)
		Chi = nu_T_aux/nu_; f_v1 = (Chi**3)/(Chi**3 + c_v1**3)
		nu_T = f_v1*nu_T_aux; mu_T = rho_*nu_T
		Omega = 1/2. * (grad(v) - grad(v).T); Omega_m = sqrt(2.*inner(Omega, Omega) + adjustment); S = Omega_m
		f_v2 = 1. - Chi/(1 + Chi*f_v1); S_tilde = ufl.Max(S + nu_T_aux/(k_von_Karman**2*(l_wall**2 + adjustment))*f_v2, 0.3*Omega_m)
		S_tilde_para_r = ufl.Max(S_tilde, 1.E-6)
		r_i = ufl.Min(nu_T_aux/(S_tilde_para_r*k_von_Karman**2*(l_wall**2 + adjustment)), 10.0)
		g_i = r_i + c_w2*(r_i**6 - r_i)
		f_w = g_i*((1. + c_w3**6)/(g_i**6 + c_w3**6))**(1./6)
		c_w1 = c_b1/k_von_Karman**2 + (1 + c_b2)/sigma
		F_SA = inner(v, rho_*grad(nu_T_aux))*w_nu_T_aux*dx - (c_b1*rho_*S_tilde*nu_T_aux)*w_nu_T_aux*dx + inner( rho_*(nu_ + nu_T_aux)*grad(nu_T_aux), grad(w_nu_T_aux)/sigma)*dx - c_b2/sigma*rho_*inner(grad(nu_T_aux), grad(nu_T_aux))*w_nu_T_aux*dx - ( - c_w1*f_w*rho_*(nu_T_aux**2)/(l_wall**2 + adjustment) )*w_nu_T_aux*dx + lambda_kappa_v*k(alpha)*(nu_T_aux - 0.0)*w_nu_T_aux*dx
	else:
		mu_T = 0

	# Weak form of the pressure-velocity formulation
	I_ = as_tensor(np.eye(2)); T = -p*I_ + (mu_ + mu_T)*(grad(v) + grad(v).T); v_mat = v
	F_PV = div(v) * w_p*dx + inner( grad(w_v), T )*dx + rho_ * inner( dot(grad(v), v), w_v )*dx + inner(k(alpha) * v_mat, w_v)*dx

	# Full weak form
	if flow_regime == 'laminar':                        F = F_PV
	elif flow_regime == 'turbulent (Spalart-Allmaras)': F = F_PV + F_SA

	# Boundary conditions
	bcs = [DirichletBC(U.sub(0), wall_velocity_value, boundary_markers, marker_numbers['wall']), DirichletBC(U.sub(0), inlet_velocity_expression, boundary_markers, marker_numbers['inlet'])]
	if flow_regime == 'turbulent (Spalart-Allmaras)':
		bcs += [DirichletBC(U.sub(2), Constant(0.0), boundary_markers, marker_numbers['wall']), DirichletBC(U.sub(2), inlet_nu_T_aux_value, boundary_markers, marker_numbers['inlet'])]

	# Prepare the FEniCSFoamSolver a single time
	if type(fenics_foam_solver_with_update).__name__ == 'NoneType': 
		fenics_foam_solver_with_update = prepareFEniCSFoamSolverWithUpdate(u, alpha, mesh, boundary_markers, marker_numbers, bcs)
	else:
		fenics_foam_solver_with_update.u = u; fenics_foam_solver_with_update.alpha = alpha

	# Solve the simulation
	dF = derivative(F, u); problem = NonlinearVariationalProblem(F, u, bcs, dF)
	nonlinear_solver = UncoupledNonlinearVariationalSolver(problem, simulation_solver = fenics_foam_solver_with_update)
	u = nonlinear_solver.solve()
	return u

# Initial setup for topology optimization
alpha = interpolate(Constant(f_V), A) # Initial guess for the design variable
set_working_tape(Tape())              # Clear all annotations and restart the adjoint model

# Solve the simulation
u = solve_forward_problem(alpha)

# Visualization files
alpha_pvd_file = File("%s/alpha_iterations.pvd" %(output_folder)); alpha_viz = Function(A, name = "AlphaVisualisation")
dj_pvd_file = File("%s/dj_iterations.pvd" %(output_folder)); dj_viz = Function(A, name = "dJVisualisation")

# Callback during topology optimization
global current_iteration
current_iteration = 0
def derivative_cb_post(j, dj, current_alpha):
	global current_iteration
	print("\n [Iteration: %d] J = %1.7e\n" %(current_iteration, j)); current_iteration += 1
	# Save for visualization
	alpha_viz.assign(current_alpha); alpha_pvd_file << alpha_viz
	dj_viz.assign(dj); dj_pvd_file << dj_viz

# Objective function
u_split = split(u); v = u_split[0]
J = assemble((1/2.*(mu_ + mu_T)*inner(grad(v) + grad(v).T, grad(v) + grad(v).T) + inner(k(alpha) * v, v))*dx)
print(" Current objective function value: %1.7e" %(J))

# Set the topology optimization problem and solver
alpha_C = Control(alpha)
Jhat = ReducedFunctional(J, alpha_C, derivative_cb_post = derivative_cb_post)
problem_min = MinimizationProblem(Jhat, bounds = (0.0, 1.0), constraints = [UFLInequalityConstraint((f_V - alpha)*dx, alpha_C)])
solver_opt = IPOPTSolver(problem_min, parameters = {'maximum_iterations': 100})

# Perform topology optimization
alpha_opt = solver_opt.solve()
alpha.assign(alpha_opt); alpha_viz.assign(alpha)
alpha_pvd_file << alpha_viz

# Plot a simulation
u = solve_forward_problem(alpha)
fenics_foam_solver_with_update.plotResults(file_type = 'VTK', tag_folder_name = '_final')
u_split_deepcopy = u.split(deepcopy = True)
v_plot = u_split_deepcopy[0]
p_plot = u_split_deepcopy[1]
File("%s/simulation_final_v.pvd" %(output_folder)) << v_plot
File("%s/simulation_final_p.pvd" %(output_folder)) << p_plot
if flow_regime == 'turbulent (Spalart-Allmaras)':
	nu_T_aux_plot = u_split_deepcopy[2]
	File("%s/simulation_final_nu_T_aux.pvd" %(output_folder)) << nu_T_aux_plot



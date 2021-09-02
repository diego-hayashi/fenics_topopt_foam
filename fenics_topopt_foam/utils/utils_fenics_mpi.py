################################################################################
#                               utils_fenics_mpi                               #
################################################################################
# Some utilities for using with FEniCS with MPI.

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

############################### Python libraries ###############################

# Inspect
import inspect

# Functools
import functools

# NumPy
import numpy as np

# mpi4py
import mpi4py

############################# Project libraries ################################

# Utilities
from ..utils import utils

############################### FEniCS libraries ###############################

# Global MPI variables
global comm, comm_self, rank

# FEniCS
try:
	import fenics
	from fenics import *
	 # Do not import dolfin-adjoint here. This is a non-annotated module!

	# MPI from FEniCS
	comm = MPI.comm_world 
	comm_self = MPI.comm_self
	rank = comm.Get_rank()

except:
	utils.printDebug(" ❌ FEniCS not installed! Can't use FEniCSFoamSolver!")

	# Dummy MPI communicator
	class DummyCommunicator():
		"""
		A dummy communicator for running in serial mode, when FEniCS
		is not available in your installation.
		"""
		def Get_size(self):
			return 1
		def Barrier(self):
			pass
		def Get_rank(self):
			return 0
	comm = DummyCommunicator()
	comm_self = DummyCommunicator()
	rank = comm.Get_rank()

# First processor lock
global _first_processor_lock_count
_first_processor_lock_count = []

############################# MPI communicator #################################

def changeMPIcomm(new_comm, new_comm_self = None):
	"""
	Change the MPI communicator for FEniCS TopOpt Foam.
	"""

	# Global MPI variables
	global comm, comm_self, rank

	# comm
	if type(new_comm).__name__ == 'NoneType':
		comm = DummyCommunicator()
	else:
		comm = new_comm

	# comm_self
	if type(new_comm_self).__name__ == 'NoneType':
		comm_self = new_comm
	else:
		comm_self = new_comm_self

	# rank
	rank = comm.Get_rank()

##################################### MPI ######################################

def runningInSerialOrFirstProcessor():
	"""
	Checks if the code is running in serial, or if it is the first processor.
	"""
	return (not runningInParallel()) or (runningInFirstProcessor())

def runningInParallel(mpi_single_processor = False):
	"""
	Checks if the code is running in parallel.
	"""
	if mpi_single_processor == False:
		return comm.Get_size() > 1
	else:
		return 0

def runningInFirstProcessor():
	"""
	Check if it is processor 0.
	"""
	return rank == 0

def checkProcessor(num):
	"""
	Check if it is processor num.
	"""
	return rank == num

def waitProcessors():
	"""
	Barrier synchronization.
	"""
	assertFirstProcessorUnlocked()
	comm.Barrier()

def quantityOfProcessors():
	"""
	Quantity of processors.
	"""
	return comm.Get_size()

def checkValidProcessorNumber(num):
	"""
	Check if it is a valid processor number.
	"""
	assert type(num).__name__ == 'int'
	assert num < quantityOfProcessors(), " ❌ ERROR: Invalid processor number %d >= %d" %(num, quantityOfProcessors())

######################### MPI - First processor lock ###########################

def checkFirstProcessorLockFunctionTraceback():
	"""
	Check if the first processor lock's function traceback.
	"""
	global _first_processor_lock_count
	return _first_processor_lock_count

def assertFirstProcessorUnlocked():
	"""
	Check if the first processor's lock has been released.
	"""
	global _first_processor_lock_count
	assert checkFirstProcessorUnlocked(), " ❌ ERROR: Deadlock! Can not perform barrier synchronization due to first processor lock having been acquired before! Function traceback: %s" %(checkFirstProcessorLockFunctionTraceback())

def checkFirstProcessorUnlocked():
	"""
	Check if the first processor's lock has been released.
	"""
	global _first_processor_lock_count
	return len(_first_processor_lock_count) == 0

def checkFirstProcessorLocked():
	"""
	Check if the first processor's lock has been acquired.
	"""
	global _first_processor_lock_count
	return len(_first_processor_lock_count) != 0

def acquireFirstProcessorLock():
	"""
	Acquire first processor lock.
	"""
	global _first_processor_lock_count
	if runningInParallel():
		num_lock_traceback = 4
		stack_len = max(min(len(inspect.stack()) - 1, num_lock_traceback), 1)
		lock_traceback = ' < '.join([str(inspect.stack()[2 + i].function) for i in range(stack_len)])
		_first_processor_lock_count += [lock_traceback]
		#print("[%d]" %(rank), " -- ACQUIRED: ", _first_processor_lock_count)

def releaseFirstProcessorLock():
	"""
	Release first processor lock.
	"""
	global _first_processor_lock_count
	if runningInParallel():
		_first_processor_lock_count.pop()
		assert len(_first_processor_lock_count) >= 0, " ❌ ERROR: Releasing more first processor locks than what we have. What is going on?"
		#print("[%d]" %(rank), " -- RELEASED: ", _first_processor_lock_count)

class first_processor_lock():
	"""
	A context manager for locking the operations to the first processor.
	This is important in order to avoid deadlocks, or at least to avoid 
	weird results due to broadcasting variables at different positions of the
	code.

	It is suggested to use it as:

	1) In conjunction with 'runningInFirstProcessor()':

	if utils_fenics_mpi.runningInFirstProcessor():
		with utils_fenics_mpi.first_processor_lock():
			[...]

	2) In conjunction with 'runningInSerialOrFirstProcessor()':

	if utils_fenics_mpi.runningInSerialOrFirstProcessor():
		with utils_fenics_mpi.first_processor_lock():
			[...]

	-----------------------------------------------------------------------

	Example:

	Consider the following case:

	if utils_fenics_mpi.runningInFirstProcessor():
		checked = utils.checkIfFileExists([...], broadcast_result = True, wait_for_everyone = True) # (* returns True or False) 
	subfolders = utils.getSubfolderNames([...], broadcast_result = True, wait_for_everyone = True) # (* returns list of subfolder names)

	* These two functions ("utils.checkIfFileExists" and "utils.getSubfolderNames") are wrapped by "utils.only_the_first_processor_executes"

	=> This would mean that:
		 - Processor 0 would be stuck at the broadcasting from "utils.checkIfFileExists"
		 - Processor 1 would be waiting for the broadcasting at "utils.getSubfolderNames"

	=> In terms of values, the following would happen:

	if utils_fenics_mpi.runningInFirstProcessor():
		checked = utils.checkIfFileExists([...], broadcast_result = True, wait_for_everyone = True) # (* returns True or False) 
		 # [Processor 0] ----> checked = True or False

	subfolders = utils.getSubfolderNames([...], broadcast_result = True, wait_for_everyone = True) # (* returns list of subfolder names)
	 # [Processor 0] ----> subfolders = List of subfolder names
	 # [Processor 1] ----> subfolders = True or False

	which is obviously wrong!

	Thus, it is suggested to use the "utils_fenics_mpi.first_processor_lock" context manager.

	if utils_fenics_mpi.runningInFirstProcessor():
		with utils_fenics_mpi.first_processor_lock():
			checked = utils.checkIfFileExists([...], broadcast_result = True, wait_for_everyone = True) # (* returns True or False) 
	subfolders = utils.getSubfolderNames([...], broadcast_result = True, wait_for_everyone = True) # (* returns list of subfolder names)

	=> When a broadcasting operation or a barrier synchronization are called,
	   the code will return an error, indicating that there is a deadlock!
	   This indicates the need to fix this part of the code. As a matter of
	   fact, the corrected code would be in one of the following approaches
	   (being the second one the best in this case):

	1) Considering "broadcast_result = False, wait_for_everyone = False"
	   in the "utils.checkIfFileExists" call and broadcasting the result:

	if utils_fenics_mpi.runningInFirstProcessor():
		with utils_fenics_mpi.first_processor_lock():
			checked = utils.checkIfFileExists([...], broadcast_result = False, wait_for_everyone = False) # (* returns True or False) 
			 # [Processor 0] ----> checked = True or False

	checked = broadcastToAll(checked)
	 # [Processor 1] ----> checked = True or False

	subfolders = utils.getSubfolderNames([...], broadcast_result = True, wait_for_everyone = True) # (* returns list of subfolder names)
	 # [Processor 0] ----> subfolders = List of subfolder names
	 # [Processor 1] ----> subfolders = List of subfolder names

	2) Considering that the "utils.checkIfFileExists" is already prepared for it!

	checked = utils.checkIfFileExists([...], broadcast_result = True, wait_for_everyone = True) # (* returns True or False) 
	 # [Processor 0] ----> checked = True or False
	 # [Processor 1] ----> checked = True or False

	subfolders = utils.getSubfolderNames([...], broadcast_result = True, wait_for_everyone = True) # (* returns list of subfolder names)
	 # [Processor 0] ----> subfolders = List of subfolder names
	 # [Processor 1] ----> subfolders = List of subfolder names

	"""
	def __init__(self, mpi_single_processor = False):
		self.mpi_single_processor = mpi_single_processor

	def __enter__(self):
		"""
		Run when entering the "with" call.
		"""

		if self.mpi_single_processor == False:
			# Acquire lock
			acquireFirstProcessorLock()

	def __exit__(self, type, value, traceback):
		"""
		Run when exiting the "with" call.
		"""

		if self.mpi_single_processor == False:
			# Release lock
			releaseFirstProcessorLock()

######################### evaluateBetweenProcessors ############################

def evaluateBetweenProcessors(val, operation = 'sum', proc_destination = 'all', **kwargs):
	"""
	Evaluate an operation between processors.
	"""

	if runningInParallel():

		# Check if the destination processor is valid.
		if proc_destination == 'all':
			pass
		else:
			checkValidProcessorNumber(proc_destination)

		# Check for first processor lock
		assertFirstProcessorUnlocked()

		#### Sum
		if operation == 'sum':
			if proc_destination == 'all':
				res = comm.allreduce(val, op = mpi4py.MPI.SUM)
			else:
				res = comm.reduce(val, op = mpi4py.MPI.SUM, root = proc_destination)

		#### Juntar em lista
		if operation == 'create list':
			val = [val]
			res = compute_MPI_operation(
				val,
				operation = lambda x_combined, x_next : x_combined + x_next,
				proc_destination = proc_destination
				)

		#### Product
		elif operation == 'product':
			if proc_destination == 'all':
				res = comm.allreduce(val, op = mpi4py.MPI.PROD)
			else:
				res = comm.reduce(val, op = mpi4py.MPI.PROD, root = proc_destination)

		#### Maximum
		elif operation == 'maximum':
			if proc_destination == 'all':
				res = comm.allreduce(val, op = mpi4py.MPI.MAX)
			else:
				res = comm.reduce(val, op = mpi4py.MPI.MAX, root = proc_destination)

		#### Minimum
		elif operation == 'minimum':
			if proc_destination == 'all':
				res = comm.allreduce(val, op = mpi4py.MPI.MIN)
			else:
				res = comm.reduce(val, op = mpi4py.MPI.MIN, root = proc_destination)

		#### And
		elif operation == 'and':
			res = compute_MPI_operation(
				val,
				operation = lambda x_combined, x_next : x_combined and x_next, 
				proc_destination = proc_destination
				)

		#### ==
		elif operation == '==':
			res = compute_MPI_operation(
				val,
				operation = lambda x_combined, x_next : x_combined and (val == x_next), 
				proc_destination = proc_destination
				)

		#### Sum (NumPy)
		elif operation == 'term-by-term sum of arrays':
			axis = kwargs.get('axis', 0)
			res = compute_MPI_operation(
				val,
				operation = lambda x_combined, x_next : x_combined + x_next,
				proc_destination = proc_destination
				)

		#### Append (arrays)
		elif operation == 'append':
			axis = kwargs.get('axis', 0)
			value_displacement = kwargs.get('value_displacement', None)

			if type(value_displacement).__name__ != 'NoneType':

				res = compute_MPI_operation(val,
					operation = lambda x_combined, x_next : np.append(x_combined, x_next + value_displacement(), axis = axis), 
					proc_destination = proc_destination
					)

			else:
				res = compute_MPI_operation(val,
					operation = lambda x_combined, x_next : np.append(x_combined, x_next, axis = axis), 
					proc_destination = proc_destination
					)

		#### Broadcast
		elif operation == 'broadcast':
			assert proc_destination == 'all'
			proc_source = kwargs.get('proc_source', 0)
			res = comm.bcast(val, root = proc_source)

		else:
			raise ValueError(" ❌ ERROR: operation == '%s' is not defined!" %(operation))

	else:
		res = val

	return res

def compute_MPI_operation(var, operation = lambda x_combined, x_next : x_next, proc_destination = 'all'):
	"""
	Compute MPI operation.
	"""

	if runningInParallel():

		# Wait for everyone!
		waitProcessors()

		for i in range(quantityOfProcessors()):
			
			# Broadcast from processor "i" to everyone else
			var_i = comm.bcast(var, root = i)
			
			# Compute new value
			if (proc_destination == 'all') or (checkProcessor(rank)):
				if i == 0:
					var_new = var_i
				else:
					var_new = operation(var_new, var_i)
			else:
				pass

		if (proc_destination == 'all') or (checkProcessor(rank)):
			pass
		else:
			var_new = None

		# Wait for everyone!
		waitProcessors()

	else:
		var_new = var

	return var_new

def broadcastToAll(val, **kwargs):
	"""
	Broadcast result to all.
	"""
	kwargs.pop('operation', 'broadcast')
	kwargs.pop('proc_destination', 'all')
	return evaluateBetweenProcessors(val, operation = 'broadcast', proc_destination = 'all', **kwargs)

######################## checkIfMeshIsInSingleProcessor ########################

def checkIfMeshIsInSingleProcessor(mesh):
	"""
	Check if the mesh is located in a single processor.
	"""

	quantity_of_not_nones = evaluateBetweenProcessors(0 if type(mesh).__name__ == 'NoneType' else 1, operation = 'and', proc_destination = 'all')

	if quantity_of_not_nones == 1:
		comparison = True

	else:

		mesh_rank = mesh.Get_rank()
		all_equal_mesh_ranks = evaluateBetweenProcessors(mesh_rank, operation = '==', proc_destination = 'all')

		if all_equal_mesh_ranks == True:
			comparison = True
		else:
			comparison = False

	return comparison

##################### reuse_computed_value_if_possible #########################

def reuse_computed_value_if_possible(value_name):
	"""
	Decorator to reuse a previously computed value if possible.
	"""

	def decorator(original_function):
		@functools.wraps(original_function) # For using the correct 'docstring' in the wrapped function
		def wrapper(self, *args, **kwargs):

			# Additional kwargs
			reuse_value_if_possible = kwargs.pop('reuse_value_if_possible', True)

			# Saved value name, intended to be a private term of the class
			proc_destination = self.proc_destination
			class_name = type(self).__name__
			saved_value_name = '_%s_%s_proc_%s' %(class_name, value_name, proc_destination)

			# Check if it already exists
			inside_self = (saved_value_name in self.__dict__) and (type(self.__dict__[saved_value_name]).__name__ != 'NoneType')

			# Compute if necessary
			if (reuse_value_if_possible == False) or (inside_self == False):
				self.__dict__[saved_value_name] = original_function(self, *args, **kwargs)

			# Get the current value
			current_value = self.__dict__[saved_value_name]

			return current_value

		return wrapper
	return decorator

################################################################################
##################### Unified structures for parallelism #######################
################################################################################
#
# * Some concepts being used:
#
#   - "Gathered array": Array composed by appending the arrays in each processor sequentially by processor number (0, 1, 2 ...)
# 
#      Example:
# 
#           Processor 0:     [ X X X X X X ]
#           Processor 1:                 [ Y Y Y Y Y Y Y Y Y ]
#	    --------------------------------------------------
#           Gathered array : [ X X X X X X Y Y Y Y Y Y Y Y Y ]
#                              ----------- -----------------
#                              Processor 0     Processor 1
# 
#   - "Unified array": Array composed by eliminating repeating elements from the gathered array,
#                      which may arise due in mesh entities that are shared between processors.
# 
#      Example:
# 
#           Processor 0:     [ X_1 X_2 X_3 ]
#           Processor 1:                 [ X_4 X_5 X_3 X_6 ]
#	    --------------------------------------------------
#           Gathered array : [ X_1 X_2 X_3 X_4 X_5 X_3 X_6 ]
#                             -----------  ---------------
#                             Processor 0    Processor 1
#	    --------------------------------------------------
#           Unified array :  [ X_1 X_2 X_3 X_4 X_5 X_6 ]
# 
#            * The unified array may be reordered, or keep the same original order.
# 

################################################################################
#################### Unified mesh structures for parallelism ###################
################################################################################

############################### UnifiedMesh ####################################

class UnifiedMesh():
	"""
	A unified mesh structure that deals with parallel access to mesh entities.
	The idea is to use this when dealing with parallelism.
	-> From FEniCS 'Mesh' object
	"""

	def __init__(self, mesh, prepare_beforehand = [], proc_destination = 'all'):

		utils.printDebug(" ⚙️ [%s] Creating %s..." %(self.__class__.__name__, self.__class__.__name__), mpi_wait_for_everyone = True)

		self.mesh = mesh 
		self.proc_destination = proc_destination

		assert self.proc_destination == 'all' # All is ready for 'all'. For 0, I am not so sure...

		self.entity_mapping_data = {}

		self._initialized_entities = []

		for i in range(len(prepare_beforehand)):
			if type(prepare_beforehand[i]).__name__ == 'str':
				name_method_preparation = prepare_beforehand[i]
				args_preparation = []
			else:
				name_method_preparation = prepare_beforehand[i][0]
				args_preparation = prepare_beforehand[i][1]

				if type(args_preparation).__name__ not in ['list', 'tuple']:
					args_preparation = [args_preparation]

			# Compute method from name and arguments
			getattr(self, name_method_preparation)(*args_preparation)

	###### Local operations that work are equal to global operations #######

	@reuse_computed_value_if_possible('geometric_dimension')
	def geometric_dimension(self):
		"""
		Geometric dimension.
		-> From FEniCS 'mesh.geometric_dimension' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'geometric_dimension'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		return self.mesh.geometric_dimension()

	################# Externally overloaded funcionality ###################

	@reuse_computed_value_if_possible('topology')
	def topology(self):
		"""
		Topology.
		-> From FEniCS 'mesh.topology' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'topology'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		return UnifiedTopology(self)

	########################## Coordinates #################################

	@reuse_computed_value_if_possible('coordinates')
	def coordinates(self):
		"""
		Mesh coordinates.
		-> From FEniCS 'mesh.coordinates' function
		"""

		utils.printDebug(" ⚙️ [%s] Preparing 'coordinates'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)

		# Local coordinates
		local_coordinates = self.mesh.coordinates()

		# There are interface vertices, which should be disregarded in the unified coordinates
		if runningInParallel():

			# Data type of the indices
			dtype_ = self.mesh.cells().dtype.name

			# Gathered coordinates
			gathered_coordinates = evaluateBetweenProcessors(
				local_coordinates, 
				operation = 'append', 
				proc_destination = self.proc_destination, 
				axis = 0
				)

			if self.proc_destination == 'all' or runningInFirstProcessor():

				# Acquire lock
				if self.proc_destination != 'all' and runningInFirstProcessor():
					acquireFirstProcessorLock()

				# Unify the vertices
				(unified_coordinates, self.map_gathered_to_unified_vertices, self.map_unified_to_gathered_vertices, self.counts_repeated_gathered_vertices, self.map_local_to_unified_vertices) = map_values_between_gathered_and_unified(gathered_coordinates, local_coordinates, dtype = dtype_)
					# return_index = True <-> gathered_coordinates[map_gathered_to_unified_vertices] = unified_coordinates
					# return_inverse = True <-> unified_coordinates[map_unified_to_gathered_vertices] = gathered_coordinates

				mesh_coordinates = unified_coordinates

				# Release lock
				if self.proc_destination != 'all' and runningInFirstProcessor():
					releaseFirstProcessorLock()

			else:
				mesh_coordinates = None
		else:
			mesh_coordinates = local_coordinates

		return mesh_coordinates

	############################# Entities #################################

	@reuse_computed_value_if_possible('cells')
	def cells(self):
		"""
		Cell vertices.
		-> From FEniCS 'mesh.cells' function
		"""
		return self.cell_vertices()

	def init(self, *args):
		"""
		Initialize mesh structures.
		-> From FEniCS 'mesh.init' function
		"""

		if type(args).__name__ == 'tuple':
			str_args = str(args).split('(')[1].split(')')[0]
			if len(args) == 1:
				str_args = str_args[:-1] # Remove last comma
			str_args = "(%s)" %(str_args)
		else:
			str_args = str(args)

		if str_args in self._initialized_entities:
			utils.printDebug(" ⚙️ [%s] %s is already initialized!" %(self.__class__.__name__, str_args), mpi_wait_for_everyone = True)
			return
		else:
			self._initialized_entities += [str_args]

		# Print the initialization type
		utils.printDebug(" ⚙️ [%s] Initializing %-8s" %(self.__class__.__name__, "%s: " %(str_args)) , end = "", mpi_wait_for_everyone = True)

		# Initialization in the local mesh
		self.mesh.init(*args)

		# Initialization in the unified mesh
		if len(args) == 1:
			dim_entity = args[0]
			name_entity = self.map_entity_dim_to_name(dim_entity)

			# Print the initialization name
			utils.printDebug("%s mapping" %(name_entity), mpi_wait_for_everyone = True)

			# Edge-vertex
			if name_entity == 'edge':
				self.entity_mapping_data[name_entity] = self.edge_vertices()

			# Face-vertex
			elif name_entity == 'face':
				self.entity_mapping_data[name_entity] = self.face_vertices()

			# Cell-vertex
			elif name_entity == 'cell':
				self.entity_mapping_data[name_entity] = self.cell_vertices()

			else:
				raise ValueError(" ❌ ERROR: name_entity1 = '%s' is not defined!" %(name_entity1))

		elif len(args) == 2:
			dim_entity1 = args[0]
			dim_entity2 = args[1]
		
			name_entity1 = self.map_entity_dim_to_name(dim_entity1)
			name_entity2 = self.map_entity_dim_to_name(dim_entity2)
			name_entity_mapping = '%s-%s' %(name_entity1, name_entity2)

			# Print the initialization name
			utils.printDebug("%s mapping" %(name_entity_mapping), mpi_wait_for_everyone = True)

			if name_entity2 == 'vertex':

				# Edge-vertex
				if name_entity1 == 'edge':
					self.entity_mapping_data[name_entity_mapping] = self.edge_vertices()

				# Face-vertex
				elif name_entity1 == 'face':
					self.entity_mapping_data[name_entity_mapping] = self.face_vertices()

				# Cell-vertex
				elif name_entity1 == 'cell':
					self.entity_mapping_data[name_entity_mapping] = self.cell_vertices()

				else:
					raise ValueError(" ❌ ERROR: name_entity1 = '%s' is not defined!" %(name_entity1))
			
			elif name_entity2 in ['edge', 'face', 'cell']:

				# Initialize for both entities
				self.init(dim_entity1)
				self.init(dim_entity2)

				if name_entity_mapping in ['edge-edge', 'face-face', 'cell-cell']:
					return

				# Data type of the indices
				dtype_ = self.mesh.cells().dtype.name

				# Local map: [entity1]-[entity2]
				if name_entity1 == 'vertex': # Vertex-[entity2]
					(map_gathered_to_unified_entity1, map_unified_to_gathered_entity1) = self.vertex_map_gathered_unified()
					counts_repeated_gathered_entity1 = self.counts_repeated_gathered_vertices
					local_entities_1_2 = np.array([vertex.entities(dim_entity2).copy() for vertex in vertices(self.mesh)])

				elif name_entity1 == 'edge': # Edge-[entity2]
					(map_gathered_to_unified_entity1, map_unified_to_gathered_entity1) = self.edge_map_gathered_unified()
					counts_repeated_gathered_entity1 = self.counts_repeated_gathered_edges
					local_entities_1_2 = np.array([edge.entities(dim_entity2).copy() for edge in edges(self.mesh)])

				elif name_entity1 == 'face': # Face-[entity2]
					(map_gathered_to_unified_entity1, map_unified_to_gathered_entity1) = self.face_map_gathered_unified()
					counts_repeated_gathered_entity1 = self.counts_repeated_gathered_faces
					local_entities_1_2 = np.array([face.entities(dim_entity2).copy() for face in faces(self.mesh)])

				elif name_entity1 == 'cell': # Cell-[entity2]
					(map_gathered_to_unified_entity1, map_unified_to_gathered_entity1) = self.cell_map_gathered_unified()
					counts_repeated_gathered_entity1 = self.counts_repeated_gathered_cells
					local_entities_1_2 = np.array([cell.entities(dim_entity2).copy() for cell in cells(self.mesh)])

				else:
					raise ValueError(" ❌ ERROR: name_entity1 = '%s' is not defined here!" %(name_entity1))

				# Map from local to unified [entity2]
				if name_entity2 == 'vertex': # Vertex
					map_local_to_unified_entity2 = self.map_local_to_unified_vertices
				elif name_entity2 == 'edge': # Edge
					map_local_to_unified_entity2 = self.map_local_to_unified_edges
				elif name_entity2 == 'face': # Face
					map_local_to_unified_entity2 = self.map_local_to_unified_faces
				elif name_entity2 == 'cell': # Cell
					map_local_to_unified_entity2 = self.map_local_to_unified_cells
				else:
					raise ValueError(" ❌ ERROR: name_entity2 = '%s' is not defined here!" %(name_entity2))

				# Convert local to unified [entity2]
				for i in range(len(local_entities_1_2)):
					for j in range(len(local_entities_1_2[i])):
						local_entities_1_2[i][j] = map_local_to_unified_entity2[local_entities_1_2[i][j]]

				# Gather [entity1]
				gathered_local_entities_1_2 = evaluateBetweenProcessors(
					local_entities_1_2,
					operation = 'append', 
					proc_destination = self.proc_destination, 
					axis = 0
					)

				# Unify [entity1], leaving out the repeated [entity1] entities that came from the gathered entities
				unified_entities_1_2 = gathered_local_entities_1_2[map_gathered_to_unified_entity1]

				if dim_entity1 < dim_entity2:

					# Check for the "repeated terms", which are the ones that represent the mesh relations between processors
					repeated_term_unified_indices = np.where(counts_repeated_gathered_entity1 > 1)[0]

					# Include the missing "repeated terms"
					for repeated_term_unified_index in repeated_term_unified_indices:

						# Map to gathered indices
						repeated_term_gathered_index = np.where(map_unified_to_gathered_entity1 == repeated_term_unified_index)[0]

						# Get the dependencies from the gathered indices
						missing_dependencies = gathered_local_entities_1_2[repeated_term_gathered_index]

						# Merge the arrays and remove repeated dependencies
						missing_dependencies_merged = None
						for i in range(len(missing_dependencies)):
							if type(missing_dependencies_merged).__name__ == 'NoneType':
								missing_dependencies_merged = missing_dependencies[i]
							else:
								missing_dependencies_merged = np.append(missing_dependencies_merged, missing_dependencies[i])
						missing_dependencies_unique = np.unique(missing_dependencies_merged, axis = None)

						# Include
						unified_entities_1_2[repeated_term_unified_index] = missing_dependencies_unique

				# Save the resulting map
				self.entity_mapping_data[name_entity_mapping] = unified_entities_1_2

			else:
				raise NotImplementedError

		else:
			raise NotImplementedError

	def map_entity_dim_to_name(self, entity_dim):
		"""
		Map the entity dimension to the corresponding name.
		"""

		if entity_dim == 0:
			entity_name = 'vertex'

		elif entity_dim == 1:
			entity_name = 'edge'

		elif entity_dim == 2:

			if self.topology().dim() == 3:
				entity_name = 'face'
			elif self.topology().dim() == 2:
				entity_name = 'cell'
			else:
				raise ValueError(" ❌ ERROR: Can not determine for entity_dim = '%d'" %(entity_dim))

		elif entity_dim == 3:
			if self.topology().dim() == 3:
				entity_name = 'cell'
			else:
				raise ValueError(" ❌ ERROR: Can not determine for entity_dim = '%d'" %(entity_dim))
		else:
			raise ValueError(" ❌ ERROR: Can not determine for entity_dim = '%d'" %(entity_dim))

		return entity_name

	############################## Vertices ################################

	@reuse_computed_value_if_possible('cell_vertices')
	def cell_vertices(self):
		"""
		Cell vertices.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'cell_vertices'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		(cell_vertices, self.map_gathered_to_unified_cells, self.map_unified_to_gathered_cells, self.counts_repeated_gathered_cells, self.map_local_to_unified_cells) = self._computeEntityVertexMapping('cells')
		return cell_vertices

	@reuse_computed_value_if_possible('face_vertices')
	def face_vertices(self):
		"""
		Face vertices.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'face_vertices'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		(face_vertices, self.map_gathered_to_unified_faces, self.map_unified_to_gathered_faces, self.counts_repeated_gathered_faces, self.map_local_to_unified_faces) = self._computeEntityVertexMapping('faces')
		return face_vertices

	@reuse_computed_value_if_possible('edge_vertices')
	def edge_vertices(self):
		"""
		Edge vertices.
		"""
		edge_vertices, self.map_gathered_to_unified_edges, self.map_unified_to_gathered_edges, self.counts_repeated_gathered_edges, self.map_local_to_unified_edges = self._computeEntityVertexMapping('edges')
		return edge_vertices

	########################### Entity values ##############################

	def _computeEntityVertexMapping(self, entity_name):
		"""
		Compute local entity vertex mapping.
		"""

		# Data type of the indices
		dtype_ = self.mesh.cells().dtype.name

		# Local entity vertices
		if entity_name == 'edges':
			local_entity_vertices = np.array([edge.entities(0) for edge in edges(self.mesh)])
		elif entity_name == 'faces':
			local_entity_vertices = np.array([face.entities(0) for face in faces(self.mesh)])
		elif entity_name == 'cells':
			local_entity_vertices = self.mesh.cells()
		else:
			raise NotImplementedError

		if runningInParallel():

			# Get the vertex map
			(map_gathered_to_unified_vertices, map_unified_to_gathered_vertices) = self.vertex_map_gathered_unified()
			map_local_to_unified_vertices = self.map_local_to_unified_vertices

			# Convert local to unified vertices
			for i in range(len(local_entity_vertices)):
				for j in range(len(local_entity_vertices[i])):
					local_entity_vertices[i][j] = map_local_to_unified_vertices[local_entity_vertices[i][j]]

			# Gather the entity
			gathered_entity_vertices = evaluateBetweenProcessors(
				local_entity_vertices,
				operation = 'append', 
				proc_destination = self.proc_destination, 
				axis = 0
			)

			# Unify the entity
			if self.proc_destination == 'all' or runningInFirstProcessor():

				# Acquire lock
				if self.proc_destination != 'all' and runningInFirstProcessor():
					acquireFirstProcessorLock()

				(unified_entity_vertices, map_gathered_to_unified_entities, map_unified_to_gathered_entities, counts_repeated_gathered_entities, map_local_to_unified_entities) = map_values_between_gathered_and_unified(gathered_entity_vertices, local_entity_vertices, dtype = dtype_)

				# Release lock
				if self.proc_destination != 'all' and runningInFirstProcessor():
					releaseFirstProcessorLock()

			else:
				unified_entity_vertices = None

		else:
			unified_entity_vertices = local_entity_vertices
			map_gathered_to_unified_entities = None
			map_unified_to_gathered_entities = None
			counts_repeated_gathered_entities = None
			map_local_to_unified_entities = None

		return unified_entity_vertices, map_gathered_to_unified_entities, map_unified_to_gathered_entities, counts_repeated_gathered_entities, map_local_to_unified_entities

	################## Entity maps: Gathered <-> Unified ###################

	@reuse_computed_value_if_possible('cell_map_gathered_unified')
	def cell_map_gathered_unified(self):
		"""
		Create a cell map between gathered and unified cells.
		"""
		cell_vertices = self.cell_vertices()
		return self.map_gathered_to_unified_cells, self.map_unified_to_gathered_cells

	@reuse_computed_value_if_possible('face_map_gathered_unified')
	def face_map_gathered_unified(self):
		"""
		Create a face map between gathered and unified faces.
		"""
		face_vertices = self.face_vertices()
		return self.map_gathered_to_unified_faces, self.map_unified_to_gathered_faces

	@reuse_computed_value_if_possible('edge_map_gathered_unified')
	def edge_map_gathered_unified(self):
		"""
		Create an edge map between gathered and unified edges.
		"""
		edge_vertices = self.edge_vertices()
		return self.map_gathered_to_unified_edges, self.map_unified_to_gathered_edges

	@reuse_computed_value_if_possible('vertex_map_gathered_unified')
	def vertex_map_gathered_unified(self):
		"""
		Create a edge map between gathered and unified vertices.
		"""
		mesh_coordinates = self.coordinates()
		return self.map_gathered_to_unified_vertices, self.map_unified_to_gathered_vertices

	@reuse_computed_value_if_possible('facet_map_gathered_unified')
	def facet_map_gathered_unified(self):
		"""
		Create a facet map between gathered and unified facets.
		"""
		dim_facets = self.topology().dim() - 1
		if dim_facets == 1:
			(self.map_gathered_to_unified_facets, self.map_unified_to_gathered_facets) = self.edge_map_gathered_unified()
		elif dim_facets == 2:
			(self.map_gathered_to_unified_facets, self.map_unified_to_gathered_facets) = self.face_map_gathered_unified()
		else:
			raise ValueError(" ❌ ERROR: Can not determine Facet for dim_facets = '%d'" %(entity_dim))

		return self.map_gathered_to_unified_facets, self.map_unified_to_gathered_facets

	############################# Entities #################################

	@reuse_computed_value_if_possible('get_cells')
	def get_cells(self):
		"""
		Return the cell entities.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'get_cells'..." %(self.__class__.__name__))
		if runningInParallel():
			num_cells_ = self.num_cells()
			cells_ = UnifiedIterator([UnifiedCell(self, i) for i in range(num_cells_)])
		else:
			cells_ = cells(self.mesh)

		return cells_

	@reuse_computed_value_if_possible('get_faces')
	def get_faces(self):
		"""
		Return the face entities.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'get_faces'..." %(self.__class__.__name__))
		if runningInParallel():
			num_faces_ = self.num_faces()
			faces_ = UnifiedIterator([UnifiedFace(self, i) for i in range(num_faces_)])
		else:
			faces_ = faces(self.mesh)

		return faces_

	@reuse_computed_value_if_possible('get_edges')
	def get_edges(self):
		"""
		Return the edge entities.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'get_edges'..." %(self.__class__.__name__))
		if runningInParallel():
			num_edges_ = self.num_edges()
			edges_ = UnifiedIterator([UnifiedEdge(self, i) for i in range(num_edges_)])
		else:
			edges_ = edges(self.mesh)

		return edges_

	@reuse_computed_value_if_possible('get_vertices')
	def get_vertices(self):
		"""
		Return the vertex entities.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'get_vertices'..." %(self.__class__.__name__))
		if runningInParallel():
			num_vertices_ = self.num_vertices()
			vertices_ = UnifiedIterator([UnifiedVertex(self, i) for i in range(num_vertices_)])
		else:
			vertices_ = vertices(self.mesh)

		return vertices_

	@reuse_computed_value_if_possible('get_facets')
	def get_facets(self):
		"""
		Return the vertex entities.
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'get_facets'..." %(self.__class__.__name__))
		if runningInParallel():
			num_facets_ = self.num_facets()
			facets_ = UnifiedIterator([UnifiedFacet(self, i) for i in range(num_facets_)])
		else:
			facets_ = facets(self.mesh)

		return facets_

	########################## Mesh properties #############################

	@reuse_computed_value_if_possible('hmin')
	def hmin(self):
		"""
		-> From FEniCS 'mesh.hmin' function
		"""

		utils.printDebug(" ⚙️ [%s] Preparing 'hmin'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)

		if runningInParallel():
			hmin = evaluateBetweenProcessors(self.mesh.hmin(), operation = 'minimum', proc_destination = self.proc_destination)
		else:
			hmin = self.mesh.hmin()

		return hmin

	######################### Numbers of entities ##########################

	@reuse_computed_value_if_possible('num_cells')
	def num_cells(self):
		"""
		Number of cells.
		-> From FEniCS 'mesh.num_cells' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'num_cells'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		if runningInParallel():
			num_cells = len(self.cells())
		else:
			num_cells = self.mesh.num_cells()
		return num_cells

	@reuse_computed_value_if_possible('num_edges')
	def num_edges(self):
		"""
		Number of edges.
		-> From FEniCS 'mesh.num_edges' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'num_edges'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		if runningInParallel():
			num_edges = len(self.edge_vertices())
		else:
			num_edges = self.mesh.num_edges()
		return num_edges

	@reuse_computed_value_if_possible('num_faces')
	def num_faces(self):
		"""
		Number of faces.
		-> From FEniCS 'mesh.num_faces' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'num_faces'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		if runningInParallel():
			num_faces = len(self.face_vertices())
		else:
			num_faces = self.mesh.num_faces()
		return num_faces

	@reuse_computed_value_if_possible('num_vertices')
	def num_vertices(self):
		"""
		Number of vertices.
		-> From FEniCS 'mesh.num_vertices' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'num_vertices'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		if runningInParallel():
			num_vertices = len(self.coordinates())
		else:
			num_vertices = self.mesh.num_vertices()
		return num_vertices

	@reuse_computed_value_if_possible('num_facets')
	def num_facets(self):
		"""
		Number of facets.
		-> From FEniCS 'mesh.num_facets' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'num_facets'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)
		dim_facets = self.topology().dim() - 1
		if dim_facets == 1:
			return self.num_edges()
		elif dim_facets == 2:
			return self.num_faces()
		else:
			raise ValueError(" ❌ ERROR: Can not determine Facet for dim_facets = '%d'" %(entity_dim))

############################ UnifiedMeshFunction ###############################

class UnifiedMeshFunction():
	"""
	A unified MeshFunction structure that deals with parallel access to mesh entities.
	The idea is to use this when dealing with parallelism.
	-> This class only works for MeshFunctions of Facets.
	-> From FEniCS 'MeshFunction' object
	"""

	def __init__(self, mesh_function, unified_mesh, prepare_beforehand = [], proc_destination = 'all'):

		utils.printDebug(" ⚙️ [%s] Creating %s..." %(self.__class__.__name__, self.__class__.__name__), mpi_wait_for_everyone = True)

		self.mesh_function = mesh_function 
		self.unified_mesh = unified_mesh
		self.proc_destination = proc_destination

		for i in range(len(prepare_beforehand)):
			if type(prepare_beforehand[i]).__name__ == 'str':
				name_method_preparation = prepare_beforehand[i]
				args_preparation = []
			else:
				name_method_preparation = prepare_beforehand[i][0]
				args_preparation = prepare_beforehand[i][1]

				if type(args_preparation).__name__ not in ['list', 'tuple']:
					args_preparation = [args_preparation]

			# Compute method from name and arguments
			getattr(self, name_method_preparation)(*args_preparation)

	@reuse_computed_value_if_possible('array')
	def array(self):
		"""
		Array.
		-> From FEniCS 'MeshFunction.array' function
		"""

		utils.printDebug(" ⚙️ [%s] Preparing 'array'..." %(self.__class__.__name__), mpi_wait_for_everyone = True)

		# Only implemented for Facets
		assert self.mesh_function.dim() == self.unified_mesh.topology().dim() - 1

		# Local array
		local_array = self.mesh_function.array()

		# Gather the entity
		gathered_array = evaluateBetweenProcessors(
			local_array,
			operation = 'append', 
			proc_destination = self.proc_destination, 
			axis = 0
		)

		# Convert to unified facets
		(map_gathered_to_unified_facets, map_unified_to_gathered_facets) = self.unified_mesh.facet_map_gathered_unified()
		unified_array = gathered_array[map_gathered_to_unified_facets]

		return unified_array

	def where_equal(self, value):
		"""
		Locate where a value is. 
		-> From FEniCS 'MeshFunction.where_equal' function
		"""
		return np.where(self.array() == value)[0].tolist()

	def set_values(self, array_of_values):
		"""
		Set values.
		-> From FEniCS 'MeshFunction.set_values' function
		"""

		# Unified facets
		unified_array = array_of_values
	
		# Data type of the indices
		dtype_ = self.unified_mesh.mesh.cells().dtype.name

		# Convert to unified facets
		(map_gathered_to_unified_facets, map_unified_to_gathered_facets) = self.unified_mesh.facet_map_gathered_unified()
		gathered_array = unified_array[map_unified_to_gathered_facets]

		# Local -> Unified
		previous_local_array = self.mesh_function.array()
		new_local_array = getLocalArrayFromGatheredArray(gathered_array, len(previous_local_array), dtype = dtype_)

		# Set local
		self.mesh_function.set_values(new_local_array)

############################## UnifiedEntity ###################################

class UnifiedEntity():
	"""
	Unified entity for the UnifiedMesh.
	-> From FEniCS 'Cell' function
	"""

	def __init__(self, unified_mesh, index, dim):

		self.unified_mesh = unified_mesh
		self._index = index
		self._dim = dim
		self._name = self.unified_mesh.map_entity_dim_to_name(self._dim)

	def index(self):
		"""
		Cell index.
		-> From FEniCS 'Cell.index' function
		"""
		return self._index

	def dim(self):
		"""
		Entity dimension.
		-> From FEniCS 'Cell.dim' function
		"""
		return self._dim

	def entities(self, entity_dim):
		"""
		Connected entities.
		-> From FEniCS 'Cell.entities' function
		"""
		entity_name = self.unified_mesh.map_entity_dim_to_name(entity_dim)

		name_entities_mapping = "%s-%s" %(self._name, entity_name)

		if name_entities_mapping not in self.unified_mesh.entity_mapping_data:
			raise ValueError(" ❌ ERROR: Entity mapping data '%s' (%d-%d) has not been initialized yet!" %(name_entities_mapping, self._dim, entity_dim))

		entities = self.unified_mesh.entity_mapping_data[name_entities_mapping][self._index]
		return entities

	def midpoint(self):
		"""
		Midpoint.
		-> From FEniCS 'Cell.midpoint' function
		"""
		coords = self.unified_mesh.coordinates()
		vertices_ = self.entities(0)
		return Point(sum([coords[vertex] for vertex in vertices_])/len(vertices_))

class UnifiedCell(UnifiedEntity):
	"""
	Cell entity for the UnifiedMesh.
	-> From FEniCS 'Cell' function
	"""
	def __init__(self, unified_mesh, index):
		super().__init__(unified_mesh, index, unified_mesh.topology().dim())

class UnifiedFace(UnifiedEntity):
	"""
	Face entity for the UnifiedMesh.
	-> From FEniCS 'Face' function
	"""
	def __init__(self, unified_mesh, index):
		super().__init__(unified_mesh, index, 2)

class UnifiedEdge(UnifiedEntity):
	"""
	Edge entity for the UnifiedMesh.
	-> From FEniCS 'Edge' function
	"""
	def __init__(self, unified_mesh, index):
		super().__init__(unified_mesh, index, 1)

class UnifiedVertex(UnifiedEntity):
	"""
	Vertex entity for the UnifiedMesh.
	-> From FEniCS 'Vertex' function
	"""
	def __init__(self, unified_mesh, index):
		super().__init__(unified_mesh, index, 0)

class UnifiedFacet(UnifiedEntity):
	"""
	Facet entity for the UnifiedMesh.
	-> From FEniCS 'Facet' function
	"""
	def __init__(self, unified_mesh, index):
		super().__init__(unified_mesh, index, unified_mesh.topology().dim() - 1)

############################# UnifiedIterator ##################################

class UnifiedIterator():
	"""
	Unified iterator.
	-> From FEniCS iterators (i.e., 'CellIterator', 'FaceIterator', 'VertexIterator', 'FacetIterator')
	https://stackoverflow.com/questions/19151/build-a-basic-python-iterator
	"""
	def __init__(self, object_list):
		self.object_list = object_list
		self.counter = 0
		self.final_counter = len(self.object_list) - 1

	def __iter__(self):
		"""
		Return the iterator object.
		"""
		return self

	def __next__(self): 
		"""
		Return the next value.
		"""
		if self.counter <= self.final_counter:

			# Next object
			next_object = self.object_list[self.counter]

			# Increment iterator count
			self.counter += 1

			# Return the object
			return next_object
		else:

			# Reinitialize iterator count
			 # *  This is essential, otherwise this iterator 
			 #    would only be usable once.
			 # ** Iterators generated with the Python 'iter' function,
			 #    and other 'common' iterators, are only usable once.
			self.counter = 0

			# Stop iteration
			raise StopIteration

##################### get_cells, get_faces, get_edges ##########################

def get_cells(mesh):
	"""
	Get all Cell entities for the UnifiedMesh.
	-> From FEniCS 'cells' function
	"""

	if 'Unified' in type(mesh).__name__:
		mesh_cells = mesh.get_cells()
	else:
		mesh_cells = fenics.cells(mesh)

	return mesh_cells

def get_faces(mesh):
	"""
	Get all Face entities for the UnifiedMesh.
	-> From FEniCS 'faces' function
	"""

	if 'Unified' in type(mesh).__name__:
		mesh_faces = mesh.get_faces()
	else:
		mesh_faces = fenics.faces(mesh)

	return mesh_faces

def get_edges(mesh):
	"""
	Get all Edge entities for the UnifiedMesh.
	-> From FEniCS 'edges' function
	"""

	if 'Unified' in type(mesh).__name__:
		mesh_edges = mesh.get_edges()
	else:
		mesh_edges = fenics.edges(mesh)

	return mesh_edges

def get_vertices(mesh):
	"""
	Get all Vertex entities for the UnifiedMesh.
	-> From FEniCS 'vertices' function
	"""

	if 'Unified' in type(mesh).__name__:
		mesh_vertices = mesh.get_vertices()
	else:
		mesh_vertices = fenics.vertices(mesh)

	return mesh_vertices

def get_facets(mesh):
	"""
	Get all Facet entities for the UnifiedMesh.
	-> From FEniCS 'facets' function
	"""

	if 'Unified' in type(mesh).__name__:
		mesh_facets = mesh.get_facets()
	else:
		mesh_facets = fenics.facets(mesh)

	return mesh_facets

############################## UnifiedTopology #################################

class UnifiedTopology():
	"""
	-> From FEniCS 'mesh.topology' object
	"""
	def __init__(self, unified_mesh):

		self.unified_mesh = unified_mesh
		self.mesh = unified_mesh.mesh
		self.proc_destination = unified_mesh.proc_destination

		# Prepare dim beforehand
		self.dim()

	@reuse_computed_value_if_possible('dim')
	def dim(self):
		return self.mesh.topology().dim()
	
############################ UnifiedBoundaryMesh ###############################

class UnifiedBoundaryMesh(UnifiedMesh):
	"""
	A unified boundary mesh structure that deals with parallel access to mesh entities.
	The idea is to use this when dealing with parallelism.
	-> From FEniCS 'BoundaryMesh' object
	"""

	def __init__(self, bmesh, unified_mesh, prepare_beforehand = [], proc_destination = 'all'):

		self.orig_unified_mesh = unified_mesh
		self.orig_mesh = unified_mesh.mesh
		self.bmesh = bmesh

		super().__init__(bmesh, prepare_beforehand = prepare_beforehand, proc_destination = proc_destination)

		if self.orig_mesh.topology().dim() == 3:
			self.init(2,0) # face-vertex
		elif self.orig_mesh.topology().dim() == 2:
			self.init(1,0) # edge-vertex
		else:
			raise ValueError(" ❌ ERROR: Can not determine for self.orig_mesh.topology().dim() = '%d'" %(self.orig_mesh.topology().dim()))

	############################# entity_map ###############################

	def entity_map(self, entity_dim):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		if entity_dim == 0:
			return self.entity_map0()
		elif entity_dim == 1:
			return self.entity_map1()
		elif entity_dim == 2:
			return self.entity_map2()
		elif entity_dim == 3:
			return self.entity_map3()
		else:
			raise ValueError(" ❌ ERROR: Can not determine for entity_dim = '%d'" %(entity_dim))

	@reuse_computed_value_if_possible('entity_map0')
	def entity_map0(self):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		return self._entity_map(0)

	@reuse_computed_value_if_possible('entity_map1')
	def entity_map1(self):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		return self._entity_map(1)

	@reuse_computed_value_if_possible('entity_map2')
	def entity_map2(self):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		return self._entity_map(2)

	@reuse_computed_value_if_possible('entity_map3')
	def entity_map3(self):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		return self._entity_map(3)

	def _entity_map(self, entity_dim):
		"""
		Entity map.
		-> From FEniCS 'mesh.entity_map' function
		"""
		utils.printDebug(" ⚙️ [%s] Preparing 'entity_map' for entity_dim = %d..." %(self.__class__.__name__, entity_dim), mpi_wait_for_everyone = True)
		if runningInParallel():
			entity_map = UnifiedBoundaryMeshEntityMap(self, entity_dim)
		else:
			entity_map = self.bmesh.entity_map(entity_dim)

		return entity_map

######################### UnifiedBoundaryMeshEntityMap #########################

class UnifiedBoundaryMeshEntityMap():
	"""
	Unified entity map.
	-> From FEniCS 'bmesh.entity_map(entity_dim)'
	"""
	def __init__(self, unified_bmesh, entity_dim):

		self.unified_bmesh = unified_bmesh
		self.entity_dim = entity_dim
		self.proc_destination = self.unified_bmesh.proc_destination

		# Compute the array NOW
		self.array()

	@reuse_computed_value_if_possible('array')
	def array(self):
		"""
		Array of the entity map.
		-> From FEniCS 'bmesh.entity_map(entity_dim).array()' function
		"""

		# Data type of the indices
		dtype_ = self.unified_bmesh.orig_unified_mesh.mesh.cells().dtype.name

		if runningInParallel():

			# Get the maps
			if self.entity_dim == 0:
				mesh_map_local_to_unified_entities = self.unified_bmesh.orig_unified_mesh.map_local_to_unified_vertices
				bmesh_map_gathered_to_unified_entities = self.unified_bmesh.map_gathered_to_unified_vertices
			elif self.entity_dim == 1:
				mesh_map_local_to_unified_entities = self.unified_bmesh.orig_unified_mesh.map_local_to_unified_edges
				bmesh_map_gathered_to_unified_entities = self.unified_bmesh.map_gathered_to_unified_edges
			elif self.entity_dim == 2:
				mesh_map_local_to_unified_entities = self.unified_bmesh.orig_unified_mesh.map_local_to_unified_faces
				bmesh_map_gathered_to_unified_entities = self.unified_bmesh.map_gathered_to_unified_faces
			elif self.entity_dim == 3:
				mesh_map_local_to_unified_entities = self.unified_bmesh.orig_unified_mesh.map_local_to_unified_cells
				bmesh_map_gathered_to_unified_entities = self.unified_bmesh.map_gathered_to_unified_cells
			else:
				raise ValueError(" ❌ ERROR: self.entity_dim = '%s' is not defined here!" %(self.entity_dim))

			# Local array
			local_array = self.unified_bmesh.bmesh.entity_map(self.entity_dim).array()

			# Convert local content to unified content
			for i in range(len(local_array)):
				local_array[i] = mesh_map_local_to_unified_entities[local_array[i]]

			# Gather the entity
			gathered_array = evaluateBetweenProcessors(
				local_array,
				operation = 'append', 
				proc_destination = self.proc_destination, 
				axis = 0
			)

			# Unified array
			unified_array = gathered_array[bmesh_map_gathered_to_unified_entities]

		else:
			unified_array = self.local_entity_map.array()

		return unified_array

################### map_values_between_gathered_and_unified ####################

def map_values_between_gathered_and_unified(gathered_values, local_values, dtype = 'uint32'):
	"""
	Map values between gathered and unified.
	"""

	# Gathered <-> Unified
	(unified_values, map_gathered_to_unified_values, map_unified_to_gathered_values, counts_repeated_gathered_values) = np.unique(gathered_values, axis = 0, return_index = True, return_inverse = True, return_counts = True)
		# * gathered_values[map_gathered_to_unified_values] = unified_values
		#   unified_values[map_unified_to_gathered_values] = gathered_values

	# Local -> Unified
	map_local_to_unified_values = getLocalArrayFromGatheredArray(map_unified_to_gathered_values, len(local_values), dtype = dtype)

	return (unified_values, map_gathered_to_unified_values, map_unified_to_gathered_values, counts_repeated_gathered_values, map_local_to_unified_values)

################## getDisplacedIndicesForGatheredArray #########################

def getDisplacedIndicesForGatheredArray(local_array_length, dtype = 'uint32'):
	"""
	Get the displaced indices that define the local array inside the gathered array.
	"""

	# Local displacements
	local_displacements = evaluateBetweenProcessors(
		np.array([local_array_length], dtype = dtype),
		operation = 'append', 
		proc_destination = 'all', 
		axis = 0
		)

	# Start and finish in the gathered array
	reference_index_for_gathered = int(np.sum(local_displacements[:rank]))
	index_start = 0 + reference_index_for_gathered
	index_finish = local_array_length + reference_index_for_gathered

	return index_start, index_finish

#################### getLocalArrayFromGatheredArray ############################

def getLocalArrayFromGatheredArray(gathered_array, local_array_length, dtype = 'uint32'):
	"""
	Scatter the gathered array.
	Get the local array from the gathered array.
	"""

	# Scatter: Map gathered array to local array
	(index_start, index_finish) = getDisplacedIndicesForGatheredArray(local_array_length, dtype = dtype)
	local_array = gathered_array[index_start:index_finish]

	return local_array

################################################################################
################# Unified Function structures for parallelism ##################
################################################################################

######################### getGatheredDoFCoordinates ############################

def getGatheredDoFCoordinates(U, return_local_maps = False):
	"""
	Return the gathered DoF coordinates.
	-> In this case,"gathered" = "global" 
	-> Assuming no one is using ghost entities or something like that. Otherwise,
	   some problem may arise due to the entities the current process does not own.
	"""

	from ..utils import utils_fenics

	# Data type of the indices
	dtype_ = U.mesh().cells().dtype.name

	# Local mesh coordinates from the function space
	local_mesh_coordinates = utils_fenics.getDoFCoordinatesFromFunctionSpace(U)

	if runningInParallel() == True:

		# Set the gathered coordinates
		DoF_mesh_coords_gathered = evaluateBetweenProcessors(
			local_mesh_coordinates, 
			operation = 'append', 
			proc_destination = 'all', 
			axis = 0
			)
	
		# Map local to gathered DoFs
		(index_start, index_finish) = getDisplacedIndicesForGatheredArray(len(local_mesh_coordinates), dtype = dtype_)
		map_local_to_gathered_DoFs = np.arange(index_start, index_finish)

		# Map local to global DoFs
		map_local_to_global_DoFs = map_local_to_gathered_DoFs

	else:
		map_local_to_global_DoFs = None
		DoF_mesh_coords_gathered = DOF_mesh_coords_local

	if return_local_maps == True:

		# Map global to local DoFs
		if type(map_local_to_global_DoFs).__name__ != 'NoneType':
			map_global_to_local_DoFs = utils_fenics.invertMapOrder(map_local_to_global_DoFs)
		else:
			map_global_to_local_DoFs = None

		return DoF_mesh_coords_gathered, map_local_to_global_DoFs, map_global_to_local_DoFs
	else:
		return DoF_mesh_coords_gathered

#################### getFunctionLocalArraysFromGatheredArray ###################

def getFunctionLocalArraysFromGatheredArray(u_array_global, U):
	"""
	Scatter the gathered array.
	Get the local array from the gathered array. 
	"""

	from ..utils import utils_fenics

	# Data type of the indices
	dtype_ = U.mesh().cells().dtype.name

	# Local mesh coordinates from the function space
	num_local_values = len(utils_fenics.getDoFCoordinatesFromFunctionSpace(U))

	# Scatter: Map gathered array to local array
	(index_start, index_finish) = getDisplacedIndicesForGatheredArray(num_local_values, dtype = dtype_)
	u_array_local = u_array_global[index_start:index_finish]

	return u_array_local


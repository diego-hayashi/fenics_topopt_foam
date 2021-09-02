################################################################################
#                         SuccessiveSimulationParameter                        #
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

############################# Project libraries ################################

# Utilities
from ..utils import utils

######################### SuccessiveSimulationParameter ########################

class SuccessiveSimulationParameter():
	"""
	Successive simulation parameter.
	When you want to successively change a parameter in subsequent simulations. 
	Use this inside a configuration or property dictionary for OpenFOAM.
	You may combine as many of these special parameters as you want.

	Example if you want to successively run: 

	1) First with 'GAMG' and then with 'PCG': (for the "fvSolution" dictionary)

		'solver' : fenics_topopt_foam.SuccessiveSimulationParameter('GAMG', 'PCG'),

	2) First with 'GAMG', then with 'PCG' and then with 'smoothSolver': (for the "fvSolution" dictionary)

		'solver' : fenics_topopt_foam.SuccessiveSimulationParameter('GAMG', 'PCG', 'smoothSolver'),

	3) Successively run with different kinematic viscosities: (for the "transportProperties" dictionary)

		'nu' : fenics_topopt_foam.SuccessiveSimulationParameter(0.1, 0.01, 0.001),

	* Hint: If you want to set a list of successive parameters such as:

		list_of_values = [0.1, 0.01, 0.001]

	it can not be provided ``as is'': it needs to be unfolded by using a preceding asterisk character, such as in:

		'nu' : fenics_topopt_foam.SuccessiveSimulationParameter(*list_of_values),

	or:

		'nu' : fenics_topopt_foam.SuccessiveSimulationParameter(*[0.1, 0.01, 0.001]),

	"""

	def __init__(self, *args):

		assert len(args) > 0 # We need at least one value here!

		self.list_of_values = args
		self.current_index = 0
		self.reached_end_flag = False
		self._tag = ""

	def set_tag(self, tag_name):
		"""
		Sets a tag name for printing.
		"""
		self._tag = "%s | " %(tag_name)

	def restart(self):
		"""
		Restart the count.
		"""
		self.current_index = 0
		self.reached_end_flag = False

	def checkIfUpdatable(self):
		"""
		Check if it is possible to update this value or not.
		"""
		return not self.reached_end_flag

	def value(self):
		"""
		Return the current value and update.
		"""
		current_value = self.list_of_values[self.current_index]

		if self.current_index == 0:
			if self.size == 1:
				self.reached_end_flag = True
			else:
				utils.customPrint(" 🚩️ [%sValue count %d/%d] Setting '%s'..." %(self._tag, self.current_index + 1, self.size, current_value))
				self.current_index += 1

		elif self.current_index < len(self.list_of_values) - 1:

			previous_value = self.list_of_values[self.current_index - 1]
			utils.customPrint(" 🚩️ [%sValue count %d/%d] Changing '%s' to '%s'..." %(self._tag, self.current_index + 1, self.size, previous_value, current_value))

			self.current_index += 1
		else:
			if self.reached_end_flag == False:
				previous_value = self.list_of_values[self.current_index - 1]
				utils.customPrint(" 🚩️ [%sValue count %d/%d] Changing '%s' to '%s'..." %(self._tag, self.current_index + 1, self.size, previous_value, current_value))

			self.reached_end_flag = True

		self.current_value = current_value
		return current_value

	def checkNextValue(self):
		"""
		Check the next value.
		"""

		next_index = self.current_index
		next_value = self.list_of_values[next_index]

		if next_index == 0:
			utils.customPrint(" 🚩️ > [%sValue count  %d/%d] Value '%s' to be set" %(self._tag, next_index + 1, self.size, next_value))
		else: # self.current_index < len(self.list_of_values) - 1:
			if self.reached_end_flag == True:
				utils.customPrint(" 🚩️ > [%sValue count %d/%d] Value '%s' is already set" %(self._tag, next_index + 1, self.size, next_value))
			else:
				previous_value = self.list_of_values[next_index - 1]
				utils.customPrint(" 🚩️ > [%sValue count  %d/%d] Value '%s' will be changed to '%s'" %(self._tag, next_index + 1, self.size, previous_value, next_value))

	@property
	def size(self):
		"""
		Size of the list of values.
		"""
		return len(self.list_of_values)

	def __copy__(self):
		"""
		For copy with "copy.copy".
		  https://stackoverflow.com/questions/4794244/how-can-i-create-a-copy-of-an-object-in-python
		"""
		return type(self)(*self.list_of_values)

	def __deepcopy__(self, memo):
		"""
		For deepcopy with "copy.deepcopy".
		* 'memo' is a dictionary of id's to copies.
		  https://stackoverflow.com/questions/4794244/how-can-i-create-a-copy-of-an-object-in-python
		"""

		import copy
		id_self = id(self)
		_copy = memo.get(id_self)
		if _copy is None:
			_copy = type(self)(*copy.deepcopy(self.list_of_values))
			memo[id_self] = _copy 
		return _copy


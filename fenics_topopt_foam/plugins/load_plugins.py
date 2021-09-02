################################################################################
#                                 load_plugins                                 #
################################################################################
# Load plugins for overloading or extending FEniCS TopOpt Foam functionality

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

# FEniCS+MPI utilities
from ..utils import utils_fenics_mpi

# OpenFOAM information
from ..utils import foam_information

# FEniCS TopOpt Foam folder
from .. import __fenics_topopt_foam_folder__

############################# Global variables #################################

global _load_plugins, _plugins_dict, _plugins_dict_order, _overload_keyring

# Flag to load plugins
 # If you want the documentation without the plugins,
 #  set the following before calling FEniCS TopOpt Foam: 
  #
  # $ export FENICS_TOPOPT_FOAM_LOAD_PLUGINS="False"
  #
loaded_flag_plugins_ = utils.run_command_in_shell("echo -n $FENICS_TOPOPT_FOAM_LOAD_PLUGINS", mode = 'save output to variable', suppress_run_print = True, accept_empty_response = True)
if loaded_flag_plugins_ == "False":
	_load_plugins = False
else:
	_load_plugins = True

# Plugins dictionary
_plugins_dict = {}
_plugins_dict_order = []

# Overload keyring (for avoiding cyclic imports)
_overload_keyring = []

############################### loadPlugins ####################################

def loadPlugins(current_file_name, global_variables):
	"""
	Load plugins!

	-> Currently, plugins can extend/overload only:
		'FEniCSFoamSolver'
		'FoamSolver'

	-> If needed, more may be added to the above list.

	-> To include new plugins:
		-> Decide on a plugin name without white spaces, special characters etc.
		-> Create a folder with that name.
		-> Inside the plugin folder, create the "__init__.py" file.
		-> Try following "more or less" what the other plugin(s) do.

	-> This plugin engine does not offer any protection when the plugin
	   is broken. So care should be taken not to break the code when
	   implementing a plugin.
	"""

	# [Developer information -- Just as a reminder for me...] -> Include:
#
# ############################## Load plugins ####################################
#
# from ..plugins import load_plugins
# load_plugins.loadPlugins(__file__, globals())
#
# ################################################################################
#
	# in the end of ALL files of FEniCS TopOpt Foam that may be extended by
	# plugins from this plugin engine!

	module_being_overloaded = global_variables

	global _plugins_dict, _plugins_dict_order, _load_plugins, _overload_keyring

	################# If it is not meant to load plugins ###################
	if _load_plugins == False:
		return # No plugins shall be loaded!

	####################### Check for cyclic import! #######################

	if current_file_name in _overload_keyring:
		raise ValueError(" ❌ ERROR: Cyclic import detected for file '%s'! A plugin is probably importing the file in which the overloaded function is! In order to avoid this, please use the 'module_being_overloaded' input parameter in the overload methods/functions!" %(current_file_name))
	else:
		_overload_keyring += [current_file_name]

	########################### Load all plugins ###########################

	if len(_plugins_dict) == 0:
			
		# Locate all folders here
		subfolder_names = utils.getSubfolderNames('%s/plugins' %(__fenics_topopt_foam_folder__))

		# Remove folders that start with "__"
		subfolder_names_new = [subfolder for subfolder in subfolder_names if not(subfolder.startswith("__"))]
		subfolder_names = subfolder_names_new

		plugin_names = subfolder_names

		if len(plugin_names) > 1:			
			utils.printDebug("\n 🌀 [Plugins] Plugins to load: %s" %(plugin_names))
		else:
			utils.printDebug("\n 🌀 [Plugins] No plugins to load!")

		# Prepare the plugin data
		for plugin_name in plugin_names:
			_setPluginData(plugin_name)

		#### Add in "_plugins_dict_order" from order of priority

		# List of plugin names
		all_plugins_names = list(_plugins_dict.keys())

		# Get the priorities of the plugins
		all_priorities = [_plugins_dict[plugin_name_]['module'].plugin_priority for plugin_name_ in all_plugins_names]

		# Sort in reverse order (i.e., a higher priority value means that it will be loaded first)
		sorted_priority_indicies = sorted(range(len(all_priorities)), key = lambda k: all_priorities[k], reverse = True)
			# https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
		_plugins_dict_order = [all_plugins_names[i_] for i_ in sorted_priority_indicies]

	######################### Perform the overloads ########################

	loaded_plugin_names = []
	for plugin_name in _plugins_dict_order:

		# Dictionary of the plugin
		plugin_dict = _plugins_dict[plugin_name]

		# Plugin overloads
		plugin_overloads = plugin_dict['plugin overloads']

		for plugin_overload in plugin_overloads:

			# If the file is the same
			if plugin_overload['file'] == current_file_name:
				_overloadFromPluginOverload(plugin_name, plugin_overload, module_being_overloaded)
			else:
				pass

		# Loaded plugin names
		loaded_plugin_names += [plugin_name]

	########################## Remove from keyring #########################

	_overload_keyring.remove(current_file_name)

	# Finished!
	return

######################## _overloadFromPluginOverload ###########################

def _overloadFromPluginOverload(plugin_name, plugin_overload, module_being_overloaded):
	"""
	Overload from plugin overload definition.
	"""

	type_of_overload = plugin_overload['variable to overload']['type']

	assert type_of_overload in plugin_overload['variable to overload']

	###################### Overload method from class ######################
	if type_of_overload == 'overload method from class':

		# Plugin
		class_name = plugin_overload['variable to overload'][type_of_overload]['class name']
		method_name = plugin_overload['variable to overload'][type_of_overload]['method name']
		plugin_method = plugin_overload['variable to overload'][type_of_overload]['overloader method']

		# In the case of a "private" method
		method_name = _adjustNameForPrivate(method_name, class_name)

		# Get the original method
		original_class = module_being_overloaded[class_name]
		assert method_name in original_class.__dict__, " ❌ ERROR: Method '%s' is not available in class '%s'. Therefore, it can not be overloaded!" %(method_name, class_name)
		original_method = original_class.__dict__[method_name]

		# [Include the new method] In this case, the new method
		 # receives the original method as an input argument,
		 # in order to allow "cascading" the original class with plugins.
		#original_class.__dict__[method_name] = lambda self_, *args, **kwargs: plugin_method(self_, original_method, module_being_overloaded, *args, **kwargs)
		_setToClass(original_class, module_being_overloaded, name_ = method_name, new_ = plugin_method, type_ = 'overload method')

		# [Method docstring] Cascade the docstrings, because we don't want to lose documentation
		_appendTextToDocstring(original_class.__dict__[method_name],
			new_text = """[Plugin '%s']
%s
""" %(plugin_name, plugin_method.__doc__))

		# [Class docstring]
		_appendTextToDocstring(original_class,
			new_text = " + Plugin '%s' overloading '%s'\n" %(plugin_name, method_name))

		# Additional docstring update
		if 'add to method docstring' in plugin_overload['variable to overload'][type_of_overload]:
			additional_method_name = plugin_overload['variable to overload'][type_of_overload]['add to method docstring']

			# [Method docstring] Cascade the docstrings, because we don't want to lose documentation
			_appendTextToDocstring(original_class.__dict__[additional_method_name],
				new_text = """[Plugin '%s']
%s
""" %(plugin_name, plugin_method.__doc__))

	###################### New method for class ############################
	elif type_of_overload == 'new method for class':

		# Plugin
		class_name = plugin_overload['variable to overload'][type_of_overload]['class name']
		method_name = plugin_overload['variable to overload'][type_of_overload]['method name']
		plugin_method = plugin_overload['variable to overload'][type_of_overload]['overloader method']

		# In the case of a "private" method
		method_name = _adjustNameForPrivate(method_name, class_name)

		# Get the original class
		original_class = module_being_overloaded[class_name]

		# Include the new method
		#original_class.__dict__[method_name] = lambda self_, *args, **kwargs: plugin_method(self_, module_being_overloaded, *args, **kwargs)
		_setToClass(original_class, module_being_overloaded, name_ = method_name, new_ = plugin_method, type_ = 'new method')

		# [Method docstring] Include the docstring, because we don't want to lose documentation
		_appendTextToDocstring(original_class.__dict__[method_name],
			new_text = """[Plugin '%s']
%s
""" %(plugin_name, plugin_method.__doc__))

		# [Class docstring]
		_appendTextToDocstring(original_class,
			new_text = " + Plugin '%s' with new method '%s'\n" %(plugin_name, method_name))

	###################### New variable for class ##########################
	elif type_of_overload == 'new variable for class':
		# * In contrary to 'new self.variable for class',
		 # this variable appears in the automatically generated documentation
		 # from pdoc3, as a member of the class. In 'new self.variable for class',
		 # the variable is defined only when the object is instantiated.
		 # Therefore, it does not appear in the automatically
		 # generated documentation from pdoc3.
		 # --> It is HIGHLY prefered to use 'new self.variable for class' instead!
		 #     If necessary, it is preferred to include "setters" and "getters".

		# Plugin
		class_name = plugin_overload['variable to overload'][type_of_overload]['class name']
		variable_name = plugin_overload['variable to overload'][type_of_overload]['variable name']
		plugin_variable = plugin_overload['variable to overload'][type_of_overload]['overloader variable']

		# In the case of a "private" variable
		variable_name = _adjustNameForPrivate(variable_name, class_name)

		# Get the original class
		original_class = module_being_overloaded[class_name]

		# Include the new variable
		#original_class.__dict__[variable_name] = plugin_variable
		_setToClass(original_class, module_being_overloaded, name_ = variable_name, new_ = plugin_variable, type_ = 'set variable')

	################## New self.variable for class #########################
	elif type_of_overload == 'new self.variable for class':

		# Plugin
		class_name = plugin_overload['variable to overload'][type_of_overload]['class name']
		variable_name = plugin_overload['variable to overload'][type_of_overload]['variable name']
		plugin_variable = plugin_overload['variable to overload'][type_of_overload]['overloader variable']

		# In the case of a "private" variable
		variable_name = _adjustNameForPrivate(variable_name, class_name)

		# Get the original class
		original_class = module_being_overloaded[class_name]

		# Include the new variable
		#original_class.__dict__[variable_name] = plugin_variable
		_setToClass(original_class, module_being_overloaded, name_ = variable_name, new_ = plugin_variable, type_ = 'set self.variable')

	########################### New class ##################################
	elif type_of_overload == 'new class':

		# Plugin
		class_name = plugin_overload['variable to overload'][type_of_overload]['class name']
		plugin_class = plugin_overload['variable to overload'][type_of_overload]['overloader class']

		# Assert that it still does not exist!
		assert class_name not in module_being_overloaded

		# Include the new class
		module_being_overloaded[class_name] = plugin_class

		# [Class docstring]
		original_class.__doc__ = """[Plugin '%s'] 
%s
""" %(plugin_name, plugin_class.__doc__)

	###################### Overload function ###############################
	elif type_of_overload == 'overload function':

		# Plugin
		function_name = plugin_overload['variable to overload'][type_of_overload]['function name']
		plugin_function = plugin_overload['variable to overload'][type_of_overload]['overloader function']

		# Get the original function
		original_function = module_being_overloaded[function_name]

		# Include the new function
		module_being_overloaded[function_name] = lambda *args, **kwargs: plugin_function(original_function, module_being_overloaded, *args, **kwargs)

		# [Function docstring] Include the docstring, because we don't want to lose documentation
		_appendTextToDocstring(module_being_overloaded[function_name],
			new_text = """[Plugin '%s']
%s
""" %(plugin_name, plugin_function.__doc__))

	########################## New function ################################
	elif type_of_overload == 'new function':

		# Plugin
		function_name = plugin_overload['variable to overload'][type_of_overload]['function name']
		plugin_function = plugin_overload['variable to overload'][type_of_overload]['overloader function']

		# Include the new function
		module_being_overloaded[function_name] = lambda *args, **kwargs: plugin_function(module_being_overloaded, *args, **kwargs)

		# [Function docstring] Include the docstring, because we don't want to lose documentation
		_appendTextToDocstring(module_being_overloaded[function_name],
			new_text = """[Plugin '%s']
%s
""" %(plugin_name, plugin_function.__doc__))

	else:
		raise ValueError(" ❌ ERROR: type_of_overload == '%s' is not defined!" %(type_of_overload))

############################## _setPluginData ##################################

def _setPluginData(plugin_name):
	"""
	Set the plugin data.
	"""

	global _plugins_dict

	utils.customPrint("\n 🌀 [Plugins] Loading plugin '%s'..." %(plugin_name))

	# Plugin
	plugin_file = "__init__.py"
	plugin_location = '%s/plugins/%s' %(__fenics_topopt_foam_folder__, plugin_name)
	plugin_file_location = '%s/%s' %(plugin_location, plugin_file)

	# Check if it has not been loaded before!
	assert plugin_name not in _plugins_dict, " ❌ ERROR: Plugin '%s' has already been loaded. Why are we loading it again?" %(plugin_name)

	# Inspect the data from the plugin
	_plugins_dict[plugin_name] = _getPluginDataFromFile(plugin_name, plugin_file_location)

	# Print an overview of the plugin
	utils.printDebug("""------------------------------------------------------------------------------
💠️ Loaded plugin: '%s'
📁️ Plugin folder: '%s'
------------------------------------------------------------------------------
""" % 	(
	plugin_name,
	plugin_location,
	)
	)

########################### _getPluginDataFromFile #############################

def _getPluginDataFromFile(plugin_name, file_location, accept_import_error = False):
	"""
	Get the plugin data from file.
	"""

	# Import the module
	try:
		# Import the module
		import importlib
		loaded_module = importlib.import_module(".plugins.%s" %(plugin_name), package = 'fenics_topopt_foam')
	except:
		import traceback
		traceback.print_exc()
		if accept_import_error == True:
			print(" ❌ ERROR: Unable to import plugin from '%s'! Ignoring import..." %(file_location))
		else:
			raise ImportError(" ❌ ERROR: Unable to import plugin from '%s'!" %(file_location))

	# Create the data dictionary
	dict_ = {}

	# Save the location
	dict_['file_location'] = file_location

	# Save the module
	dict_['module'] = loaded_module

	# Description
	dict_['description'] = loaded_module.__doc__

	# Plugin overloads
	dict_['plugin overloads'] = dict_['module'].plugin_overloads

	# Plugin priority
	dict_['plugin priority'] = dict_['module'].plugin_priority

	return dict_

################################## _setToClass #################################

def _setToClass(original_class, module_being_overloaded, name_ = '', new_ = None, type_ = 'overload method'):
	"""
	Set a method or a variable to a class.
	"""

	if type_ == 'overload method':

		new_method = new_
		method_name = name_

		# [Include the new method] In this case, the new method
		 # receives the original method as an input argument,
		 # in order to allow "cascading" the original class with plugins.

		original_method = original_class.__dict__[method_name]
		additional_args = [original_method, module_being_overloaded]

		# https://stackoverflow.com/questions/20019333/how-to-modify-class-dict-a-mappingproxy
		# original_class.__dict__[method_name] = lambda self_, *args, **kwargs: new_method(self_, *additional_args, *args, **kwargs)
		new_method_for_original_class = lambda self_, *args, **kwargs: new_method(self_, *additional_args, *args, **kwargs)
		setattr(original_class, method_name, new_method_for_original_class)

	elif type_ == 'new method':

		new_method = new_
		method_name = name_

		additional_args = [module_being_overloaded]

		# https://stackoverflow.com/questions/20019333/how-to-modify-class-dict-a-mappingproxy
		# original_class.__dict__[method_name] = lambda self_, *args, **kwargs: new_method(self_, *additional_args, *args, **kwargs)
		new_method_for_original_class = lambda self_, *args, **kwargs: new_method(self_, *additional_args, *args, **kwargs)
		setattr(original_class, method_name, new_method_for_original_class)

	elif type_ == 'set variable':

		new_variable = new_
		variable_name = name_

		# Include the new variable
		#original_class.__dict__[variable_name] = plugin_variable
		setattr(original_class, variable_name, new_variable)

	elif type_ == 'set self.variable':

		new_variable = new_
		variable_name = name_

		def new__init__(self, original_method_from_original_class, *args, **kwargs):
			self.__dict__[variable_name] = new_variable
			original_method_from_original_class(self, *args, **kwargs)

		original_method_from_original_class = original_class.__dict__['__init__']
		new_method_for_original_class = lambda self_, *args, **kwargs: new__init__(self_, original_method_from_original_class, *args, **kwargs)

		# Include the new variable
		setattr(original_class, '__init__', new_method_for_original_class)
		original_class.__dict__['__init__'].__doc__ = original_method_from_original_class.__doc__

	else:
		raise ValueError(" ❌ ERROR: type_ == '%s' is not defined!" %(type_))

######################## _adjustNameForPrivate #################################

def _adjustNameForPrivate(method_name, class_name):
	"""
	Adjust name for private member.
	"""
	if method_name.startswith('__'):
		assert class_name not in method_name
		new_method_name = '_%s%s' %(class_name, method_name)
		utils.printDebug(" ❗ Changing the name of the private method '%s' to '%s' (as Python normally does)..." %(method_name, new_method_name))
	else:
		new_method_name = method_name

	return new_method_name

######################## _appendTextToDocstring ################################

def _appendTextToDocstring(original_object, new_text = ""):
	"""
	Append text to docstring.
	"""
	previous_doc = original_object.__doc__

	separator = " ------------------------------------------------------------------------------ \n"
	if type(previous_doc).__name__ == 'NoneType':
		new_doc = "%s" %(separator)

	elif previous_doc.endswith(separator):
		new_doc = "%s" %(previous_doc)

	else:
		new_doc = "%s" %(previous_doc)
		new_doc += "\n%s" %(separator)

	new_doc += "%s" %(new_text)
	new_doc += "\n%s" %(separator)

	original_object.__doc__ = new_doc


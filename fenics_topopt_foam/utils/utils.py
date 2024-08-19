################################################################################
#                                    utils                                     #
################################################################################
# Some utilities.

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

# Functools
import functools

# NumPy
import numpy as np

# io
import io

# os
import os
import os.path as os_path

# sys
import sys

# High-level file operations
import shutil

# subprocess
import subprocess

# time
import time

# For regular expressions
import re

# Default dictionary -> Dictionary initialized with a given type of content
from collections import defaultdict

############################# Project libraries ################################

# FEniCS TopOpt Foam folder
from .. import __fenics_topopt_foam_folder__

################################ Small check ###################################

if sys.version_info[0] < 3:
	raise ValueError(" ❌ ERROR: FEniCS TopOpt Foam requires (at least) Python 3")

################################ setPrintLevel #################################

# The code is started with the prints in debug mode, because it is normally
 # interesting to know everything (or almost everything) that the code is doing.
global print_level
print_level = 'debug'

def setPrintLevel(new_print_level):
	"""
	Set the desired print level for FEniCS TopOpt Foam.
	"""

	global print_level
	if new_print_level in ['debug', 'simple']:
		print_level = new_print_level
	else:
		raise ValueError(" ❌ ERROR: new_print_level == '%s' is not defined!" %(new_print_level))

################################# printDebug ###################################

def printDebug(*args, **kwargs):
	"""
	Print only if print_level == 'debug'.
	"""

	if print_level == 'debug':
		customPrint(*args, **kwargs)
	else:
		pass

################################ customPrint ###################################

def customPrint(*args, **kwargs):
	"""
	Print function.
	In parallel, only the first processors prints.
	No synchronization is imposed here!
	"""

	from ..utils import utils_fenics_mpi
	rank = utils_fenics_mpi.rank

	# Special keyword arguments
	mpi_wait_for_everyone = kwargs.pop('mpi_wait_for_everyone', False)
	mpi_only_first_processor_prints = kwargs.pop('mpi_only_first_processor_prints', True)
	mpi_print_beforehand = kwargs.pop('mpi_print_beforehand', False)
	mpi_wait_for_print = kwargs.pop('mpi_wait_for_print', False)

	mpi_single_processor = kwargs.pop('mpi_single_processor', False)
	if mpi_single_processor == True:
		mpi_only_first_processor_prints = True
		mpi_wait_for_everyone = False
		mpi_broadcast_result = False

	def _print_simple():
		if mpi_print_beforehand == True:
			print(content)
		else:
			print(*args, **kwargs)

	def _print_parallel():
		if mpi_print_beforehand == True:
			print("[%d] " %(rank), content)
		else:
			# All processors print
			print("[%d] " %(rank), *args, **kwargs)

	if utils_fenics_mpi.runningInParallel():

		# Print content to string
		if mpi_print_beforehand == True:
			content = print_to_string(*args, **kwargs)

		# Wait for everyone!
		if mpi_wait_for_everyone == True:
			utils_fenics_mpi.waitProcessors()

		# Print
		if mpi_only_first_processor_prints == True:

			# Only first processor prints
			if utils_fenics_mpi.runningInFirstProcessor():
				if mpi_wait_for_everyone == True:
					with utils_fenics_mpi.first_processor_lock():
						_print_simple()
				else:
					_print_simple()
			else:
				pass

		else:
			if mpi_wait_for_everyone == True:
				with utils_fenics_mpi.first_processor_lock():
					_print_parallel()
			else:
				_print_parallel()

		# Wait for everyone!
		if mpi_wait_for_everyone == True:
			utils_fenics_mpi.waitProcessors()

	else:
		print(*args, **kwargs)

	# Flush to stdout 
	 # https://stackoverflow.com/questions/43195996/how-to-get-mpi4py-processes-to-finish-printing-before-executing-time-sleep
	if utils_fenics_mpi.runningInParallel():
		sys.stdout.flush()

	if mpi_wait_for_print == True:

		time_to_wait = 0.1 # s

		import time
		time.sleep(time_to_wait)

		# Wait for everyone!
		if mpi_wait_for_everyone == True:
			utils_fenics_mpi.waitProcessors()

############################## print_to_string #################################

def print_to_string(*args, **kwargs):
	"""
	Print to string.
	"""

	# https://stackoverflow.com/questions/39823303/python3-print-to-string?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	output = io.StringIO()
	print(*args, file=output, **kwargs)
	contents = output.getvalue()
	output.close()
	return contents

##################### only_the_first_processor_executes #########################

def only_the_first_processor_executes(broadcast_result = False, wait_for_everyone = True):
	"""
	Decorator for only the first processor to execute a function.

	*  Try not to use the 'wait_for_everyone' parameter in the decorator call. 

	** Using the keyword argument 'mpi_wait_for_everyone' when calling the 
	   decorated function is a nice alternative to 'wait_for_everyone',
	   because it does not leave weird*** MPI dependency relations.

	*** i.e., "hard to find", in the case of a bug.
 
	**** This construction is used just to make it possible to write less code.

	"""

	mpi_wait_for_everyone = wait_for_everyone
	mpi_broadcast_result = broadcast_result

	def decorator(original_function):
		@functools.wraps(original_function) # For using the correct 'docstring' in the wrapped function
		def wrapper(*args, **kwargs):

			# Check additional inputs
			wait_for_everyone = kwargs.pop('mpi_wait_for_everyone', mpi_wait_for_everyone)
			broadcast_result = kwargs.pop('mpi_broadcast_result', mpi_broadcast_result)
			single_processor = kwargs.pop('mpi_single_processor', False)
			if single_processor == True:
				wait_for_everyone = False
				broadcast_result = False

			# FEniCS+MPI utilities
			from ..utils import utils_fenics_mpi

			if wait_for_everyone == True:
				utils_fenics_mpi.assertFirstProcessorUnlocked()

			# Compute the function
			if utils_fenics_mpi.runningInParallel():

				# Wait for everyone!
				if wait_for_everyone == True:
					utils_fenics_mpi.waitProcessors()

				# First processor executes!
				if utils_fenics_mpi.runningInFirstProcessor():
					with utils_fenics_mpi.first_processor_lock(mpi_single_processor = single_processor):
						res = original_function(*args, **kwargs)
				else:
					res = None

				# Wait for everyone!
				if wait_for_everyone == True:
					utils_fenics_mpi.waitProcessors()

				# Broadcast result to everyone!
				if broadcast_result == True:

					assert wait_for_everyone == True # Just to be sure nothing will go badly wrong in here...

					#print("[%d] BROADCAST type(res).__name__ = " %(utils_fenics_mpi.rank), type(res).__name__)

					#res = utils_fenics_mpi.evaluateBetweenProcessors(res, operation = 'broadcast', proc_destination = 'all')
					res = utils_fenics_mpi.broadcastToAll(res)

					# Wait for everyone!
					if wait_for_everyone == True:
						utils_fenics_mpi.waitProcessors()

			else:
				res = original_function(*args, **kwargs)

			import inspect
			num_lock_traceback = 4
			stack = inspect.stack()[1:] # retira o FrameInfo corrente
			stack_len = min(len(stack), num_lock_traceback)
			lock_traceback = ' < '.join([str(stack[i].function) for i in range(stack_len)])
			#stack_len = max(min(len(inspect.stack()) - 1, num_lock_traceback), 1)
			#lock_traceback = ' < '.join([str(inspect.stack()[2 + i].function) for i in range(stack_len)])
			#print("[%d] FINAL lock_traceback = " %(utils_fenics_mpi.rank), lock_traceback)
			#print("[%d] FINAL type(res).__name__ = " %(utils_fenics_mpi.rank), type(res).__name__)
			return res

		return wrapper
	return decorator

############################# run_command_in_shell #############################

def run_command_in_shell(string_command, mode = 'save output to variable', include_delimiters_for_print_to_terminal = True, indent_print = False, suppress_run_print = False, accept_empty_response = False):
	"""
	Run a command using Shell.
	* "subprocess.Popen" keeps the same environment variables (of the export type)
	  that are in the Shell of the terminal that called this Python code.
	  Therefore, there is no need to reinitialize the environment variables
	  of OpenFOAM.

	* The "executable" ('/bin/bash') is specified in "subprocess.Popen" call, 
	  otherwise, somehow, the following was happening:

		Fedora => /bin/bash
		Ubuntu => /bin/dash

		* To check in terminal:
			$ ls -l /bin/sh

		* To check in Python:
			$ python
			>>> import os
			>>> os.system("ls -l /bin/sh")

	   -> Without specifying "executable" as '/bin/bash', exported functions 
	      (such as "export -f foamVersion") would not work in subshells in 
	      Ubuntu. Just don't ask me why  /bin/dash is incapable of doing it...
			$ export -f foamVersion
			$ python
			>>> import os
			>>> os.system("foamVersion")

		https://www.saltycrane.com/blog/2011/04/how-use-bash-shell-python-subprocess-instead-binsh/
		https://askubuntu.com/questions/976485/what-is-the-point-of-sh-being-linked-to-dash
		https://wiki.ubuntu.com/DashAsBinSh
		https://www.difference.wiki/bash-vs-dash/
		https://anandmpandit.blogspot.com/2011/09/how-to-change-default-shell-from-dash.html

	"""

	# Force to use locale in English (* Such as en_US.utf8)
	 # * Because it's cumbersome to implement output checks in more than one language.
	 # https://askubuntu.com/questions/264283/switch-command-output-language-from-native-language-to-english
	 # https://askubuntu.com/questions/264283/switch-command-output-language-from-native-language-to-english/264709#264709
	if '\n' in string_command:
		string_command = "export LC_ALL=C\n%s" %(string_command)
	else:
		string_command = "export LC_ALL=C; %s" %(string_command)

	if indent_print == True:
		start_of_print = " ╰"
	else:
		start_of_print = ""

	# Run the command
	if mode == 'no prints':
		p = subprocess.Popen(string_command, shell = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL, executable = '/bin/bash')
		suppress_run_print = True
		accept_empty_response = True

	elif mode == 'save output to variable':
		if suppress_run_print == False:
			printDebug("%s 💻 Running '%s'..." %(start_of_print, string_command))
		p = subprocess.Popen(string_command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE, executable = '/bin/bash')

	elif mode == 'save all prints to variable':
		if suppress_run_print == False:
			printDebug("%s 💻 Running '%s'..." %(start_of_print, string_command))
		p = subprocess.Popen(string_command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, executable = '/bin/bash')

	elif mode == 'print directly to terminal':

		if suppress_run_print == False:
			run_print = "%s 💻 Running '%s'..." %(start_of_print, string_command)
		else:
			run_print = ""

		if include_delimiters_for_print_to_terminal == True:
			customPrint("""%s
 ▼ ────────────────────────────────────────────────────────────────────────── ▼
""" %(run_print))
		else:
			if suppress_run_print == False:
				customPrint("%s" %(run_print))

		p = subprocess.Popen(string_command, shell = True, executable = '/bin/bash')

	elif mode == 'print directly to terminal and save all prints to variable':

		if suppress_run_print == False:
			run_print = "%s 💻 Running '%s'..." %(start_of_print, string_command)
		else:
			run_print = ""

		if include_delimiters_for_print_to_terminal == True:
			customPrint("""%s
 ▼ ────────────────────────────────────────────────────────────────────────── ▼
""" %(run_print))
		else:
			if suppress_run_print == False:
				customPrint("%s" %(run_print))

	else:
		raise ValueError(" ❌ ERROR: mode == '%s' is not defined!" %(mode))

	# Check the message of the subprocess
	if mode == 'print directly to terminal and save all prints to variable':
		# https://stackoverflow.com/questions/25750468/displaying-subprocess-output-to-stdout-and-redirecting-it
		# https://stackoverflow.com/questions/29546311/popen-communicate-throws-unicodedecodeerror
		with subprocess.Popen(string_command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, bufsize = 1, universal_newlines = True, encoding = "utf-8", executable = '/bin/bash') as p, io.StringIO() as buf:

			for line in p.stdout:
				customPrint(line, end = '')
				buf.write(line)

			stdoutdata = buf.getvalue()

		stderrdata = None
	else:
		(stdoutdata, stderrdata) = p.communicate()

	#rc = p.returncode

	if mode in ['print directly to terminal', 'print directly to terminal and save all prints to variable']:
		if include_delimiters_for_print_to_terminal == True:
			customPrint("""
 ▲ ────────────────────────────────────────────────────────────────────────── ▲
""")

	if type(stdoutdata).__name__ != 'NoneType':
		if mode == 'print directly to terminal and save all prints to variable':
			stdout_response = stdoutdata
		else:
			# Message to load (* Needs to be decoded because from the "byte literal" format to the String format)
			stdout_response = stdoutdata.decode()
	else:
		stdout_response = None

	# Check error messages
	if type(stderrdata).__name__ != 'NoneType':
		if mode == 'print directly to terminal and save all prints to variable':
			stderr_response = stderrdata
		else:
			# Error message (* Needs to be decoded because from the "byte literal" format to the String format)
			stderr_response = stderrdata.decode()
	else:
		stderr_response = None

	# Check for error
	def check_for_words(text, list_words):
		for i in range(len(list_words)):
			if (
				(type(list_words[i]).__name__ != 'NoneType')
				and (list_words[i] != '')
				and ("command not found" in list_words[i])
				and ("No such file or directory" in list_words[i])
				):

				return True
		return False

	list_error_words = ['command not found', 'No such file or directory']
	if check_for_words(stdout_response, list_error_words) and check_for_words(stderrdata, list_error_words):
		raise ValueError(""" ❌ ERROR: Command not found in Shell environment. 

stdout:
 ------------------------------------------------------------------------------

%s
 ------------------------------------------------------------------------------

stderr:
 ------------------------------------------------------------------------------

%s
 ------------------------------------------------------------------------------

This could mean that some function has not been set beforehand as an export 
function in Shell, which blocks Python to recognize the variable with Popen:

For example, if the problem is a function, running the following
line should fix it (for example, for the function foamVersion):
$ export -f foamVersion

* If the problem is a Shell variable, then it should be fixed by something like 
(for example, for the variable SOME_VARIABLE):
$ export SOME_VARIABLE

""" %(stdout_response, stderr_response))

	# Prepare final response
	if mode in ['save output to variable', 'print directly to terminal and save all prints to variable']:
		response = stdout_response

		if response == "" and accept_empty_response == False:
			raise ValueError(""" ❌ ERROR: No stdout response?
stdout:

%s

 ------------------------------------------------------------------------------

stderr:

%s

 ------------------------------------------------------------------------------
""" %(stdout_response, stderr_response))

	elif mode == 'save all prints to variable':
		response = stdout_response
	elif mode == 'print directly to terminal' or mode == 'no prints':
		response = None
	else:
		raise ValueError(" ❌ ERROR: mode == '%s' is not defined!" %(mode))

	return response

############################ removeTrailingSlash ###############################

def removeTrailingSlash(file_path):
	"""
	Removes a trailing forward slash ("/") if there is one in the end of the file_path.
	"""
	if file_path.endswith('/'):
		file_path = file_path[:len(file_path)-1]
	return file_path

###################### removeRepeatedElementsFromList ##########################

def removeRepeatedElementsFromList(list_orig, keep_order = True):
	"""
	Remove repeated elements from list.
	"""
	if keep_order == True:
		new_list = sorted(set(list_orig), key = list_orig.index)
	else:
		new_list = list(set(list_orig))

	return new_list

############################### loadPythonFile #################################

def loadPythonFile(file_location, force_reload = True):
	"""
	Load Python file.
	https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
	https://stackoverflow.com/questions/31410419/python-reload-file
	"""

	if force_reload == True:
		# Update file modification time to current time, to guarantee that the file will be reloaded.
		 # * Remember that the Python interpreter reloads modules ONLY WHEN the modification time of the .py file is more recent than the time of the creation of the .pyc files.
		 # The command below is the same as the Shell command "touch $file_location"
		 # run_command_in_shell("touch '%s'" %(file_location), mode = 'print directly to terminal')
		os.utime(file_location, None)

	split_file_location = file_location.split('/')
	module_name = split_file_location[len(split_file_location) - 1].split('.py')[0]

	from importlib.machinery import SourceFileLoader
	loaded_module = SourceFileLoader(module_name, file_location).load_module()

	return loaded_module

######################## checkInstalledOpenFOAMUtilities ######################

@only_the_first_processor_executes()
def checkCustomOpenFOAMUtilities():
	"""
	Check if all custom OpenFOAM utilities are installed.
	"""

	# Get all module names in folder
	module_paths_in_folder = ["%s/cpp_openfoam_utilities/%s" %(__fenics_topopt_foam_folder__, name) for name in os.listdir("%s/cpp_openfoam_utilities" %(__fenics_topopt_foam_folder__)) if os.path.isdir("%s/cpp_openfoam_utilities/%s" %(__fenics_topopt_foam_folder__, name))]

	# Check if the modules are in the OpenFOAM environment
	for module_path in module_paths_in_folder:
		module_path_split = module_path.split('/')
		module_name = module_path_split[len(module_path_split) - 1]

		compileModuleIfNeeded(module_path, force_recompile = False, mpi_broadcast_result = False, mpi_wait_for_everyone = False)

		#customPrint(" 🌀 Checking if module '%s' is already compiled in the OpenFOAM environment ..." %(module_name))
		#bash_response = run_command_in_shell('echo -n $(type %s)' %(module_name), mode = 'save all prints to variable', indent_print = True)

		#if "not found" in bash_response:
#			raise ValueError(""" ❌ ERROR: Module \"%s\" (\"%s\") not found: %s
#Please install all necessary utilities in fenics_topopt_foam/cpp_openfoam_utilities. Check
#the script fenics_topopt_foam/cpp_openfoam_utilities/compile_utilities_for_openfoam.sh
#for the instructions of what and how to install.
#""" %(module_name, module_path, bash_response))

############################ compileModuleIfNeeded #############################

@only_the_first_processor_executes()
def compileModuleIfNeeded(module_path, force_recompile = False, is_test_module = False):
	"""
	Compile module in folder for OpenFOAM.
	Folder structure:
	   📂 moduleFirst
	   ├ 📂 Make
	   ├ 📃 moduleFirst.C
	   ╰ 📃 readme

	"""

	# Check if the module is in the OpenFOAM environment
	module_path_split = module_path.split('/')
	module_name = module_path_split[len(module_path_split) - 1]

	if module_name.startswith('__'):
		return None # Sorry, we are ignoring folders starting with '__', such as __pycache__

	if is_test_module == True:
		module_name = "Test-%s" %(module_name) # All test modules begin with "Test-"

	customPrint(" 🌀 Checking if module '%s' is already compiled in the OpenFOAM environment ..." %(module_name))
	bash_response = run_command_in_shell('echo -n $(type %s)' %(module_name), mode = 'save all prints to variable', indent_print = True)

	if force_recompile == True:
		flag_compile_module = True
	else:
		if "not found" in bash_response:
			customPrint(" 📌️ Module \"%s\" (\"%s\") not found: %s" %(module_name, module_path, bash_response))
			flag_compile_module  = True
		else:
			if is_test_module == True:
				flag_compile_module = False
			else:

				# Remove failed folders in Make
				folders_in_make = getSubfolderNames("%s/Make" %(module_path), mpi_broadcast_result = False, mpi_wait_for_everyone = False)
				for folder_in_make in folders_in_make:
					if folder_in_make.startswith("FAIL_"):
						removeFolderIfItExists("%s/Make/%s" %(module_path, folder_in_make), mpi_broadcast_result = False, mpi_wait_for_everyone = False)

				# Check last modified timestamp for the source files
				latest_timestamp_in_files = verifyLatestTimeInFolder(module_path, ignore_folders = ['Make'], return_timestamp = True)
				latest_timestamp_in_Make = verifyLatestTimeInFolder("%s/Make" %(module_path), ignore_folders = 'all subfolders', return_timestamp = True)
				latest_timestamp = max([latest_timestamp_in_files, latest_timestamp_in_Make])

				# Check last modified timestamp for the compiled files
				latest_timestamp_in_compiled_files = verifyLatestTimeInFolder("%s/Make" %(module_path), ignore_folders = 'base folder', return_timestamp = True)

				# Looks like you have modified the source files since last compilation
				if type(latest_timestamp_in_compiled_files).__name__ == 'NoneType':
					customPrint(""" 📌️ Module \"%s\" (\"%s\") found but not compiled:
 ╰ 💻: %s""" %(module_name, module_path, bash_response))
					flag_compile_module  = True
				elif latest_timestamp > latest_timestamp_in_compiled_files:
					customPrint(""" 📌️ Module \"%s\" (\"%s\") found but with older timestamp (%s < %s):
 ╰ 💻: %s""" %(module_name, module_path, convertTimeStampToTime(latest_timestamp_in_compiled_files), convertTimeStampToTime(latest_timestamp), bash_response))
					flag_compile_module  = True
				else:
					customPrint(""" 📌️ Module \"%s\" (\"%s\") found:
 ╰ 💻: %s""" %(module_name, module_path, bash_response))
					flag_compile_module  = False

	if flag_compile_module == False:
		pass
	else:

		customPrint("""
────────────────────────────────────────────────────────────────────────────────
	           📌️ Compilation of module for OpenFOAM 📌️
 ▼ -------------------------------------------------------------------------- ▼ 

""")

		customPrint(" 📌️ Cleaning up previous compilation of \"%s\"..." %(module_name))
		run_command_in_shell("wclean '%s'" %(module_path), mode = 'print directly to terminal', indent_print = True)

		customPrint(" 📌️ Compiling module \"%s\"..." %(module_name))
		compilation_text = run_command_in_shell("wmake '%s'" %(module_path), mode = 'print directly to terminal and save all prints to variable', indent_print = True)
			#run_command_in_shell("wmake '%s'" %(module_path), mode = 'print directly to terminal', indent_print = True)
			#compilation_text = run_command_in_shell("wmake '%s'" %(module_path), mode = 'save all prints to variable', indent_print = True)

#		customPrint(""" ▼ ────────────────────────────────────────────────────────────────────────── ▼
#""")
#		customPrint(compilation_text, end = "")
#		customPrint("""
# ▲ ────────────────────────────────────────────────────────────────────────── ▲
#""")


		customPrint("""
 ▲ -------------------------------------------------------------------------- ▲ 

""")

		customPrint("""
────────────────────────────────────────────────────────────────────────────────
""")

		customPrint(" 🌀 Checking if module '%s' is successfully compiled in the OpenFOAM environment ..." %(module_name))

		# Check the compilation text
		if 'Error' in compilation_text:
			made_folder = getSubfolderNames("%s/Make" %(module_path), mpi_broadcast_result = False, mpi_wait_for_everyone = False)[0]
			current_time = getCurrentFormattedTime(format_type = 'for file name')
			failed_compilation_folder = "%s/Make/FAIL_%s_%s" %(module_path, current_time, made_folder)
			os.rename("%s/Make/%s" %(module_path, made_folder), failed_compilation_folder)
			raise ValueError(" ❌ ERROR: Compilation failed! Failed compilation is moved to \"%s\"." %(failed_compilation_folder))

		# Check if it is available
		bash_response = run_command_in_shell('echo -n $(type %s)' %(module_name), mode = 'save all prints to variable', indent_print = True)
		if "not found" in bash_response:
			raise ValueError(""" ❌ ERROR: Module \"%s\" (\"%s\") not found: %s
This means that the compilation failed!
""" %(module_name, module_path, bash_response))

		# Check last modified timestamp for the compiled files
		if is_test_module == False:
			if 'latest_timestamp_in_compiled_files' in locals():
				latest_timestamp_in_compiled_files_new = verifyLatestTimeInFolder("%s/Make" %(module_path), ignore_folders = 'base folder', return_timestamp = True)
				if (type(latest_timestamp_in_compiled_files).__name__ != 'NoneType') and (latest_timestamp_in_compiled_files > latest_timestamp_in_compiled_files_new):
					raise ValueError(""" ❌ ERROR: Timestamp not updated!
	This means that the compilation failed!
""")

	return module_name

############################ compileLibraryIfNeeded ############################

@only_the_first_processor_executes(broadcast_result = True)
def compileLibraryIfNeeded(lib_src_path, force_recompile = False):
	"""
	Compile library for using in OpenFOAM.

	Source folder structure:
	   📂 FirstLib
	   ├ 📂 Make
	   ├ 📃 FirstLib.C
	   ╰ 📃 readme

	Folder that OpenFOAM uses to locate custom libraries:
	   📂 $FOAM_USER_LIBBIN
	   ╰ 📃 libFirstLib.so

	"""

	# Location of the user library files: $FOAM_USER_LIBBIN
	foam_user_libbin = run_command_in_shell("echo -n $FOAM_USER_LIBBIN", mode = 'save all prints to variable', indent_print = True)

	# Check the name of the library from Make/files
	lib_path_in_src_file = run_command_in_shell("echo -n $(grep 'LIB = ' %s/Make/files)" %(lib_src_path), mode = 'save all prints to variable', indent_print = True).split(' = ')[1]
	lib_path_in_src_file_split = lib_path_in_src_file.split('/')
	lib_name = lib_path_in_src_file_split[1] + '.so'
	libbin_location = lib_path_in_src_file_split[0]

	if (len(lib_path_in_src_file_split) > 2) or (libbin_location not in ['$(FOAM_USER_LIBBIN)', '$FOAM_USER_LIBBIN']):
		raise ValueError(" ❌ ERROR: Library '%s' is located in '%s' and not in '$(FOAM_USER_LIBBIN) (%s)'" %(lib_name, libbin_location, foam_user_libbin))

	if lib_name.startswith('__'):
		return None # Sorry, we are ignoring folders starting with '__', such as __pycache__

	# Library path
	lib_path = '%s/%s' %(foam_user_libbin, lib_name)

	# Check if the library is in the OpenFOAM environment
	customPrint(" 🌀 Checking if the library '%s' is available in OpenFOAM in '%s'..." %(lib_name, lib_path))
	bash_response = run_command_in_shell("""echo -n $(
if test -f "%s"; then
	echo "%s is available"
else 
	echo "%s not found"
fi
)""" %(lib_path, lib_path, lib_path), mode = 'save all prints to variable', indent_print = True)

	if force_recompile == True:
		flag_compile_lib = True
	else:
		if "not found" in bash_response:
			customPrint(" 📌️ Library \"%s\" (\"%s\") not found: %s" %(lib_name, lib_path, bash_response))
			flag_compile_lib  = True
		else:

			# Check last modified timestamp for the source files
			latest_timestamp_in_files = verifyLatestTimeInFolder(lib_src_path, ignore_folders = ['Make'], return_timestamp = True)
			latest_timestamp_in_Make = verifyLatestTimeInFolder("%s/Make" %(lib_src_path), ignore_folders = 'all subfolders', return_timestamp = True)
			latest_timestamp = max([latest_timestamp_in_files, latest_timestamp_in_Make])

			# Check last modified timestamp for the .so file
			latest_timestamp_in_compiled_so_file = verifyLatestTimeOfFile("%s" %(lib_path), return_timestamp = True)

			# Looks like you have modified the source files since last compilation
			if latest_timestamp > latest_timestamp_in_compiled_so_file:
				customPrint(""" 📌️ Library \"%s\" (\"%s\") found but with older timestamp (%s < %s):
 ╰ 💻: %s""" %(lib_name, lib_path, convertTimeStampToTime(latest_timestamp_in_compiled_so_file), convertTimeStampToTime(latest_timestamp), bash_response))
				flag_compile_lib  = True
			else:
				customPrint(""" 📌️ Library \"%s\" (\"%s\") found:
 ╰ 💻: %s""" %(lib_name, lib_path, bash_response))
				flag_compile_lib  = False

	if flag_compile_lib == False:
		pass
	else:

		customPrint("""
────────────────────────────────────────────────────────────────────────────────
	           📌️ Compilation of library for OpenFOAM 📌️
 ▼ -------------------------------------------------------------------------- ▼ 

""")

		customPrint(" 📌️ Cleaning up previous compilation of \"%s\"..." %(lib_name))
		run_command_in_shell("wclean '%s'" %(lib_src_path), mode = 'print directly to terminal', indent_print = True)

		customPrint(" 📌️ Compiling library \"%s\" (generating .so)..." %(lib_name))
		compilation_text = run_command_in_shell("wmake libso '%s'" %(lib_src_path), mode = 'print directly to terminal and save all prints to variable', indent_print = True)

#		customPrint(""" ▼ ────────────────────────────────────────────────────────────────────────── ▼
#""")
#		customPrint(compilation_text, end = "")
#		customPrint("""
# ▲ ────────────────────────────────────────────────────────────────────────── ▲
#""")


		customPrint("""
 ▲ -------------------------------------------------------------------------- ▲ 

""")

		customPrint("""
────────────────────────────────────────────────────────────────────────────────
""")

		customPrint(" 🌀 Checking if library '%s' is successfully compiled in the OpenFOAM environment ..." %(lib_name))

		# Check the compilation text
		if 'Error' in compilation_text:
			removeFileIfItExists('%s' %(lib_path))
			raise ValueError(" ❌ ERROR: Compilation failed! Failed compilation removed from '%s'!" %(lib_path))

		# Check if it is available
		bash_response = run_command_in_shell("""echo -n $(
if test -f "%s"; then
	echo "%s is available"
else 
	echo "%s not found"
fi
)""" %(lib_path, lib_path, lib_path), mode = 'save all prints to variable', indent_print = True)
		if "not found" in bash_response:
			raise ValueError(""" ❌ ERROR: Library \"%s\" (\"%s\") not found: %s
This means that the compilation failed!
""" %(lib_name, lib_path, bash_response))

		# Check last modified timestamp for the compiled files
		if 'latest_timestamp_in_compiled_so_file' in locals():
			latest_timestamp_in_compiled_so_file_new = verifyLatestTimeOfFile("%s" %(lib_path), return_timestamp = True)
			if latest_timestamp_in_compiled_so_file > latest_timestamp_in_compiled_so_file_new:
				raise ValueError(""" ❌ ERROR: Timestamp not updated!
This means that the compilation failed!
""")

	return lib_name

########################## getCurrentFormattedTime #############################

def getCurrentFormattedTime(format_type = 'simple', time_selection = 'local'):
	"""
	Returns the current formatted time.
	https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
	"""

	if time_selection == 'local':
		time_current = time.localtime()
	elif time_selection == 'GMT':
		time_current = time.gmtime()
	else:
		raise ValueError(" ❌ ERROR: time_selection == '%s' is not defined!" %(time_selection))

	if format_type == 'simple':
		return time.strftime("%Y-%m-%d %H:%M:%S", time_current)
	elif format_type == 'for file name':
		return time.strftime("%Y_%m_%d_-_%H_%M_%S", time_current)
	else:
		raise ValueError(" ❌ ERROR: format_type == '%s' is not defined!" %(format_type))

####################### compileLibraryFoldersIfNeeded ###########################

@only_the_first_processor_executes(broadcast_result = True)
def compileLibraryFoldersIfNeeded(folder_with_lib_src, force_recompile = False):
	"""
	Compile libraries in folder for OpenFOAM.

	Source folder structure:
	  📂 [folder_with_libraries] 
	   │
	   ├ 📂 FirstLib
	   │ ├ 📂 Make
	   │ ├ 📃 FirstLib.C
	   │ ╰ 📃 readme
	   │
	   ╰ 📂 SecondLib
	     ├ 📂 Make
	     ├ 📃 SecondLib.C
	     ╰ 📃 readme

	Folder that OpenFOAM uses to locate custom libraries:
	   📂 $FOAM_USER_LIBBIN
	   ├ 📃 libFirstLib.so
	   ╰ 📃 libSecondLib.so

	"""

	customPrint("📌️ Checking if there is any library in '%s' needing to be compiled..." %(folder_with_lib_src))

	# Get all library names in folder
	lib_src_paths_in_folder = getSubfolderNames(folder_with_lib_src, mpi_broadcast_result = False, mpi_wait_for_everyone = False)

	# Cross-check nearby dependencies of the module
	lib_src_paths_in_folder = crossCheckNearbyDependencies(folder_with_lib_src, lib_src_paths_in_folder)

	lib_names = []
	for lib_src_path in lib_src_paths_in_folder:
		lib_names += [compileLibraryIfNeeded("%s/%s" %(folder_with_lib_src, lib_src_path), force_recompile = force_recompile, mpi_broadcast_result = False, mpi_wait_for_everyone = False)]

	return lib_names

####################### compileModuleFoldersIfNeeded ###########################

@only_the_first_processor_executes()
def compileModuleFoldersIfNeeded(folder_with_modules, force_recompile = False):
	"""
	Compile modules in folder for OpenFOAM.
	Folder structure:
	  📂 [folder_with_modules] 
	   │
	   ├ 📂 moduleFirst
	   │ ├ 📂 Make
	   │ ├ 📃 moduleFirst.C
	   │ ╰ 📃 readme
	   │
	   ╰ 📂 moduleSecond
	     ├ 📂 Make
	     ├ 📃 moduleSecond.C
	     ╰ 📃 readme
	"""

	customPrint("📌️ Checking if there is any module in '%s' needing to be compiled..." %(folder_with_modules))

	# Get all module names in folder
	module_paths_in_folder = getSubfolderNames(folder_with_modules, mpi_broadcast_result = False, mpi_wait_for_everyone = False)

	# Cross-check nearby dependencies of the module
	module_paths_in_folder = crossCheckNearbyDependencies(folder_with_modules, module_paths_in_folder)

	# For all modules in folder
	for module_path in module_paths_in_folder:
		compileModuleIfNeeded("%s/%s" %(folder_with_modules, module_path), force_recompile = force_recompile, mpi_broadcast_result = False, mpi_wait_for_everyone = False)

######################## crossCheckNearbyDependencies ##########################

def crossCheckNearbyDependencies(folder_with_modules, module_names):
	"""
	Cross-check nearby dependencies of the module.
	* Nearby = Dependencies inside the "folder_with_modules".

	This functions reorders the modules to compile in the right order:
	From the dependencies (hierarchically) until the current module.

	Is this really needed?

		No, because OpenFOAM's "wmake" function is supposed to compile
		in the right order.

	Why include this function? Does it fail without using this function?

		It may fail when the dependent module has already been compiled
		in another computer, because OpenFOAM creates an "lnInclude" folder
		with links, which are broken when someone copies the source code
		to another computer.

		When the "lnInclude" folder exists but is empty or all
		files it contains are broken links, the compilation fails.

		-> Example of folder from a module used by another module (moduleFirst, for example):
		    📂 moduleSecond
		     ├ 📂 lnInclude
		     │ ╰ [empty or with broken links]
		     ├ 📂 Make
		     ├ 📃 moduleSecond.C
		     ╰ 📃 readme

		How to solve this issue?

			1) [simplest way, but must remember to do so] Delete the 
				"lnInclude" folder before copying the source code.

			2) [not so simple, but "works" and may possibly cover more 
				cases (?)] Adequately order the modules that are to be compiled
				[which is performed by the "crossCheckNearbyDependencies" function].

			3) [not so simple, but maybe too specific (?)]
				Check all "lnInclude" folders, if they are not empty
				or if they include broken links.
				-> In such case, remove the "lnInclude" folder.

	"""

	# Dependency names
	dependency_inclusion_names = {module_name : '-I../%s/lnInclude' %(module_name) for module_name in module_names}

	# Create a dependency cabinet (figuratively, of course. It is just a dictionary with "list" contents)
	dependency_cabinet = {}

	# For all module names
	for module_name in module_names:

		# Create a drawer of the dependency cabinet
		dependency_cabinet[module_name] = []

		# Check the 'options' file
		with open("%s/%s/Make/options" % (folder_with_modules, module_name), "r", encoding = 'utf-8') as text_file:

			# Read the first line
			line = text_file.readline()

			# While there is a line that has been read
			while line:

				# If there is a local include in the line
				if '-I../' in line:

					# Check for the dependent inclusions
					for dependency_module_name in dependency_inclusion_names:
						dependency_inclusion = dependency_inclusion_names[dependency_module_name]
						if dependency_inclusion in line:

							# Include the content in the cabinet
							dependency_cabinet[module_name] += [dependency_module_name]

							break

				# Read the next line
				line = text_file.readline()

	max_recursion_depth = 10
	def getAllDependencies_recursive(module_name, __recursion_depth = 0):
		"""
		Get all dependencies of the module (recursively).
		"""

		# List of dependencies
		module_dependency_list = dependency_cabinet[module_name]

		# Maximum recursion depth
		if __recursion_depth > max_recursion_depth:
			raise ValueError(" ❌ ERROR: Maximum recursion depth %d reached!" %(__recursion_depth))

		# Get all subdependencies
		return_list = []
		for dependent_module_name in module_dependency_list:

			# Recursion to get the subdependencies
			return_list += getAllDependencies_recursive(dependent_module_name, __recursion_depth = __recursion_depth + 1) 

		# Include the current dependency
		return_list += [module_name]

		# Stopping criterion -> When "module_dependency_list == []"
		return return_list

	# Check all dependent modules and order them
	module_names_new = []
	for module_name in dependency_cabinet:

		# If not already included
		if module_name not in module_names_new:

			# Get all dependencies of the module (recursively)
			all_dependencies = getAllDependencies_recursive(module_name)

			# Include the module names in the right order to compile
			for dependent_module_name in all_dependencies:

				# Only include if not already included
				if dependent_module_name not in module_names_new:
					module_names_new += [dependent_module_name]

	return module_names_new

############################ verifyLatestTimeOfFile ############################

def verifyLatestTimeOfFile(file_for_checking, return_timestamp = True):
	"""
	Verify the latest timestamp of a file.
	"""

	latest_timestamp = os.path.getmtime(file_for_checking)

	if return_timestamp == True:
		return latest_timestamp
	else:
		return convertTimeStampToTime(latest_timestamp)

############################ verifyLatestTimeInFolder ##########################

def verifyLatestTimeInFolder(folder_for_checking, ignore_folders = ['Make'], return_timestamp = True):
	"""
	Verify the latest timestamp of all files in the folder.
	"""

	def check_ignored_folder(currentfolder):
		"""
		Check if the folder is to be ignored.
		"""
		if ignore_folders == 'all subfolders':
			if currentfolder == folder_for_checking:
				return False
			else:
				return True

		elif ignore_folders == 'base folder':
			if currentfolder == folder_for_checking:
				return True
			else:
				return False
		else:
			for folder in ignore_folders:
				if folder in currentfolder:
					return True
			return False

	# Create a walk through the folders
	folder_walks = list(os.walk(folder_for_checking))

	# Create a walk through the files
	file_walk = []
	for folder_walk in folder_walks:

		currentfolder = folder_walk[0]
		subfolders = folder_walk[1]
		filenames = folder_walk[2]

		# Include the files in the walk if it is not inside an ignored folder
		if check_ignored_folder(currentfolder) == False:
			for filename in filenames:
				file_walk += ["%s/%s" %(currentfolder, filename)]

	if len(file_walk) == 0:
		return None
	else:

		def _getmtime(current_file):
			try: 
				return os.path.getmtime(current_file)
			except:
				customPrint(" ❗ '%s' is broken! Maybe it is a broken link due to copying from different computers..." %(current_file))
				return 0

		latest_timestamp = max([_getmtime(current_file) for current_file in file_walk])

		if return_timestamp == True:
			return latest_timestamp
		else:
			return convertTimeStampToTime(latest_timestamp)

############################# getSubfolderNames ################################

@only_the_first_processor_executes(broadcast_result = True)
def getSubfolderNames(folder_for_checking, startswith = ''):
	"""
	Get the names of the immediate subfolders.
	"""

	# Create a walk through the folders
	subfolder_names = next(os.walk('%s' %(folder_for_checking)))[1]

	if startswith != '':
		subfolder_names_new = []
		for subfolder_name in subfolder_names:
			if subfolder_name.startswith(startswith):
				subfolder_names_new += [subfolder_name]

		subfolder_names = subfolder_names_new

	return subfolder_names

######################### getNamesOfFilesInFolder ##############################

@only_the_first_processor_executes(broadcast_result = True)
def getNamesOfFilesInFolder(folder_for_checking, file_extensions_to_ignore = [], file_extensions_to_consider = []):
	"""
	Get the names of the immediate files.
	"""

	# Create a walk through the folders
	file_names = next(os.walk('%s' %(folder_for_checking)))[2]

	# Ignore some files
	if len(file_extensions_to_ignore) > 0 or len(file_extensions_to_consider) > 0:
		file_names_previous = file_names
		file_names = []
		for i in range(len(file_names_previous)):

			# File extensions to ignore
			to_ignore = False
			for file_extension_to_ignore in file_extensions_to_ignore:
				if file_names_previous[i].endswith(file_extension_to_ignore):
					to_ignore = True
					break

			if to_ignore == False:

				# File extensions to consider
				if len(file_extensions_to_consider) == 0:
					to_consider = True
				else:
					to_consider = False
					for file_extension_to_consider in file_extensions_to_consider:
						if file_names_previous[i].endswith(file_extension_to_consider):
							to_consider = True

				if to_consider == True:
					file_names += [file_names_previous[i]]

	return file_names

############################# convertTimeStampToTime ###########################

def convertTimeStampToTime(timestamp):
	"""
	Convert timestamp to a "human-friendly" representation.
	>>> time.ctime(1581790233.9666302)
	'Sat Feb 15 15:10:33 2020'
	"""
	return time.ctime(timestamp)

#################### getLastModificationTimeStampOfFile ########################

def getLastModificationTimeStampOfFile(file_path):
	"""
	Timestamp of last modification.
	It is the number of seconds since epoch (check out: https://en.wikipedia.org/wiki/Unix_time)
	https://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python
	"""
	return os.path.getmtime(file_path)

################################## moveFolder ##################################

@only_the_first_processor_executes()
def moveFolder(origin, destination, remove_if_it_already_exists = False):
	"""
	Move folder to another.
	"""

	# Check if the names are the same

	origin_last_folder_name = origin.split('/')[-1]
	destination_last_folder_name = destination.split('/')[-1]

	if origin_last_folder_name == destination_last_folder_name:
		if remove_if_it_already_exists == True:

			if os.path.isdir(destination):
				removeFolderIfItExists(destination, mpi_wait_for_everyone = False)
			else:
				removeFileIfItExists(destination, mpi_wait_for_everyone = False)

		else:
			raise ValueError(" ❌ ERROR: Folder '%s' exists!" %(destination))

	# If both are folders
	if os.path.isdir(origin) and os.path.isdir(destination):
		subfolders_destination = getSubfolderNames(origin, mpi_broadcast_result = False, mpi_wait_for_everyone = False)
		if origin_last_folder_name in subfolders_destination:
			if remove_if_it_already_exists == True:
				destination_subfolder = "%s/%s" %(destination, origin_last_folder_name)
				removeFolderIfItExists(destination_subfolder, mpi_wait_for_everyone = False)

			else:
				raise ValueError(" ❌ ERROR: Folder '%s' exists in destination ('%s')!" %(origin_last_folder_name, destination))

	if remove_if_it_already_exists == True:
		run_command_in_shell("mv --force '%s' '%s'" %(origin, destination), mode = 'print directly to terminal')
	else:
		run_command_in_shell("mv '%s' '%s'" %(origin, destination), mode = 'print directly to terminal')

################################## copyFolder ##################################

@only_the_first_processor_executes()
def copyFolder(origin, destination, overwrite = False, ommit_files_inside_folder = []):
	"""
	Copy folder to another.
	"""
	if overwrite == False and os.path.exists("%s" % (destination)):
		raise ValueError(" ❌ ERROR: Folder '%s' exists!" %(destination))

	if len(ommit_files_inside_folder) == 0:
		run_command_in_shell("/bin/cp -pr '%s' '%s'" %(origin, destination), mode = 'print directly to terminal')
	else:
		
		createFolderIfItDoesntExist(destination, mpi_wait_for_everyone = False)

		files_names = getFileNamesInFolder(folder_path, mpi_wait_for_everyone = False)
		for file_name in files_names:
			if file_name not in ommit_files_inside_folder:
				copyFile("%s/%s" %(origin, file_name), "%s/%s" %(destination, file_name), overwrite = overwrite, mpi_wait_for_everyone = False)

		subfolder_names = getFileNamesInFolder(folder_path, mpi_wait_for_everyone = False)
		for subfolder_name in subfolder_names:
			if subfolder_name not in ommit_files_inside_folder:
				copyFolder("%s/%s" %(origin, subfolder_name), "%s/%s" %(destination, subfolder_name), overwrite = overwrite, mpi_wait_for_everyone = False)

################################### copyFile ###################################

@only_the_first_processor_executes()
def copyFile(origin, destination, overwrite = False):
	"""
	Copy folder to another.
	"""
	if overwrite == False and os.path.exists("%s" % (destination)):
		raise ValueError(" ❌ ERROR: Folder '%s' exists!" %(destination))

	run_command_in_shell("/bin/cp '%s' '%s'" %(origin, destination), mode = 'print directly to terminal')

################################ checkIfFileExists #############################

@only_the_first_processor_executes(broadcast_result = True)
def checkIfFileExists(file_location):
	"""
	Check if file (or folder) exists.
	"""

	if os.path.exists("%s" % (file_location)):
		return True
	else:
		return False

########################## createFolderIfItDoesntExist #########################

@only_the_first_processor_executes()
def createFolderIfItDoesntExist(folder_location):
	"""
	Create folder if it doesn't exist.
	"""
	if not os.path.exists("%s" % (folder_location)):
		os.makedirs("%s" % (folder_location))

########################### removeFolderIfItExists #############################

@only_the_first_processor_executes()
def removeFolderIfItExists(folder_location):
	"""
	Remove folder if it exists.
	https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty
	https://stackoverflow.com/questions/1557351/python-delete-non-empty-dir
	"""

	if os.path.exists("%s" % (folder_location)):
		shutil.rmtree("%s" % (folder_location), ignore_errors = True)

####################### removeFolderContentIfItExists ##########################

@only_the_first_processor_executes()
def removeFolderContentIfItExists(folder_location):
	"""
	Remove all files inside folder if the folder exists.
	https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python
	"""

	if os.path.exists("%s" % (folder_location)):
		for root, dirs, files in os.walk(folder_location):
			for file_ in files:
				os.remove(os_path.join(root, file_))

########################### removeFileIfItExists #############################

@only_the_first_processor_executes()
def removeFileIfItExists(file_location):
	"""
	Remove file if it exists.
	"""

	if os.path.exists("%s" % (file_location)):
		os.remove("%s" % (file_location))

################# removeFilesInFolderAndSubfoldersIfItExists ###################

@only_the_first_processor_executes()
def removeFilesInFolderAndSubfoldersIfItExists(folder_location):
	"""
	Remove files in folder and subfolder if it exists.
	https://python-forum.io/Thread-Delete-files-inside-Folder-and-subfolders
	"""

	if os.path.exists("%s" % (folder_location)):
		for root_directory, folders, file_names in os.walk("%s" %(folder_location)):
			for file_name in file_names:
				customPrint(" ❗ Removing '%s'" %(os.path.join(root_directory, file_name)))
				os.remove(os.path.join(root_directory, file_name))

########################### renameFolderIfItExists #############################

@only_the_first_processor_executes()
def renameFolderIfItExists(folder_location, new_name_type = 'bak', new_folder_location = ''):
	"""
	Rename folder if it exists.
	https://datatofish.com/rename-file-python/
	"""

	if os.path.exists("%s" % (folder_location)):

		if new_name_type == 'bak':

			# Find name of folder available for renaming
			max_number = 100
			def recursive_search_for_available_folder(number = 1):
				"""
				Recursively search for an available folder.
				"""
				if number > max_number:
					raise ValueError(" ❌ ERROR: Reached maximum number (%d) of folders!" %(number))

				if not os.path.exists("%s_bak%03d" % (folder_location, number)):
					return "%s_bak%03d" % (folder_location, number)
				else:
					return recursive_search_for_available_folder(number = number + 1)

			new_folder_location = recursive_search_for_available_folder()

			customPrint(" ❗ Renaming '%s' to '%s'..." %(folder_location, new_folder_location))

			os.rename(r'%s' % (folder_location), r'%s' % (new_folder_location))

		elif new_name_type == 'new location':

			assert new_folder_location != ''

			if new_folder_location != folder_location:
				os.rename(r'%s' % (folder_location), r'%s' % (new_folder_location))

		else:
			raise ValueError(" ❌ ERROR: new_name_type == '%s' is not defined!" %(new_name_type))

############################# getFileNamesInFolder #############################

@only_the_first_processor_executes(broadcast_result = True)
def getFileNamesInFolder(folder_path):
	"""
	Get the file names in a folder.
	https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
	"""

	onlyfiles = [f for f in os.listdir(folder_path) if os_path.isfile(os_path.join(folder_path, f))]
	return onlyfiles

################## splitStringRemovingSubsequentSpaces #########################

def splitStringRemovingSubsequentSpaces(text):
	"""
	Split allowing various spaces
	https://stackoverflow.com/questions/6478845/python-split-consecutive-delimiters
	"""

	text_split = text.split()
	#text_split = re.split(' +', text) 
		# * text.split(' ') removes only one space...

	return text_split

########################### removeLastCharacter ################################

def removeLastCharacter(text):
	"""
	Remove the last character of a string.
	"""
	return text[:len(text)-1]

####################### convertToFoamUnitSpecification #########################

def convertToFoamUnitSpecification(unit_name):
	"""
	Convert a measurement unit to OpenFOAM specification format.

	https://cfd.direct/openfoam/user-guide/v7-basic-file-format/

	4.2.6 Dimensional units
		1       Mass                    kilogram (kg)   pound-mass (lbm)
		2       Length                  metre (m)       foot (ft)
		3       Time                    second (s)      second (s)
		4       Temperature             Kelvin (K)      degree Rankine (∘R)
		5       Quantity                mole (mol)      mole (mol)
		6       Current                 ampere (A)      ampere (A)
		7       Luminous intensity      candela (cd)    candela (cd) 

	-> These units work more like a "safeguard against implementing a meaningless operation".
	-> The values of the dimensioned constants should be changed by the user 
	   in 'controlDict'.

	"""

	if type(unit_name).__name__ == 'list':
		dimension_set = unit_name

	elif unit_name == 'kinematic viscosity':
		dimension_set = [0, 2, -1, 0, 0, 0, 0] # m²/s

	elif unit_name == 'dynamic viscosity':
		dimension_set = [1, -1, -1, 0, 0, 0, 0] # Pa s = kg/(m s)

	elif unit_name == 'density':
		dimension_set = [1, -3, 0, 0, 0, 0, 0] # kg/m³

	elif unit_name == 'time':
		dimension_set = [0, 0, 1, 0, 0, 0, 0] # s

	elif unit_name == 'length':
		dimension_set = [0, 1, 0, 0, 0, 0, 0] # s

	elif unit_name == 'dimensionless':
		dimension_set = [0, 0, 0, 0, 0, 0, 0] # [dimensionless]

	elif unit_name == 'pressure':
		dimension_set = [1, -1, -2, 0, 0, 0, 0] # kg/(m.s²)

	elif unit_name == 'rho-normalized pressure': # Pressure normalized by the density ("rho")
			# https://www.cfd-online.com/Forums/openfoam/69589-pressure-unit-boundary-condition.html
		dimension_set = [0, 2, -2, 0, 0, 0, 0] # kg/(m.s²)/(kg/m³) = m²/s²

	elif unit_name == 'specific heat':
		dimension_set = [0, 2, -2, -1, 0, 0, 0] # J/(kg.K) = kg m²/s² . 1/kg . 1/K = m²/s² . 1/K 

	elif unit_name == 'thermal conductivity':
		dimension_set = [1, 1, -3, -1, 0, 0, 0] # W/(m.K) = kg m²/s³ . 1/m . 1/K = kg m/s³ . 1/K

	elif unit_name == 'velocity':
		dimension_set = [0, 1, -1, 0, 0, 0, 0] # m/s

	elif unit_name == 'inverse permeability':
		dimension_set = [1, -3, -1, 0, 0, 0, 0] # kg/(m³.s)

	elif unit_name == 'volumetric heat transfer coefficient for the porous medium':
		dimension_set = [1, -1, -3, -1, 0, 0, 0] # W/(m³.K) = ((N.m)/s)/(m³.K) = (kg m/s²)/(m².K.s) = kg/(m.K.s³)

	elif unit_name == 'wall penalization factor for stabilized Eikonal equation':
		dimension_set = [0, -1, 0, 0, 0, 0, 0] # 1/m¹

	elif unit_name == 'temperature':
		dimension_set = [0, 0, 0, 1, 0, 0, 0] # K

	# Turbulent variables
	elif unit_name == 'specific turbulent kinetic energy': # k
		dimension_set = [0, 2, -2, 0, 0, 0, 0] # J/kg = m²/s²

	elif unit_name == 'specific rate of dissipation of turbulent kinetic energy': # epsilon
		dimension_set = [0, 2, -3, 0, 0, 0, 0]# W/kg = m²/s³

	elif unit_name == 'specific frequency of dissipation of turbulent kinetic energy': # omega
		dimension_set = [0, 0, -1, 0, 0, 0, 0] # W/J = s⁻¹

	elif unit_name == 'thermal diffusivity multiplied by the density': # alpha_th . rho
		dimension_set = [1, -1, -1, 0, 0, 0, 0] # m²/s . (kg/m³) = kg/(m s)

	elif unit_name == 'fluctuating velocity normal to the streamlines': # v2
		dimension_set = [0, 2, -2, 0, 0, 0, 0] # m²/s²

	elif unit_name == 'relaxation function': # f
		dimension_set = [0, 0, -1, 0, 0, 0, 0] # 1/s

	else:
		raise ValueError(" ❌ ERROR: unit_name == '%s' is not defined yet!" %(unit_name))

	return dimension_set

############################# findNearestSubArray ##############################

def findNearestSubArray(numpy_array, subarray, parameters_to_return = 'value'):
	"""
	Finds the nearest sub-array in a NumPy array.
	https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array#
	"""

	numpy_array = np.asarray(numpy_array)

	if ('array' not in type(subarray).__name__) and ('list' not in type(subarray).__name__):
		idx = (np.abs(numpy_array - subarray)).argmin()
	elif ('list' in type(subarray).__name__) and len(subarray) == 1:
		subarray = subarray[0]
		idx = (np.abs(numpy_array - subarray)).argmin()
	elif ('array' in type(subarray).__name__) and len(numpy_array.shape) == 1:
		subarray = subarray[0]
		idx = (np.abs(numpy_array - subarray)).argmin()
	else:
		subarray = np.asarray(subarray)
		idx = np.linalg.norm(numpy_array - subarray, axis = 1).argmin()

	if parameters_to_return == 'index':
		return idx
	elif parameters_to_return == 'value':
		return numpy_array[idx]
	elif parameters_to_return == 'index and value':
		return idx, numpy_array[idx]
	else:
		return numpy_array[idx]

################### createMapsBetweenCoordinateArrays ##########################

def createMapsBetweenCoordinateArrays(coord_array1, coord_array2, tol_mapping = None, additional_multiplier = 1.0, _retrial_number = 0):
	"""
	Create maps between coordinate arrays.
	"""

	#saveNumPyArrayToFile(coord_array1, '../coord_array1.npy')
	#saveNumPyArrayToFile(coord_array2, '../coord_array2.npy')
		#with open('../coord_array1.npy', 'wb') as file_np: np.save(file_np, coord_array1)
		#with open('../coord_array2.npy', 'wb') as file_np: np.save(file_np, coord_array2)
	#customPrint(xxx)

	customPrint(" 🌀 Creating the maps between the coordinate arrays (mapping tolerance = %e)..." %(tol_mapping*additional_multiplier))

	########################### Create the maps ############################
	# Works better for uniform meshes

	# Initialize coordinates maps
	assert len(coord_array1) == len(coord_array2)
	map_from_1_to_2 = [-1] * len(coord_array1)
	map_from_2_to_1 = [-1] * len(coord_array1)

	# Tolerance for the search
	assert type(tol_mapping).__name__ != 'NoneType'
	delta = tol_mapping*additional_multiplier

	# Detect if the coordinates are 3D
	assert len(coord_array1[0]) == len(coord_array2[0])
	flag3D = len(coord_array1[0]) == 3

	# Sort the coordinates from the x coordinate
	ind2x = coord_array2[:,0].argsort()
	coord_array2x = coord_array2[ind2x]

	# Print progress
	printProgress(0, max_iterations = len(coord_array1))

	# For all 'array 1' coordinates
	for i1 in range(len(coord_array1)):

		# x, y and z coordinates
		x1 = coord_array1[i1][0]
		y1 = coord_array1[i1][1]
		if flag3D:
			z1 = coord_array1[i1][2]

		# Binary search for the coordinate -> Index for "forward" search
		i2x = np.searchsorted(coord_array2x[:,0], x1)

		# Immediately smaller index -> Index for "backward" search
		i2xx = (i2x - 1) if i2x > 0 else 0

		# Search hopping between "forward" and "backward" (in a "zigzag"-like motion)
		found = False
		while i2x < len(coord_array1) or i2xx >= 0:

			# Search "forward"
			if i2x < len(coord_array1):
				i2 = ind2x[i2x]

				# If inside tolerance
				if (
					abs(y1 - coord_array2[i2][1]) < delta
					and abs(x1 - coord_array2[i2][0]) < delta
					and (not flag3D or abs(z1 - coord_array2[i2][2]) < delta)
					):
					found = True
					break

				# Increase index
				i2x += 1

			# Search "backward"
			if i2xx >= 0:
				i2 = ind2x[i2xx]

				# If inside tolerance
				if (
					abs(y1 - coord_array2[i2][1]) < delta
					and abs(x1 - coord_array2[i2][0]) < delta 
					and (not flag3D or abs(z1 - coord_array2[i2][2]) < delta)
					):
					found = True
					break

				# Decrease index
				i2xx -= 1

		# Not found, so let's stop here and return, because everything may be failing!
		if not found:
			customPrint(" ❌ ERROR: Could not locate coord_array1[%d] = %s. The mesh may possibly be too coarse, and the tolerance may have been set too large for this case... Discarding all mapping performed up to now..." %(i1, coord_array1[i1]))
			map_from_1_to_2 = [-1] * len(coord_array1)
			map_from_2_to_1 = [-1] * len(coord_array1)
			break

		# Save for the maps
		if map_from_2_to_1[i2] == -1:
			map_from_1_to_2[i1] = i2
			map_from_2_to_1[i2] = i1
		else:
			customPrint(" ❌ ERROR: coord_array1[%d] = %s could not be mapped, because one of the mappings had already been mapped before (map_from_1_to_2[%d] = %s, map_from_2_to_1[%d] = %s). The mesh may possibly be too coarse, and the tolerance may have been set too large for this case... Discarding all mapping performed up to now..." %(i1, coord_array1[i1], i1, map_from_1_to_2[i1], i2, map_from_2_to_1[i2]))
			map_from_1_to_2 = [-1] * len(coord_array1)
			map_from_2_to_1 = [-1] * len(coord_array1)
			break

		# Print progress
		printProgress(i1 + 1)

	################## Check if the first vertex failed ####################

	if i1 == 0:
		if _retrial_number < 2:

			# Get the current trial number
			trial_list = [2.0, 3.0]
				# 1.0 =>  5% (according to the input) -- First try
				# 2.0 => 10% (according to the input) -- _retrial_number = 0
				# 3.0 => 15% (according to the input) -- _retrial_number = 1
			additional_multiplier = trial_list[_retrial_number]

			customPrint(" 🌀 [Retry %d/2] Retrying for mapping tolerance = %e..." %(_retrial_number + 1, tol_mapping*additional_multiplier))

			return createMapsBetweenCoordinateArrays(coord_array1, coord_array2, tol_mapping = tol_mapping, additional_multiplier = additional_multiplier, _retrial_number = _retrial_number + 1)

		else:
			additional_multiplier = 1.0

	################## Check for mistakes and fix them #####################

	customPrint(" 🌀 Checking for mistakes and if it is needed to fix them...")

	# Check for mistakes when mapping, due to numerical precision
	tol_mesh = tol_mapping*additional_multiplier
	coord_array2_mapped_to_1 = coord_array2[map_from_1_to_2]
	mistakes2 = np.where(np.array([np.linalg.norm(coord_array1[i1] - coord_array2_mapped_to_1[i1]) for i1 in range(len(coord_array1))]) > tol_mesh)[0]

	# Check if any points were missed
	missed2 = np.where(np.array(map_from_1_to_2) == -1)[0]

	# If there are any missed points
	if len(missed2) > 0:
		customPrint(" ❌ ERROR: Missed %d points (for mapping tolerance = %e). Are you sure the meshes are corresponding?" %(len(missed2), tol_mapping))
		customPrint(" ❗ Appending %d points to the array of wrongly mapped points..." %(len(missed2)))
		mistakes2 = np.unique(np.append(mistakes2, missed2, axis = 0))

		current_timestamp = time.time()
		mapping_failure_folder = 'mapping_failure_%s' %(current_timestamp)
		customPrint(" ❗ Saving failed mappings and arrays to '%s' folder..." %(mapping_failure_folder))

		# Create uniquely named folder
		createFolderIfItDoesntExist(mapping_failure_folder)

		if utils_fenics_mpi.runningInSerialOrFirstProcessor():

			# Mapping data
			writeTextToFile("""
❌ ERROR: Missed %d points. Are you sure the meshes are corresponding?

 ------------------------------------------------------------------------------

Function:          utils/utils.py > createMapsBetweenCoordinateArrays
Time:              %s
Mapping tolerance: %e
Index that failed: i1 = %d

 ------------------------------------------------------------------------------

""" 				%(
				len(missed2),
				convertTimeStampToTime(current_timestamp),
				tol_mapping,
				i1
				), 
				'%s/data.txt' %(mapping_failure_folder))

			# Input arrays
			saveNumPyArrayToFile(coord_array1, '%s/coord_array1.npy' %(mapping_failure_folder))
			saveNumPyArrayToFile(coord_array2, '%s/coord_array2.npy' %(mapping_failure_folder))

			# Misses and mistakes
			saveNumPyArrayToFile(missed2, '%s/missed2.npy' %(mapping_failure_folder))
			saveNumPyArrayToFile(mistakes2, '%s/mistakes2.npy' %(mapping_failure_folder))

			# Maps
			saveNumPyArrayToFile(np.array(map_from_1_to_2), '%s/map_from_1_to_2.npy' %(mapping_failure_folder))
			saveNumPyArrayToFile(np.array(map_from_2_to_1), '%s/map_from_2_to_1.npy' %(mapping_failure_folder))

		# Wait for everyone!
		if wait_for_everyone == True:
			utils_fenics_mpi.waitProcessors()

	# If there are any mistakes
	if len(mistakes2) > 0:

		customPrint("\n ❗ %d vertices found with wrong mapping due to numerical precision. Attempting to manually fix each wrong mapping..." %(len(mistakes2)))

		# Print progress
		printProgress(0, max_iterations = len(mistakes2))

		# Go through all vertices of the OpenFOAM mesh and find the equivalent in the FEniCS mesh
		cont = 0
		for i1 in mistakes2:
		
			coord1 = coord_array1[i1]

			i2 = findNearestSubArray(coord_array2, coord1, parameters_to_return = 'index')

			map_from_2_to_1[i2] = i1
			map_from_1_to_2[i1] = i2

			# Print progress
			cont += 1
			printProgress(cont)

	##################### Print the overall results ########################

	customPrint(" 🌀 Mapping finished! Overall results:")
	coord_array2_mapped_to_1 = coord_array2[map_from_1_to_2]
	difference_arrays = abs(coord_array1 - coord_array2_mapped_to_1)

	customPrint(' ╎ x: [{:.7f} {:.7f}] | Δx = {:.7f}'.format(coord_array1[:,0].min(), coord_array1[:,0].max(), difference_arrays[:,0].max()))
	customPrint(' ╎ y: [{:.7f} {:.7f}] | Δy = {:.7f}'.format(coord_array1[:,1].min(), coord_array1[:,1].max(), difference_arrays[:,1].max()))
	if len(coord_array1[0]) == 3:
		customPrint(' ╎ z: [{:.7f} {:.7f}] | Δz = {:.7f}'.format(min(coord_array1[:,2]), max(coord_array1[:,2]), max(difference_arrays[:,2])))

	############################ Final checkup #############################

	if np.any(map_from_1_to_2 == -1):
		unmapped2 = np.where(map_from_1_to_2 == -1)[0]
		error_message = " ❌ ERROR: Mapping FAILED for %d vertices! Are you sure there are no overlapping vertices in the mesh?" %(len(unmapped))

		customPrint(error_message)
		customPrint(" ❗Listing first 100 unmapped vertices from array 1: ")

		for i in range(len(unmapped2)):
			unmapped_i1 = unmapped[unmapped_i1]
			if i >= 100:
				break
			else:
				customPrint(" i1 = %d => " %(unmapped_i1), coord_array1[unmapped_i1])

		raise ValueError(error_message)

	if np.any(map_from_2_to_1 == -1):
		unmapped1 = np.where(map_from_2_to_1 == -1)[0]
		error_message = " ❌ ERROR: Mapping FAILED for %d vertices! Are you sure there are no overlapping vertices in the mesh?" %(len(unmapped))

		customPrint(error_message)
		customPrint(" ❗Listing first 100 unmapped vertices from array 2: ")

		for i in range(len(unmapped1)):
			unmapped_i2 = unmapped[unmapped_i2]
			if i >= 100:
				break
			else:
				customPrint(" i2 = %d => " %(unmapped_i2), coord_array2[unmapped_i2])

		raise ValueError(error_message)

	################## Convert the maps to dictionaries ####################

	map_from_1_to_2 = {i1 : map_from_1_to_2[i1] for i1 in range(len(map_from_1_to_2))}
	map_from_2_to_1 = {i2 : map_from_2_to_1[i2] for i2 in range(len(map_from_2_to_1))}

	return map_from_1_to_2, map_from_2_to_1

############################ writeTextToFile ###################################

def writeTextToFile(text, file_name):
	"""
	Write some text to file.
	"""
	with open(file_name, 'w', encoding = 'utf-8') as file_txt: 
		file_txt.write(text)

########################## saveNumPyArrayToFile ################################

def saveNumPyArrayToFile(np_array, file_name):
	"""
	Save a NumPy array to file.
	"""
	with open(file_name, 'wb') as file_np: 
		np.save(file_np, np_array)

######################## loadNumPyArrayFromFile ################################

def loadNumPyArrayFromFile(file_name):
	"""
	Load a NumPy array from file.

	* Convert the array to .txt:
		loaded_array = np.load('coord_array1.npy')
		np.savetxt('coord_array1.txt', loaded_array, delimiter=' ')

	"""
	return np.load(file_name)

###################### order_list_from_preferred_order #########################

def order_list_from_preferred_order(list_orig, preferred_order = []):
	"""
	Order list from preferred order.
	"""
	list_new = []
	for i in range(len(preferred_order)):
		if preferred_order[i] in list_orig:
			list_new += [preferred_order[i]]

	for i in range(len(list_orig)):
		if list_orig[i] not in list_new:
			list_new += [list_orig[i]]
	return list_new

######################## findFoamVariableFolders ###############################

@only_the_first_processor_executes(broadcast_result = True)
def findFoamVariableFolders(problem_folder):
	"""
	Finds and orders all folders that are named as numbers, which 
	correspond, per OpenFOAM convention, to folders containing results.
	"""

	variable_names = getSubfolderNames(problem_folder, mpi_broadcast_result = False, mpi_wait_for_everyone = False)

	number_folders = []
	for var_name in variable_names:
		if var_name.isdigit():
			number_folders += [var_name]

	 # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
	sorted_indices = sorted(range(len(number_folders)), key = lambda k: float(number_folders[k]))

	sorted_number_folders =  ["%s" %(number_folders[i]) for i in sorted_indices]

	return sorted_number_folders

############################### strIsIn ########################################

def strIsIn(string1, string2, ignore_case = False):
	"""
	Check if a String is inside another.
	https://stackoverflow.com/questions/6579876/how-to-match-a-substring-in-a-string-ignoring-case
	"""
	if ignore_case == True:
		return re.search(string1, string2, re.IGNORECASE)
	else:
		return string1 in string2

################################# revertOrder ##################################

def revertOrder(list_to_revert):
	"""
	Revert a list or NumPy array.
	https://dbader.org/blog/python-reverse-list
	"""
	if type(list_to_revert).__name__ == 'list':
		return list_to_revert[::-1]
	elif type(list_to_revert).__name__ == 'ndarray':
		return np.flip(list_to_revert, axis = 0)
	else:
		raise ValueError(" ❌ ERROR: type(list_to_revert).__name__ == '%s' is not defined!" %(type(list_to_revert).__name__))

################## numpy_append_with_multiple_dimensions #######################

def numpy_append_with_multiple_dimensions(*args, **kwargs):
	"""
	Join multiple NumPy arrays.
	If each element has different dimensions, the final array will be of 
	dtype 'object' with elements of the original dtype (such as 'uint64')
	"""
	axis = kwargs.pop('axis', 0)
	try:
		
		full_array = np.append(*args, axis = axis)
		return full_array
	except:
		if axis == 0:
			list_gather = []
			for ar in args:
				list_gather += [ar[i] for i in range(len(ar))]

		else:
			raise ValueError(" ❌ ERROR: axis == %d is not implemented!" %(axis))

		return create_array_from_list_of_arrays(list_gather)

##################### create_array_from_list_of_arrays #########################

def create_array_from_list_of_arrays(list_of_arrays):
	"""
	Create an array from a list of arrays.
	The list of arrays may have the same size or not.
	"""

	if checkMaximumVersion(np, '1.21.6'): # Check the NumPy version
		return np.array(list_of_arrays)
	else:
		try:
			return np.array(list_of_arrays)
		except:
			return np.array(list_of_arrays, dtype = 'object')

######################## computeCenterCoordinates ##############################

def computeCenterCoordinates(vertex_array):
	"""
	Compute the center coordinate from a vertex list.
	"""
	return vertex_array.sum(axis = 0)/len(vertex_array)

########################### printFullDictionary ################################

def printFullDictionary(dictionary, indent=0, caracteres_before = "", compact_type = False):
	"""
	Prints an entire dictionary.
	https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
	"""

	for key, value in dictionary.items():
		if compact_type == False:
			customPrint('%s'%(caracteres_before), '│ \t' * indent + '│▶ ' + str(key))
			#customPrint('%s'%(caracteres_before), '│ \t' * indent + '│- ' + str(key))
			#customPrint('│ \t' * indent + '│- ' + str(key))
			#customPrint('│ \t' * indent + '│ ')
		else:
			#customPrint('%s'%(caracteres_before), '│ \t' * indent + '│- ' + str(key), end = "")
			customPrint('%s'%(caracteres_before), '│ \t' * indent + '│ ' + str(key), ": ", end = "")

		if isinstance(value, dict):
			if compact_type == False:
				pass
			else:
				customPrint()

			printFullDictionary(value, indent+1, caracteres_before = caracteres_before, compact_type = compact_type)
		else:
			if compact_type == False:
				customPrint('%s'%(caracteres_before), '│ \t' * (indent+1) + '│▶ '  + str(value))
				#customPrint('%s'%(caracteres_before), '│ \t' * (indent+1) + '│ ')
				#customPrint('%s'%(caracteres_before), '│ \t' * (indent+1) + '│- '  + str(value))
				#customPrint('│ \t' * (indent+1) + '│- '  + str(value))
				#customPrint('│ \t' * (indent+1) + '│ ')
			else:
				customPrint(str(value))
		if compact_type == False:
			customPrint('%s'%(caracteres_before), '│ \t' * indent + '│ ')
			#customPrint('│ \t' * indent + '│ ')
		else:
			pass

########################### get_max_unsigned_int_c #############################

def get_max_unsigned_int_c():
	"""
	Returns the maximum value of an unsigned int from C / C++.
	* FEniCS is implemented in C++, as well as OpenFOAM...
	https://stackoverflow.com/questions/13795758/what-is-sys-maxint-in-python-3
	"""
	import struct
	platform_c_max_unsigned_int = 2 ** (struct.Struct('i').size * 8) - 1
	return platform_c_max_unsigned_int

############################ searchMatchingIndices #############################

global _search_maching_indices_profile
_search_maching_indices_profile = {}

def resetSearchMatchingElements(profile_key = None):
	"""
	Reset the global variable.
	"""

	global _search_maching_indices_profile
	if type(profile_key).__name__ == 'NoneType':
		_search_maching_indices_profile = {} 
	else:
		if profile_key in _search_maching_indices_profile:
			_search_maching_indices_profile.pop(profile_key)

def searchMatchingElements(array_with_elements_to_find, array_with_findable_elements, reuse_previous_dictionary = True, profile_key = 'default'):
	"""
	Search matching indices by using Python dictionaries (i.e., hash tables*).

	* Something nice to remember is that the implementation of dictionaries in Python is through hash tables.
		( https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented )

	https://stackoverflow.com/questions/10320751/numpy-array-efficiently-find-matching-indices

	Other possible approaches without the bennefits of 'reuse_previous_dictionary':
			https://stackoverflow.com/questions/10320751/numpy-array-efficiently-find-matching-indices
		NumPy by using searchsorted (fast)
		Python using 'brute force' (slow)
		NumPy using 'brute force' (slow)
		NumPy using the bisect library (fast for large arrays)

	** Check out: https://stackoverflow.com/questions/10320751/numpy-array-efficiently-find-matching-indices
	"""

	global _search_maching_indices_profile

	if reuse_previous_dictionary == False:

		dictionary = defaultdict(list)
		for ind, ele in enumerate(array_with_findable_elements):
			dictionary[ele].append(ind)

		# Just some reinitializations
		if profile_key in _search_maching_indices_profile:
			_search_maching_indices_profile[profile_key]['dictionary'].clear()
			_search_maching_indices_profile[profile_key]['current_len_array_with_findable_elements'] = 0
	else:

		# Just some reinitializations
		if profile_key not in _search_maching_indices_profile:
			_search_maching_indices_profile[profile_key] = {}
			_search_maching_indices_profile[profile_key]['dictionary'] = defaultdict(list)
			_search_maching_indices_profile[profile_key]['current_len_array_with_findable_elements'] = 0

		dictionary = _search_maching_indices_profile[profile_key]['dictionary']
		current_len_array_with_findable_elements = _search_maching_indices_profile[profile_key]['current_len_array_with_findable_elements']

		if current_len_array_with_findable_elements == 0:
			new_findable_elements = array_with_findable_elements
		else:
			new_findable_elements = array_with_findable_elements[current_len_array_with_findable_elements:]

		for index_of_element, findable_element in enumerate(new_findable_elements):
			dictionary[findable_element].append(index_of_element)

		# Update array length. This is performed in order to check only the new elements in the next searches.
		_search_maching_indices_profile[profile_key]['current_len_array_with_findable_elements'] = len(array_with_findable_elements)

	# Gather the matching indices
	matching_indices = []
	for element_to_find in array_with_elements_to_find:
		if len(dictionary[element_to_find]) > 0: # Found element in dictionary!
			matching_indices += [element_to_find]

	return matching_indices

######################### findIndexOfElementInArray ############################

def findIndexOfElementInArray(complete_array, array_element):
	"""
	Find the index of the element by using the "rolling window" technique

	https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray
	https://www.reddit.com/r/learnpython/comments/2xqlwj/using_npwhere_to_find_subarrays/
	"""
	complete_array = np.asarray(complete_array)

	def rolling_window(a, window):
		shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
		strides = a.strides + (a.strides[-1],)
		return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
		   
	def findFirst_numpy(a, b):
		temp = rolling_window(a, len(b))

		#result = np.where(np.all(temp == b, axis=1))
		#return result[0][0] if result else None

		all_temp = np.all(temp == b, axis=1)

		result = np.where(all_temp[:,0] == True)
		for i in range(len(b)):
			if i > 0:
				ar1 = np.where(all_temp[:,i] == True)
				result = np.intersect1d(result, ar1)

		return result[0] if len(result) > 0 else None

	return findFirst_numpy(complete_array, array_element)

############################# checkIfInArray ###################################

def checkIfInArray(element, array_with_findable_elements, profile_key = 'checkIfInArray'):
	"""
	Check if an element is in an array.

	* Some other methods: list, set and bisect
		https://stackoverflow.com/questions/7571635/fastest-way-to-check-if-a-value-exists-in-a-list
	"""
	matching_elements = searchMatchingElements([element], array_with_findable_elements, profile_key = profile_key)
	return len(matching_elements) > 0

############################# Print progress ###################################
# Some useful time functions

global _progress_bar_step, _progress_bar_progress, _progress_bar_max_iterations
_progress_bar_step = 0
_progress_bar_progress = 0
_progress_bar_max_iterations = 10
_progress_bar_number_of_prints = 10
_progress_bar_block = "■"
_progress_bar_with_time_count = False

def printProgress(iteration_number, max_iterations = 10, number_of_prints = 10, type_of_progress_bar = 'line-by-line v2', reset_progress_bar = False, with_time_count = False):
	"""
	Print progress bar part from time to time.
	"""

	global _progress_bar_step, _progress_bar_progress, _progress_bar_max_iterations, _progress_bar_number_of_prints, _progress_bar_with_time_count
	if iteration_number == 0 or _progress_bar_progress == 0:
		_progress_bar_max_iterations = max_iterations

		_progress_bar_step = int(round(max_iterations/number_of_prints))
				# Round:    int(round(max_iterations/number_of_prints))
				# Truncate: int(max_iterations/number_of_prints)
		if _progress_bar_step == 0: 
			_progress_bar_step = 1

		adjusted_number_of_prints = max_iterations//_progress_bar_step
		_progress_bar_number_of_prints = adjusted_number_of_prints

		_progress_bar_progress = 0
		_progress_bar_with_time_count = with_time_count
		initTimeCount('_progress_bar_time_count', noprint = True)

	if _progress_bar_progress == 0:
		customPrint("\n", end = "")

	if type_of_progress_bar == 'update':

		if _progress_bar_progress == 0:
			customPrint(" ⌚ ▶[", end = "")
		elif _progress_bar_progress%_progress_bar_step == 0:
			customPrint("%s" %(_progress_bar_block), end = "")

		if iteration_number == _progress_bar_max_iterations:
			if _progress_bar_with_time_count == True:
				current_time_seconds = finishTimeCount('_progress_bar_time_count', noprint = True)

				# https://stackoverflow.com/questions/26276906/python-convert-seconds-from-epoch-time-into-human-readable-time
				hours = current_time_seconds//3600 
				minutes = int(current_time_seconds//60 % 60)
				seconds = current_time_seconds % 60
				 # * Not expecting this to take more than a day or so.

				current_time = "%02d:%02d:%02d" %(hours, minutes, seconds)

				customPrint("]◀ [%s]" %(current_time))

			else:
				customPrint("]◀")

	elif type_of_progress_bar == 'line-by-line':
		if _progress_bar_progress%_progress_bar_step == 0:
			if _progress_bar_with_time_count == True:
				current_time_seconds = finishTimeCount('_progress_bar_time_count', noprint = True)

				# https://stackoverflow.com/questions/26276906/python-convert-seconds-from-epoch-time-into-human-readable-time
				hours = current_time_seconds//3600 
				minutes = int(current_time_seconds//60 % 60)
				seconds = current_time_seconds % 60
				 # * Not expecting this to take more than a day or so.

				current_time = "%02d:%02d:%02d" %(hours, minutes, seconds)

				customPrint(" ⌚ ▶[ " + '%s'%(_progress_bar_block)*(_progress_bar_progress//_progress_bar_step), "]◀ [%s]" %(current_time))
			else:
				customPrint(" ⌚ ▶[ " + '%s'%(_progress_bar_block)*(_progress_bar_progress//_progress_bar_step), "]◀")

	elif type_of_progress_bar == 'line-by-line v2':

		if _progress_bar_progress%_progress_bar_step == 0:
			current_blocks = '%s'%(_progress_bar_block)*(_progress_bar_progress//_progress_bar_step)

			if _progress_bar_with_time_count == True:
				current_time_seconds = finishTimeCount('_progress_bar_time_count', noprint = True)

				# https://stackoverflow.com/questions/26276906/python-convert-seconds-from-epoch-time-into-human-readable-time
				hours = current_time_seconds//3600 
				minutes = int(current_time_seconds//60 % 60)
				seconds = current_time_seconds % 60
				 # * Not expecting this to take more than a day or so.

				current_time = "%02d:%02d:%02d" %(hours, minutes, seconds)
				customPrint(" ⌚ ▶[ %-*s ]◀ [%s]" %(_progress_bar_number_of_prints, current_blocks, current_time))
			else:
				customPrint(" ⌚ ▶[ %-*s ]◀" %(_progress_bar_number_of_prints, current_blocks))

	else:
		raise ValueError(" ❌ ERROR: type_of_progress_bar == '%s' is not defined!" %(type_of_progress_bar))

	if iteration_number < _progress_bar_max_iterations:

		# Increment progress
		_progress_bar_progress += 1

	elif iteration_number == _progress_bar_max_iterations:
		customPrint("")

	else:
		pass

################################ Time count ####################################
# Some useful time functions

time_count_dict = {}
def initTimeCount(tag, noprint = False):
	"""
	Initializes a time count.
	"""
	if noprint == True:
		pass
	else:
		customPrint(" ⌚ [%s] TIMED EXECUTION START" % (tag))
	start_time = time.time()
	time_count_dict[tag] = start_time

def finishTimeCount(tag, noprint = False):
	"""
	Finalizes a time count.
	"""
	try:
		start_time = time_count_dict[tag]
		final_time = time.time()
		if noprint == True:
			return final_time - start_time
		else:
			customPrint(" ⌚ [%s] TIMED EXECUTION FINISH: Elapsed time = %1.8f s" % (tag, final_time - start_time))
	except:
		if noprint == True:
			return None
		else:
			customPrint(" ⌚ [%s] TIMED EXECUTION FINISH: Elapsed time unavailable" % (tag))

####################### getOpenFOAMSolverLocation ##############################

@only_the_first_processor_executes(broadcast_result = True)
def getOpenFOAMSolverLocation(solver_name, max_levels_for_search = 10):
	"""
	Returns the location of the given OpenFOAM solver.
	"""

	# Location of the OpenFOAM solver
	count_search = 1
	while count_search <= max_levels_for_search:
		try:
			location_solver = run_command_in_shell("echo -n $(find $FOAM_SOLVERS -maxdepth %s | grep -w %s)" %(count_search, solver_name), mode = 'save output to variable')
			break
		except:
			location_solver = ""
			count_search += 1
			continue

	if location_solver == "":
		raise ValueError(" ❌ ERROR: Could not find solver \"%s\"!" %(solver_name))
	else:
		assert location_solver.count(solver_name) == 1, "❌ ERROR: Something went wrong when determining the location of the OpenFOAM solver '%s': location_solver = '%s'" %(solver_name, location_solver)

	return location_solver

################# getMustReadVariableNamesFromSolverFile #######################

@only_the_first_processor_executes(broadcast_result = True)
def getMustReadVariableNamesFromSolverFile(filename, variable_types, return_types = False):
	"""
	Returns the MUST_READ variable names from a file according to variable_types.
	"""

	# Previous version, without checking if the variable is MUST_READ or not.
	#
	#variable_names_from_solver = []
	#for var_type in variable_types:

	#	# Find all lines with variable names of the given type
	#	 # * It searches for all lines that start with var_type 
	#	var_lines_found = run_command_in_shell("echo -n \"$(grep '^%s' %s)\"" %(var_type, variable_definition_file), mode = 'save output to variable', indent_print = True).split('\n')

	#	# Get the names of the variables
	#	for i in range(len(var_lines_found)):

	#		# Split allowing various spaces
	#		var_name_split = splitStringRemovingSubsequentSpaces(var_lines_found[i]) #var_lines_found[i].split(' ')
	#		if len(var_name_split) >= 2:
	#			var_name_to_use = var_name_split[1]
	#			if var_name_to_use.endswith('('):
	#				var_name_to_use = var_name_to_use[:len(var_name_to_use) - 1] # Remove last character
	#			variable_names_from_solver += [var_name_to_use]

	# Create array of variables from solver file
	variable_names_from_solver = []
	variable_types_from_solver = []

	# Read all lines to file
	 # * This may be quite troubling for extremely large files (because of the memory consumption), 
	 #   but the file we are reading is small.
	with open(filename, "r", encoding = "utf-8") as text_file:
		lines = text_file.readlines()

	# For all lines
	for i in range(len(lines)):
		current_line = lines[i]
		for variable_type in variable_types:
			if current_line.startswith(variable_type):

				variable_name = current_line.split()[1].split('(')[0]

				found = False
				j = i
				while j < len(lines):
					searchline = lines[j]

					# OpenFOAM solvers for compressible flow normally define "p" with "thermo.p()"
					if "thermo.p()" in searchline:
						searchline = "MUST_READ"

					if "MUST_READ" in searchline:
						found = True
						variable_names_from_solver += [variable_name]
						variable_types_from_solver += [variable_type]

					elif "READ_IF_PRESENT" in searchline:
						found = True

					elif "NO_READ" in searchline:
						found = True

					elif "MUST_READ_IF_MODIFIED" in searchline:
						found = True

					if ");" in searchline:
						break

					j += 1

				if found == False:
					raise ValueError(" ❌ ERROR: variable_name == %s does not have a set readOption ('MUST_READ', 'READ_IF_PRESENT', 'NO_READ' or 'MUST_READ_IF_MODIFIED')!" %(variable_name))

				# Let's continue from j
				i = j

	if return_types == True:
		return variable_names_from_solver, variable_types_from_solver
	else:
		return variable_names_from_solver

############################# checkListInsideList ##############################

def checkListInsideList(list1, list2):
	"""
	Check if all elements of one list are inside the other list.
	"""

	for el1 in list1:
		found = False

		# Search list2 for el1
		for el2 in list2:
			if el1 == el2:
				found = True

		# If el1 not found in list2
		if found == False:
			return False

	return True

####################### copyFilesFromFolderToNewLocation #######################

def copyFilesFromFolderToNewLocation(problem_folder, destination_folder_name = '0', original_folder_name = '1', file_names_in_folder = None):
	"""
	Copy files from a folder to a new location.
	* It DOES NOT consider files inside subfolders!
	"""
	if destination_folder_name != original_folder_name:

		original_folder_path = '%s/%s' %(problem_folder, original_folder_name)
		destination_folder_path = '%s/%s' %(problem_folder, destination_folder_name)

		customPrint(" 🌀 Copying '%s' to '%s'..." %(original_folder_path, destination_folder_path))

		if type(file_names_in_folder).__name__ == 'NoneType':
			files_to_copy = getFileNamesInFolder(original_folder_path)
		else:
			files_to_copy = file_names_in_folder
	
		for file_to_copy in files_to_copy:
			original_file_path = '%s/%s' %(original_folder_path, file_to_copy)
			destination_file_path = '%s/%s' %(destination_folder_path, file_to_copy)
			copyFile(original_file_path, destination_file_path)
			substituteFoamFileLocation(destination_file_path, '%s' %(destination_folder_name))
	else:
	
		customPrint(" 🌀 Skipping copy, because the last folder is equal to the first folder: '%s'" %(destination_folder_name))

######################### substituteFoamFileLocation ###########################

@only_the_first_processor_executes()
def substituteFoamFileLocation(file_with_location_to_be_substituted, new_location_name):
	"""
	Substitute FoamFile location.
	"""
	# Current location: sed -n 's/^.*location.*\("[0-9]*"\).*/\1/p' U_copy
	# Substitute in new file: sed 's/^\(.*location.*\)\("[0-9]*"\)\(.*\)/\1"300"\3/g' U > U_temp
	# Substitute in-place: sed -i 's/^\(.*location.*\)\("[0-9]*"\)\(.*\)/\1"300"\3/g' U

	command = r"""sed -i 's/^\(.*location.*\)\("[0-9]*"\)\(.*\)/\1""" + "\"%s\"" %(new_location_name) + r"""\3/g' """ + file_with_location_to_be_substituted

	customPrint(" 🌀 Substituting FoamFile location...")
	run_command_in_shell(command, mode = 'print directly to terminal', indent_print = True, accept_empty_response = True)

########################### getLastLinesOfFile #################################

@only_the_first_processor_executes(broadcast_result = True)
def getLastLinesOfFile(file_location, number_of_lines = 5):
	"""
	Check the last lines of a file.
	"""
	last_lines_of_file = run_command_in_shell("echo -n $(tail --lines %d %s)" %(number_of_lines, file_location), mode = 'save output to variable', suppress_run_print = True)
	return last_lines_of_file

###################### createForkForAuxiliaryFunction ##########################

def createForkForAuxiliaryFunction(*args, **kwargs):
	"""
	Decorator for spawning a child process to continuously execute an
	auxiliary function while a main function is executed.
	"""

	def decorator(original_function):
		@functools.wraps(original_function) # For using the correct 'docstring' in the wrapped function
		def wrapper(*args, **kwargs):

			# Auxiliary function
			auxiliary_function_in_fork = kwargs.pop('auxiliary_function_in_fork', None)

			# Time interval in seconds
			time_interval_auxiliary_function_in_fork = kwargs.pop('time_interval_auxiliary_function_in_fork', 2)

			# Use lowest priority in plotting?
			use_lowest_priority_in_fork = kwargs.pop('use_lowest_priority_in_fork', False)

			if type(auxiliary_function_in_fork).__name__ == 'NoneType':

				# Execute original function
				original_output = original_function(*args, **kwargs)

				# Return original output
				return original_output
			else:

				from ..utils import utils_fenics_mpi

				# Create child process
				if utils_fenics_mpi.runningInFirstProcessor():
					child_pid = os.fork()
				else:
					child_pid = None

				if (child_pid == 0) and utils_fenics_mpi.runningInFirstProcessor(): # Child process

					# Use lowest priority for child process?
					if use_lowest_priority_in_fork == True:
						# Niceness -> [-20, 19]
							# Higher niceness => Lower priority
							# Lower niceness  => Higher priority
							# Default is 0
							# https://www.geeksforgeeks.org/python-os-nice-method/
						delta_niceness = +40 # To set the highest niceness (19)
						os.nice(delta_niceness)

					# Infinite loop
					while True:

						# Execute auxiliary function
						ret = auxiliary_function_in_fork()

						# Sleep for some seconds
						time.sleep(time_interval_auxiliary_function_in_fork)

						# Finish if the auxiliary function returns False
						if ret == False:
							break

					# Finish child process (* Just in case...)
					os._exit()

				else: # Parent process

					# Execute original function
					try:

						# Wait for everyone!
						if utils_fenics_mpi.runningInParallel():
							utils_fenics_mpi.waitProcessors()

						original_output = original_function(*args, **kwargs)

						if type(child_pid).__name__ != 'NoneType':

							# Kill child PID
							os.kill(child_pid, 9)

							# Check if the killed child PID exited
							 # If not checked, the child PID becomes
							 # "defunct" in the system (i.e., "dead", but with 
							 # others thinking that it is still alive).
							os.waitpid(child_pid, os.WNOHANG)

						# Wait for everyone!
						if utils_fenics_mpi.runningInParallel():
							utils_fenics_mpi.waitProcessors()

					except:

						# Wait for everyone!
						if utils_fenics_mpi.runningInParallel():
							utils_fenics_mpi.waitProcessors()

						if utils_fenics_mpi.runningInFirstProcessor():

							# Kill child PID
							os.kill(child_pid, 9)

							# Check if the killed child PID exited
							 # If not checked, the child PID becomes
							 # "defunct" in the system (i.e., "dead", but with 
							 # others thinking that it is still alive).
							os.waitpid(child_pid, os.WNOHANG)

						# Wait for everyone!
						if utils_fenics_mpi.runningInParallel():
							utils_fenics_mpi.waitProcessors()

						# Print the traceback
						import traceback
						traceback.print_exc()

						# Raise the exception
						raise ValueError(" ❌ ERROR: Some error occurred in the simulation!")

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

					# Run one last time the function from the child PID
					if utils_fenics_mpi.runningInFirstProcessor():
						auxiliary_function_in_fork()

					# Wait for everyone!
					if utils_fenics_mpi.runningInParallel():
						utils_fenics_mpi.waitProcessors()

					# Return original output
					return original_output

		return wrapper
	return decorator

###################### setToReinitializeShellEnvironment #######################

def setToReinitializeShellEnvironment(command_to_run_in_shell, variable_environment_reinitialization = {}):
	"""
	Set to reinitialize the shell variable environment.
	This is useful when running independent MPI codes.
	"""

	if len(variable_environment_reinitialization) == 0:
		return command_to_run_in_shell

	# Variable count
	var_num_current = 0

	# FEniCS TopOpt Foam export variables names
	fenics_topopt_foam_export_var_name_start = "TOP_ENV_VAR"

	# Export the scripts as Shell variables
	export_the_scripts = ""

	# Shell variables to set inside the environment
	scripts_to_set_in_environment = ""

	# Shell variables to "source" inside the environment
	source_scripts_in_environment = ""

	# For all environment scripts that need to be run
	num_environment_scripts = len(variable_environment_reinitialization['environment scripts'])
	for var_num_current in range(num_environment_scripts):

		environment_script = variable_environment_reinitialization['environment scripts'][var_num_current]

		# New Shell variable to set
		var_current = "%s%d" %(fenics_topopt_foam_export_var_name_start, var_num_current)

		# Shell variable to export
		export_the_scripts += "export %s=%s" %(var_current, environment_script)
		if var_num_current != num_environment_scripts - 1:
			export_the_scripts += "\n"

		# Shell variable to set in subshell
		scripts_to_set_in_environment += " %s=$%s" %(var_current, var_current)

		# "Source" the script inside the subshell only if the file effectively exists
		source_scripts_in_environment += """[ -f \\"$%s\\" ] && source \\\"$%s\\\"""" %(var_current, var_current)
		if var_num_current != num_environment_scripts - 1:
			source_scripts_in_environment += "\n"

	# Setup inside the environment
	additional_setup_inside_environment = variable_environment_reinitialization['additional setup inside environment']
	additional_setup_inside_environment = additional_setup_inside_environment.replace('\"', '\\"')
	if additional_setup_inside_environment == "":
		additional_setup_inside_environment = '# [None]'

	new_command_to_run_in_shell = """
# Export the scripts
%s

# Open a new Bash environment for running
env -i USER=$USER HOME=$HOME%s /bin/bash -c "

# Set the default Bash environment variables
source /etc/profile
source ~/.bash_profile

# Source the environment
%s

# Additional setup for the environment
%s

# Run the code
%s
"
""" %	(
	export_the_scripts,
	scripts_to_set_in_environment,
	source_scripts_in_environment,
	additional_setup_inside_environment,
	command_to_run_in_shell
	)

	return new_command_to_run_in_shell

#################### __remove_some_methods_from_doc ############################

def __remove_some_methods_from_doc(__pdoc__, __to_hide_in_the_main_page_of_the_docs__):
	"""
	Remove some methods from the documentation.
	https://pdoc3.github.io/pdoc/doc/pdoc/#overriding-docstrings-with-__pdoc__
	"""
	for to_hide in __to_hide_in_the_main_page_of_the_docs__:
		methods_to_hide = to_hide['hide']
		class_for_hiding = to_hide['class']

		if methods_to_hide == 'all':
			for method_name in class_for_hiding.__dict__:
				if not(method_name.startswith('_')):
					__pdoc__["%s.%s" %(class_for_hiding.__name__, method_name)] = False
		else:
			assert type(methods_to_hide).__name__ == 'list', " ❌ ERROR: Methods to hide '%s' (%s) is not a list!" %(methods_to_hide, type(methods_to_hide).__name__)
			for method_name in methods_to_hide:
				if not(method_name.startswith('_')):

					if method_name.endswith('*'):
						start_of_method_name = method_name[:-1]

						for method_name_ in class_for_hiding.__dict__:
							if not(method_name_.startswith('_')):
								if method_name_.startswith(start_of_method_name):
									__pdoc__["%s.%s" %(class_for_hiding.__name__, method_name_)] = False
					else:
						assert method_name in class_for_hiding.__dict__, " ❌ ERROR: Method '%s' unavailable in '%s'" %(method_name, class_for_hiding.__name__)
						__pdoc__["%s.%s" %(class_for_hiding.__name__, method_name)] = False

########################### checkMaximumVersion ################################

def checkMaximumVersion(lib, version_str):
	"""
	Check the maximum version of a library.
	Major.Minor.Review
	"""

	lib_version_split = lib.__version__.split('.')
	version_split = version_str.split('.')
	assert len(lib_version_split) == len(version_split)

	for i in range(len(lib_version_split)):
		version_i = int(float(version_split[i]))
		lib_version_i = int(float(lib_version_split[i]))
		if version_i <= lib_version_i:
			return False
	return True		


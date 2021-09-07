################################################################################
#                           Setup for installation                             #
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

# https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
# https://github.com/mdolab/dafoam/blob/master/setup.py

############################### Python libraries ###############################

import itertools
import setuptools

############################# Project information ##############################

from fenics_topopt_foam.__about__ import __name__ as library_name
from fenics_topopt_foam.__about__ import __version__, __description__, __author__, __maintainer__

############################ Setup configuration ###############################

#### Git website
# Get the Git website from LINK.txt file.
 # The LINK.txt file is included in MANIFEST.in for this task:
  # https://packaging.python.org/guides/using-manifest-in/
  # https://github.com/navdeep-G/setup.py
  # https://github.com/interpreters/pypreprocessor

#git_website = "https://github.com/diego-hayashi/fenics_topopt_foam"
import subprocess
p = subprocess.Popen("""echo $(cat LINK.txt | sed -ne 's/^Git repository link: \([https][^)]*\).*$/\\1/p')""", shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE, executable = '/bin/bash')
(stdoutdata, stderrdata) = p.communicate()
stdout_response = stdoutdata.decode()

git_website = stdout_response.split('\n')[0]

#### Project website

project_website = git_website

#### Keywords

keywords = "Python, OpenFOAM, FEniCS, dolfin-adjoint, Topology optimization"

#### Installation requirements

installation_requirements = [

	# NumPy
	'numpy',

	# mpi4py
	'mpi4py'

]

#### Extra requirements
# * "Extra", but needed in order to access all the functionality...

extra_requirements = {
	'finite elements': [
		# FEniCS
		'fenics-dijitso', 'fenics-dolfin', 'fenics-ffc' ,'fenics-fiat', 'fenics-ufl'
	],
	'adjoint model': [
		# dolfin-adjoint
		'dolfin-adjoint'
	],
	'plotting': [
		# Matplotlib
		'matplotlib'
	],
	'meshing': [
		# Meshio
		'meshio>=3'
	],
	'others' : [
		# PyFoam
		'PyFoam'
	],
}

############################ Developer information #############################

# Author names
author_names = [aut.split(' <')[0] for aut in __author__]

# Author e-mails
author_emails = [aut.split(' <')[1].split('>')[0] for aut in __author__]

# Maintainer names
maintainer_names = [maint.split(' <')[0] for maint in __maintainer__]

# Maintainer e-mails
maintainer_emails = [maint.split(' <')[1].split('>')[0] for maint in __maintainer__]

############################# Perform the setup ################################

# Gather all extra requirements
extra_requirements['all'] = list(itertools.chain(*extra_requirements.values()))

#### Get the long description from the read-me file
 # The read-me file is included in MANIFEST.in for this task:
  # https://packaging.python.org/guides/using-manifest-in/
  # https://github.com/navdeep-G/setup.py
  # https://github.com/interpreters/pypreprocessor
with open('README.md', 'r', encoding = 'utf-8') as f:
	long_description = f.read()

#### Setup
setuptools.setup(

	########################### Basic information ##########################

	# Name
	name = library_name,

	# Version
	version = __version__,

	############################## Packages ################################

	# Packages
	packages = setuptools.find_packages(library_name),
	package_dir = {
		library_name : library_name
	},

	############################# Requirements #############################

	# Python version
	python_requires = '>=3',

	# Install requirements
	install_requires = installation_requirements,

	# Extras requirements (not needed for basic installation, but required for 
	# being able to execute all functions)
	extras_require = extra_requirements,

	############################### Metadata ###############################

	# Author information
	author = author_names,
	author_email = author_emails,

	# Maintainer information
	maintainer = maintainer_emails,
	maintainer_email = maintainer_emails,

	# Description
	description = __description__,

	# Long description
	long_description = long_description,
	long_description_content_type = "text/markdown",

	# Keywords
	keywords = keywords,

	# Links
	url = project_website,
	project_urls = {
		"Source Code" : git_website,
	},

	# License
	license = "GPL version 3",

	# Classifiers
	 # https://pypi.org/pypi?%3Aaction=list_classifiers
	classifiers = [

		# GPL 3
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",

		# Python 3
		"Programming Language :: Python :: 3",

		# OpenFOAM is implemented in C++
		"Programming Language :: C++",

		# Bash
		"Programming Language :: Unix Shell",

		# Linux-only (using Bash)
		"Operating System :: POSIX :: Linux",

	],

	########################### Additional data ############################
	# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html

	# Include all data mentioned in MANIFEST.in
	include_package_data = True,

	# Exclude some data mentioned in MANIFEST.in
	exclude_package_data = {
		# Having just the license (COPYING) file after installation is enough
		"": ["README.md", "LINK.txt"]
	},

)



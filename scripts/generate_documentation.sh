#!/bin/bash

################################################################################
#                    📕️ Auto-generate the documentation 📕️                     #
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

################# Check if you are ready to run this script ####################

check_pdoc3_python=$(python -c "
try: 
 import pdoc
 print(\"Success\")
except ImportError:
 print(\"Fail\")
")

check_pdoc3_pip=$(pip list | grep 'pdoc3')

if [ "$check_pdoc3_python" == "Fail" ]||[ "$check_pdoc3_pip" == "" ]; then
	echo "
Missing dependency! Please, install pdoc3:
$ pip install --user pdoc3
"
	exit 1
fi

################################# Parameters ###################################

# Input parameters for this script
option=$1

# First parameter
 # Choose whether to consider the plugins in the documentation of the new/overloaded methods/classes/functions/variables or not
if [ "$option" == "noplugins" ]; then # Without the plugins

	# Add a new environment variable
	export FENICS_TOPOPT_FOAM_LOAD_PLUGINS="False"

else # Without the plugins

	# Add a new environment variable
	export FENICS_TOPOPT_FOAM_LOAD_PLUGINS="True"

fi

#################################### Data ######################################

# Library name
library_name='fenics_topopt_foam'
library_stylished_name='FEniCS TopOpt Foam'

# Images
image_folder='img'
aux_image_folder='scripts/docs/img'
logo_image='fenics_topopt_foam_logo.png'
favicon_image='fenics_topopt_foam_favicon.png'
background_image='mesh_background.png'

# Tags to substitute by the corresponding content
main_library_page_tag='MAIN_FENICS_TOPOPT_FOAM_INDEX_PAGE'
git_page_tag='GITPAGE'

######################### Get the necessary information ########################

# Get the information
date_GMT=$(date -u)
fenics_topopt_foam_version=$(cat ../$library_name/__about__.py | sed -ne "s/^__version__ = '\([^']*\)'.*/\1/p")
pdoc3_version=$(pdoc3 --version)

# Get the Git website from LINK.txt file.
git_page=$(cat ../LINK.txt | sed -ne "s/^Git repository link: \([https][^)]*\).*$/\1/p")

# Determine git logo style
if [[ "$git_page" == *"github"* ]]; then
	git_logo_style='github'
else
	git_logo_style='default'
fi

######################## Generate the background image #########################

printf "\n 📕️ Generating the background image...\n"

python docs/generate_mesh_background.py

################### Generate docs/aux_files/DOC_VERSION.md #####################

printf "\n 📕️ Setting up API documentation section...\n"

# Clear file
: >docs/aux_files/DOC_VERSION.md

#### Section: ❇️ API documentation

# Write to file
echo "<!-- ⭐️ File generated by generate_documentation.sh -->
## ❇️ API documentation

This is an auto-generated documentation from [markdown files and the source code ($library_stylished_name $fenics_topopt_foam_version)]($git_page), and was generated by using [$pdoc3_version](https://pdoc3.github.io/pdoc/). It is possible that it may contain typos and/or insufficient descriptions. Since this is a Python documentation, the C++ modules from $library_stylished_name are left outside this documentation (* but they can be viewed in the [source code]($git_page)). The main page of the documentation has reduced information of the available class methods (i.e., it only shows the basic and most commonly used set of methods), where the \"full\" functionality may be checked in the respective [sub-modules pages](#header-submodules).

" >> docs/aux_files/DOC_VERSION.md

#.. note::
#	The main page of the documentation has reduced information of the available class methods (i.e., it only shows the basic and most commonly used set of methods), where the \"full\" functionality may be checked in the respective [sub-modules pages](#header-submodules).

#.. tip::
#	The \"\`FoamSolver\`\" object is created and is accessible from the \"\`FEniCSFoamSolver\`\" object as \"\`fenics_foam_solver.foam_solver\`\". The \"\`FEniCSFoamSolver\`\" class does not extend the \"\`FoamSolver\`\" class, because not all \"\`FoamSolver\`\" methods have \"\`FEniCSFoamSolver\`\" counterparts.

################# Generate docs/custom_templates/logo.mako #####################

# Clear file
: >docs/custom_templates/logo.mako

#### Logo in sidebar

# Write to file
echo "<!-- ⭐️ File generated by generate_documentation.sh -->
<header> 
	<!-- ⭐️ Include home link with logo -->
	<a class=\"homelink\" rel=\"home\" href=\"$main_library_page_tag\"> 
		<span class=\"arrowtooltip2\">
			<img src=\"$logo_image\" alt=\"\"> 
			<span class=\"tooltiptext\">Go to the main page</span> 
		</span>
	</a>
	<br>
</header> 
" >> docs/custom_templates/logo.mako

################# Generate docs/custom_templates/credits.mako ##################

# Clear file
: >docs/custom_templates/credits.mako

#### Footer credits

# Write to file
echo "<!-- ⭐️ File generated by generate_documentation.sh -->
## ⭐️ Author information
<a href=\"$git_page\"><i>$library_stylished_name</i> $fenics_topopt_foam_version</a> API documentation.
" >> docs/custom_templates/credits.mako

############################ pdoc3 >> docs/html_doc ############################

# Using pdoc3
 # $ pip3 install --user pdoc3

dev_documentation_folder="docs/html_doc"

printf "\n 📕️ Auto-generating HTML documentation for $library_stylished_name...\n"
pdoc3 --html ../$library_name --output-dir $dev_documentation_folder --template-dir docs/custom_templates --force --config='minify=False'
	# --html = Generate .html
	# ../$library_name = Source code location
	# --output-dir $dev_documentation_folder = Output folder location
	# --template-dir docs/custom_templates = Use custom template
		# * The "docs/custom_templates" folder is copied from the "templates" from "pdoc3",
		 # including a small customization in "head.mako"
	# --force = Force overwrite
	# --config='minify=False' = Do NOT "minify" spaces and tabs
	    # 'show_source_code=False' = Ommit source code preview

# Delete the new environment variable
unset FENICS_TOPOPT_FOAM_LOAD_PLUGINS

##################### Logo, favicon etc. >> docs/html_doc ######################

printf "\n 📕️ Setting $library_stylished_name logo and favicon correct locations for the documentation files, and customizing lots of things...\n"

# Copy logo to all documentation folders
 # * This is due to pdoc using relative links to each file, which requires one 
 #   of the following alternatives:
 #     1) Copy the logo to each folder
 #     2) Get the logo from a website URL
 #     3) If pdoc3 is unable to solve it, let's edit manually by using the 'sed' command ⭐️
 # * The implementation below only works when there is ONLY ONE level of folders!

#### Copy images

/bin/cp ../$image_folder/$logo_image $dev_documentation_folder/$library_name
/bin/cp ../$aux_image_folder/$favicon_image $dev_documentation_folder/$library_name
/bin/cp ../$aux_image_folder/$background_image $dev_documentation_folder/$library_name

#### Fix the links to the images in 'index.html'

sed -i 's/'$image_folder'\/\('$logo_image'\)/\1/g' $dev_documentation_folder/$library_name/index.html
sed -i 's/'$main_library_page_tag'/index.html/g' $dev_documentation_folder/$library_name/index.html

#### Add link to index.html for any 'auto-generated API documentation' that appears in index.html.

sed -i 's/auto-generated API documentation/<a href="index.html">auto-generated API documentation<\/a>/g' $dev_documentation_folder/$library_name/index.html

#### Include the link to the 'examples' folder of the Git link in 'index.html'

# Include escape characters for using sed
 # https://stackoverflow.com/questions/407523/escape-a-string-for-a-sed-replace-pattern
escaped_examples_git_page=$(printf '%s\n' "$git_page/tree/main/examples" | sed -e 's/[\/&]/\\&/g')

# Include the link
#sed -i 's/some examples/<a href="'$escaped_examples_git_page'">some examples<\/a>/g' $dev_documentation_folder/$library_name/index.html
sed -i 's/"<code>examples<\/code>" folder/<a href="'$escaped_examples_git_page'">"examples" folder<\/a>/g' $dev_documentation_folder/$library_name/index.html

#### Change the name of the main page

sed -i 's/<title>'$library_name' API documentation<\/title>/<title>'"$library_stylished_name"'<\/title>/g' $dev_documentation_folder/$library_name/index.html

#### Customizations in the index ("navigation bar")

# Set a horizontal line above "❇️ API documentation" in the index ("navigation bar") and increase its size
sed -i 's/\(^<li><a href="#api-documentation">❇️ API documentation<\/a><\/li>\)/<br><hr class="hr_gray"><li><h3><a href="#api-documentation">❇️ API documentation<\/a><\/h3><\/li>/g' $dev_documentation_folder/$library_name/index.html

# Remove "Index" from the navigation bar, and add "$library_stylished_name" as the new name, also including the Git link
sed -i 's/\(<h1>Index<\/h1>\)//g' $dev_documentation_folder/$library_name/index.html

if [[ "$git_logo_style" == "default" ]]; then

	# Add a rocket emoji as the Git link
	sed -i 's/\(<div class="toc">\)/\1<br><hr class="hr_gray2"><h3>'"$library_stylished_name"' <span class=\"arrowtooltip2\"><a class=\"custom_button2\" href=\"'$git_page_tag'\">🚀<\/a><span class=\"tooltiptext\">Check the source code<\/span> <\/span><\/h3>/g' $dev_documentation_folder/$library_name/index.html

elif [[ "$git_logo_style" == "github" ]]; then
	
	# Add the GitHub logo from the Iconify project
	 # https://iconify.design/icon-sets/logos/github-icon.html
	 # * Anyway, the alternative is downloading the icon image and using it from https://github.com/logos ,
	   # but, if GitHub updates its logo, it would take some time until the
	   # old image is replaced in the documentation images (* Someone has to notice the change first).

	# Include escape characters for using sed
	# https://stackoverflow.com/questions/407523/escape-a-string-for-a-sed-replace-pattern
	escaped_include=$(printf '%s\n' "<script src="https://code.iconify.design/1/1.0.7/iconify.min.js"></script>" | sed -e 's/[\/&]/\\&/g')
	sed -i 's/<head>/'"$escaped_include"'/g' $dev_documentation_folder/$library_name/index.html

	# Add the GitHub logo
	sed -i 's/\(<div class="toc">\)/\1<br><hr class="hr_gray2"><h3>'"$library_stylished_name"' <span class=\"arrowtooltip2\"><a class=\"custom_button2\" href=\"'$git_page_tag'\"><span class="iconify" data-icon="logos-github-icon"><\/span><\/a><span class=\"tooltiptext\">Check the source code<\/span> <\/span><\/h3>/g' $dev_documentation_folder/$library_name/index.html

else
	echo "Error: Unknown logo style"
	exit 1
fi

#### Customizations in the main page

# Remove the big title "Package $library_name" from the beginning of the main page
 # <h1 class="title">Package <code>$library_name</code></h1>
sed -i 's/<h1 class="title">Package <code>'$library_name'<\/code><\/h1>/ /g' $dev_documentation_folder/$library_name/index.html

# Reinclude the big title "Package $library_name" later in the main page
sed -i 's/\(<h2 id="api-documentation">❇️ API documentation<\/h2>\)/\1<h1 class="title">Package <code>'$library_name'<\/code><\/h1>/g' $dev_documentation_folder/$library_name/index.html

# Add version number
sed -i 's/\(❇️ API documentation\)/\1 \(version '$fenics_topopt_foam_version'\)/g' $dev_documentation_folder/$library_name/index.html

## Set horizontal lines above the section names in the main page
sed -i 's/\(^<h2 id\)/<br><hr>\1/g' $dev_documentation_folder/$library_name/index.html

#### Adjustments

# Correcting the name just to make sure: It's pdoc3 and not pdoc
sed -i 's/\(pdoc \)/pdoc3 /g' $dev_documentation_folder/$library_name/index.html
sed -i 's/\(<cite>pdoc<\/cite> \)/<cite>pdoc3<\/cite> /g' $dev_documentation_folder/$library_name/index.html

#### Adjust the other .html files

# Edit the .html files
for current_dir in $(find $dev_documentation_folder/$library_name -type d); do

	# Files in '$library_name/plugins/[folder]'
	if [[ "$current_dir" == "$dev_documentation_folder/$library_name/plugins/"* ]] ; then

		for current_file in "$current_dir"/*.html; do

			#### Fix the links to the images

			sed -i 's/\('$logo_image'\)/..\/..\/\1/g' $current_file

			sed -i 's/\('$favicon_image'\)/..\/..\/\1/g' $current_file

			sed -i 's/\('$background_image'\)/..\/..\/\1/g' $current_file

			sed -i 's/'$main_library_page_tag'/..\/..\/index.html/g' $current_file

			#### Customizations in the index ("navigation bar")

			# Changing "Index" from the navigation bar to "❇️ API documentation (version X.X)"
			sed -i 's/\(<h1>Index<\/h1>\)/<h3>❇️ API documentation \(version '$fenics_topopt_foam_version'\)<\/h3>/g' $current_file

			#### Adjustments

			# Correcting the name just to make sure: It's pdoc3 and not pdoc
			sed -i 's/\(pdoc \)/pdoc3 /g' $current_file
			sed -i 's/\(<cite>pdoc<\/cite> \)/<cite>pdoc3<\/cite> /g' $current_file

		done

	# Files in '$library_name/[folder]'
	elif [ "$current_dir" != "$dev_documentation_folder/$library_name" ] ; then

		for current_file in "$current_dir"/*.html; do

			#### Fix the links to the images

			sed -i 's/\('$logo_image'\)/..\/\1/g' $current_file

			sed -i 's/\('$favicon_image'\)/..\/\1/g' $current_file

			sed -i 's/\('$background_image'\)/..\/\1/g' $current_file

			sed -i 's/'$main_library_page_tag'/..\/index.html/g' $current_file

			#### Customizations in the index ("navigation bar")

			# Changing "Index" from the navigation bar to "❇️ API documentation (version X.X)"
			sed -i 's/\(<h1>Index<\/h1>\)/<h3>❇️ API documentation \(version '$fenics_topopt_foam_version'\)<\/h3>/g' $current_file

			#### Adjustments

			# Correcting the name just to make sure: It's pdoc3 and not pdoc
			sed -i 's/\(pdoc \)/pdoc3 /g' $current_file
			sed -i 's/\(<cite>pdoc<\/cite> \)/<cite>pdoc3<\/cite> /g' $current_file

		done

	fi

done

#### Include 'docs/index.html' and its corresponding stylesheet

# Copy the main page of the documentation
cp docs/custom_files/index.html "$dev_documentation_folder"
cp docs/custom_files/css_stylesheet.css "$dev_documentation_folder"

#### Include the Git link in 'index.html' and '$library_name/index.html'

# Include escape characters for using sed
# https://stackoverflow.com/questions/407523/escape-a-string-for-a-sed-replace-pattern
escaped_git_page=$(printf '%s\n' "$git_page" | sed -e 's/[\/&]/\\&/g')

# Include the Git page
sed -i 's/'$git_page_tag'/'$escaped_git_page'/g' $dev_documentation_folder/index.html
sed -i 's/'$git_page_tag'/'$escaped_git_page'/g' $dev_documentation_folder/$library_name/index.html

######################### Remove previous documentation ########################

final_documentation_folder='../docs'

if [ -d "$final_documentation_folder" ]; then
	printf "\n 📕️ Removing previously generated documentation...\n"
	rm -rf ../docs/*
fi

########################### docs/html_doc >> docs ##############################

printf "\n 📕️ Moving the generated documentation to the 'docs' folder...\n"

mv docs/html_doc/* "$final_documentation_folder"

########################### Remove docs/html_doc ###############################

printf "\n 📕️ Removing the 'empty' 'docs/html_doc' folder...\n"

rm -r docs/html_doc

################################ README.md #####################################

printf "\n 📕️ Copying a README.md file to '../docs', for reference...\n"

echo "<!-- ⭐️ Don't edit this file. The editable file is in ../scripts/docs/README.md -->
$(cat docs/README.md)" > "$final_documentation_folder/README.md"

#################################### Finished ##################################

printf "\n ✅ Finished auto-generating HTML documentation for $library_stylished_name!
    Check it out in the '../docs' folder!\n\n"


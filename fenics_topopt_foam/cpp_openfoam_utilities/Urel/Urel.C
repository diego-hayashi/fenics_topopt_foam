/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    Urel

Description
    For each time: calculate and write the velocity in the rotating reference frame.

Modified by
    Diego Hayashi Alonso (2020-2021)

Based on
    https://bitbucket.org/akidess/urel/src/default/
    https://www.cfd-online.com/Forums/openfoam-post-processing/86972-how-get-relative-velocities-rotating-frame-reference.html
    https://github.com/mortbauer/relU
    http://www.cfd-online.com/Forums/openfoam-solving/71277-how-derive-relative-velocity-mrfsimplefoam.html
    https://www.cfd-online.com/Forums/openfoam-solving/71277-how-derive-relative-velocity-mrfsimplefoam-2.html

\*---------------------------------------------------------------------------*/

// Finite volumes
#include "fvCFD.H"

// MRF I/O
	// MRF = Moving Reference Frame
	// I/O = Input/Output
#include "IOMRFZoneList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[]){

	// Add options
	timeSelector::addOptions();

	// Set case
	#include "setRootCase.H"

	// Create time
	#include "createTime.H"

	// List of times
	instantList timeDirs = timeSelector::select0(runTime, args);

	// Create the mesh
	#include "createMesh.H"
	IOMRFZoneList mrfZone(mesh);

	// For all specified time folders
	forAll(timeDirs, timeI){

		// Set the current time
		runTime.setTime(timeDirs[timeI], timeI);

		// Print the current time
		Info<< "Time = " << runTime.timeName() << endl;

		// Check U
		IOobject Uheader(
			"U",
			runTime.timeName(),
			mesh,
			IOobject::MUST_READ
			);

		// Check if U exists
		if (Uheader.typeHeaderOk<volVectorField>(true)){

			// Update mesh
			mesh.readUpdate();

			// Read U
			Info<< "    Reading U" << endl;
			volVectorField U(Uheader, mesh);

			// Compute Urel
			Info<< "    Computing Urel" << endl;
			volVectorField Urel(
				IOobject(
					"Urel",
					runTime.timeName(),
					mesh,
					IOobject::NO_READ
				), U
			);
			mesh.readUpdate();

			// Assert that U and Urel are equal 
			Urel == U; 

			// Convert to relative velocity
			mrfZone.makeRelative(Urel); 

			// Write Urel to file
			Urel.write(); 

		}else{ // U does not exist
			Info<< "    No U is provided" << endl;
		}

		// End
		Info<< endl;
	}

	return 0;
}


// ************************************************************************* //

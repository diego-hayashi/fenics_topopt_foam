/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  7  
     \\/     M anipulation  |
  ---------------------------------------------------------------------------
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
    CustomSimpleFoam

Description
    Steady-state solver for incompressible, turbulent flow, using the SIMPLE
    algorithm, and also including a material model term (Brinkman model).

Modified from simpleFoam (from OpenFOAM) by 
    Diego Hayashi Alonso (2020-2021)

\* -------------------------------------------------------------------------- */

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "SolverUtils.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[]){

	/************************** Initial setup *****************************/

	#include "postProcess.H"
	#include "setRootCaseLists.H"
	#include "createTime.H"
	#include "createMesh.H"
	#include "createControl.H"
	#include "createFields.H"
	#include "initContinuityErrs.H"

	turbulence->validate();

	/**************************** Iterations ******************************/

	Info << "\nStarting time loop\n" << endl;

	volScalarField k_alpha(k_max + (k_min - k_max)*(alpha_design*(1 + q_penalization)/(alpha_design + q_penalization)));

	while (simple.loop(runTime)){

		Info << "Time = " << runTime.timeName() << nl << endl;

		/********** Solve the pressure-velocity formulation ***********/

		{
			#include "UEqn.H"
			{
			#include "pEqn.H"
			} 
		}

		laminarTransport.correct();

		/*************** Solve the turbulence equations ***************/

		turbulence->correct();

		runTime.write();

		Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
			<< "  ClockTime = " << runTime.elapsedClockTime() << " s"
			<< nl << endl;
	}

	Info << "End\n" << endl;

	return 0;
}

// ************************************************************************* //

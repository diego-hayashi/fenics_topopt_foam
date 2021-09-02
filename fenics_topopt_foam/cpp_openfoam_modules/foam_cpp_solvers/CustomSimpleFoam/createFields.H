/*----------------------------------------------------------------------------*\
                Create the necessary fields, models and variables
\* -------------------------------------------------------------------------- */

/*********************** Pressure-velocity formulation ************************/

Info<< "Reading field p\n" << endl;
volScalarField p (
	IOobject (
		"p",
		runTime.timeName(),
		mesh,
		IOobject::MUST_READ,
		IOobject::AUTO_WRITE
	),
	mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U (
	IOobject (
		"U",
		runTime.timeName(),
		mesh,
		IOobject::MUST_READ,
		IOobject::AUTO_WRITE
	),
	mesh
);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence(
	incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

/*********************** Material model configurations ************************/

volScalarField alpha_design (
	IOobject (
		"alpha_design",
		runTime.timeName(),
		mesh,
		IOobject::MUST_READ,
		IOobject::AUTO_WRITE
	),
	mesh
);

Info << "\nSetting the material model:\n"; 

IOdictionary materialmodelProperties (
	IOobject (
		"materialmodelProperties",
		runTime.constant(),
		mesh,
		IOobject::MUST_READ_IF_MODIFIED,
		IOobject::NO_WRITE
	)
);

dimensionedScalar q_penalization (
	"q_penalization",
	dimless, // dimensionSet(0, 0, 0, 0, 0, 0, 0),
	materialmodelProperties
);
printDimensionedScalarInformation(q_penalization);

dimensionedScalar k_max (
	"k_max",
	dimensionSet(1, -3, -1, 0, 0, 0, 0),
	materialmodelProperties
);
printDimensionedScalarInformation(k_max);

dimensionedScalar k_min (
	"k_min",
	dimensionSet(1, -3, -1, 0, 0, 0, 0),
	materialmodelProperties
);
printDimensionedScalarInformation(k_min);

dimensionedScalar rho_density (
	"rho_density",
	dimensionSet(1, -3, 0, 0, 0, 0, 0),
	materialmodelProperties
);
printDimensionedScalarInformation(rho_density);

Info << "\n"; 

/************************** Additional setup **********************************/

#include "createMRF.H"
#include "createFvOptions.H"





/*----------------------------------------------------------------------------*\
                                   SolverUtils
\* -------------------------------------------------------------------------- */

void printDimensionedScalarInformation(dimensionedScalar& ds){
	// Prints information from a dimensionedScalar.

	Info << "   " << ds.name() << " = " << ds.value(); 
	Info << "\n"; 
}



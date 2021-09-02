/*----------------------------------------------------------------------------*\
                   Momentum equation of the SIMPLE algorithm
\* -------------------------------------------------------------------------- */

/****************************** Momentum equation *****************************/

MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn(

	fvm::div(phi, U) 
	+ MRF.DDt(U) 
	+ turbulence->divDevReff(U)
	+ fvm::Sp(k_alpha/rho_density, U)

	==

	fvOptions(U)
);

fvVectorMatrix& UEqn = tUEqn.ref();
UEqn.relax();
fvOptions.constrain(UEqn);

/*************************** Momentum predictor *******************************/

if (simple.momentumPredictor()){
	solve(UEqn == -fvc::grad(p));
	fvOptions.correct(U);
}

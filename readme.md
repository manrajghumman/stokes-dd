Code for enforcing stokes with dd
currently can use cg with and 
without mortar and gmres 
without mortar.

Can plot the interface values
and residue values for cg and 
only properly plots interface values
for the gmres for now. Need to input
the correct residual to plot the residual.

Currently uses the Nitche form to weakly 
impose the Dirichlet condition. 

Mortar does not work with GMRES!

To plot the interface data copy test.m
file from main directory to build directory
and run and follow on screen instructions.
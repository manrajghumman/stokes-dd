Code for enforcing stokes with dd currently can use cg with and 
without mortar and gmres without mortar.

Can plot the interface values and residue values for cg and 
only properly plots interface values for the gmres for now. 
Need to input the correct residual to plot the residual.

Currently uses the Nitche form to weakly impose the Dirichlet condition. 

Mortar does not work with GMRES!

The output is in /build in folders 
 1. interface_data: has lambda, exact lambda and residual data. To plot these
 run the plot_interface.m file from /build and follow on screen instructions
 2. gnuplot_data: has mesh data and dof locations can be plotted with gnuplot
 3. convg_tables: convergence tables for each subdomain
 4. convg_table_total: convergence table for full domain

Code for modeling free fluid motion using 
Stokes equations with domain decomposition.
Problem can be scaled to multiple cores. 

Can use cg or gmres with and without mortar.

Mortar spaces can be
Q2, Q1, Q1-discont, Q0

For cg can plot (only 2D)
1. interface values with per inter face per x or y nodes.
2. residue values per inter face per x or y nodes.
3. norm of the residual as a function of iterations.

For gmres can plot (only 2D)
1. interface values with per inter face per x or y nodes.
2. norm of the residual as a function of iterations.

Flag chooses how external Dirichlet boundary condition is implemented
either weakly (via Nitche) or essentially.

Flag for choosing to print interface matrix. Is a compute
intensive operation but can import interface matrix into matlab
and play with it. Only works in 2D!

The output is in /output in folders 
 1. interface_data: has lambda, exact lambda, residual data and interface matrix. 
 To plot these run the plot_interface.m file from /build and follow on screen instructions.
 2. gnuplot_data: has mesh data and dof locations can be plotted with gnuplot.
 3. convg_tables: convergence tables for each subdomain.
 4. convg_table_total: convergence table for full domain.

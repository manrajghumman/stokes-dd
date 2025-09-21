Code for modeling free fluid motion using Stokes equations with domain decomposition.
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

Compilation instructions.
deal.ii 9.6.1 requirement (latest at the time)

Need deal.ii configured with mpi to compile and run the simulations. Latest version of dealii can be found at : https://www.dealii.org/download.html

deal.ii installation instruction: Follow readme file to install latest version of deal.ii with -DDEAL_II_WITH_MPI=ON and .. -DCMAKE_PREFIX_PATH=path_to_mpi_lib flags to cmake.

Compilation instructions.

cmake -DDEAL_II_DIR=/path to dealii installation folder/ . from the main directory

make release for faster compilations

make debug for more careful compilations with warnings

mpirun -n 'j' StokesDD where j is the number of subdomains(processses)

Please contact the author for further instructions.

Quick start guide for the simulator.

Most of the parameters mentioned above are fed to the executable file using parameters_stokes.prm in the main folder. This file can simply be modified without recompiling the program.

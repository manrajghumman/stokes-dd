/* ---------------------------------------------------------------------
 * This program implements multiscale mortar mixed finite element
 * method for stoke's equation.
 *
 * This implementation allows for non-matching grids by utilizing
 * the mortar finite element space on the interface. To speed things
 * up a little, the multiscale stress basis is also available for
 * the cases when the mortar grid is much coarser than the subdomain
 * ones.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Northwestern University, 2024
 *         Manraj Ghumman, University of Pittsburgh, 2025
 * based on Eldar Khattatov's Elasticity DD implementation from 2017.
 */

// Utilities, data, etc..
#include "../inc/stokes_mfedd.h"
#include "../inc/stokes_parameter_reader.h"
#include "../inc/parameter_interpreter.h"

// Main function is simple here
int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dd_stokes;
      using namespace Interpreter;

      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      MultithreadInfo::set_thread_limit(4);
      const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      // std::cout << "Thread limit: " << MultithreadInfo::n_threads() << ", this_mpi = " << this_mpi << std::endl;

      // Mortar mesh parameters (non-matching checkerboard)
      // int processes = 15;
      // std::vector<std::vector<unsigned int>> mesh_m2d(n_processes + 1);
      // mesh_m2d[0] = {2, 2}; // this is {x, y} for grid division
      // mesh_m2d[1] = {3, 3};
      // mesh_m2d[2] = {3, 3};
      // mesh_m2d[3] = {2, 2};//this one
      // mesh_m2d[4] = {1, 1};
      // for (unsigned int i = 0; i < mesh_m2d.size(); ++i)
      //   {
      //     mesh_m2d[i] = {1, 1};
      //   }
        // uncomment below and change manually to change
      // mesh_m2d[0] = {2, 2};
      // mesh_m2d[1] = {2, 2};
      // mesh_m2d[2] = {2, 2};
      // mesh_m2d[3] = {2, 2};
      // mesh_m2d[4] = {2, 2};
      // mesh_m2d[5] = {2, 2};
      // mesh_m2d[6] = {2, 2};
      // mesh_m2d[7] = {2, 2};
      // mesh_m2d[8] = {2, 2};

      // std::vector<std::vector<unsigned int>> mesh_m3d(9);
      // mesh_m3d[0] = {2, 2, 2};
      // mesh_m3d[1] = {3, 3, 3};
      // mesh_m3d[2] = {3, 3, 3};
      // mesh_m3d[3] = {2, 2, 2};
      // mesh_m3d[4] = {3, 3, 3};
      // mesh_m3d[5] = {2, 2, 2};
      // mesh_m3d[6] = {2, 2, 2};
      // mesh_m3d[7] = {3, 3, 3};
      // mesh_m3d[8] = {1, 1, 1};
      
      ParameterHandler prm;
      StokesParameterReader param(prm);
      param.read_parameters("../parameters_stokes.prm");
      Interpreter::StokesDD stokes_params;
      stokes_params.interpret_parameters(prm, this_mpi, n_processes);
      
      std::string name1("M0");
      std::string name2("M1");
      std::string name3("M2");
      std::string name4("M3");

      switch(stokes_params.dim)
      {
        case 2:
        {
          MixedStokesProblemDD<2> stokes(stokes_params.degree, 
                                         stokes_params.ess_dir_flag, 
                                         stokes_params.mortar_flag, 
                                         stokes_params.mortar_degree, 
                                         stokes_params.iter_meth_flag, 
                                         stokes_params.cont_mortar_flag, 
                                         stokes_params.print_interface_matrix_flag);
          stokes.run(stokes_params.refinements, 
                     stokes_params.boundary_m2d, 
                     stokes_params.mesh_m2d, 
                     stokes_params.tol, 
                     name1, 
                     stokes_params.max_iter, 
                     stokes_params.qdegree);
          break;
        }
        case 3:
        {
          AssertThrow(false, ExcMessage("3D implementation is not yet available."));
          // MixedStokesProblemDD<3> stokes(degree, 
          //                              ess_dir_flag, 
          //                              mortar_flag, 
          //                              mortar_degree, 
          //                              iter_meth_flag, 
          //                              cont_mortar_flag, 
          //                              print_interface_matrix_flag);
          //   stokes.run(refinements, 
          //              boundary_m2d, 
          //              mesh_m2d, 
          //              tol, 
          //              name1, 
          //              max_iter, 
          //              qdegree);
            break;
        }
        default:
          AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
      }
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}

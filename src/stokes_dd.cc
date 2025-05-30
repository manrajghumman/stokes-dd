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

// Main function is simple here
int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dd_stokes;

      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      MultithreadInfo::set_thread_limit(4);
      const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      // std::cout << "Thread limit: " << MultithreadInfo::n_threads() << ", this_mpi = " << this_mpi << std::endl;

      // Mortar mesh parameters (non-matching checkerboard)
      // int processes = 15;
      std::vector<std::vector<unsigned int>> mesh_m2d(n_processes + 1);
      // mesh_m2d[0] = {2, 2}; // this is {x, y} for grid division
      // mesh_m2d[1] = {3, 3};
      // mesh_m2d[2] = {3, 3};
      // mesh_m2d[3] = {2, 2};//this one
      // mesh_m2d[4] = {1, 1};
      for (unsigned int i = 0; i < mesh_m2d.size(); ++i)
        {
          mesh_m2d[i] = {2, 2};
        }
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

      std::vector<std::vector<unsigned int>> mesh_m3d(9);
      mesh_m3d[0] = {2, 2, 2};
      mesh_m3d[1] = {3, 3, 3};
      mesh_m3d[2] = {3, 3, 3};
      mesh_m3d[3] = {2, 2, 2};
      mesh_m3d[4] = {3, 3, 3};
      mesh_m3d[5] = {2, 2, 2};
      mesh_m3d[6] = {2, 2, 2};
      mesh_m3d[7] = {3, 3, 3};
      mesh_m3d[8] = {1, 1, 1};

      std::vector<unsigned int> boundary_m2d(4);
      // enter 0 for dirichlet and 1 for neumann boundary
      //{bottom, right, top, left}
      boundary_m2d.assign({1, 0, 0, 0}); 
      
      std::vector<unsigned int> boundary_m3d(6);
      boundary_m3d.assign({0, 0, 0, 0, 0, 0});

      // MixedStokesProblemDD(const unsigned int degree,
      //                  const bool ess_dir_flag          = 0, // 0 means Nitche, 1 means essential dir
      //                  const bool mortar_flag           = 0, // 0 means no mortar, 1 means mortar
      //                  const unsigned int mortar_degree = 0
      //                  const unsigned int iter_meth_flag= 0); // 0 means CG, 1 means GMRES
      std::vector<unsigned int> tmp(4);
      char tmp1, tmp2;
      MPI_Barrier(MPI_COMM_WORLD);
      if (this_mpi == 0)
      {
        std::cout << "Running Stokes Domain Decomposition with Mortar Finite Elements \n" 
                  << "This is a 2D example with matching grids.\n"
                  << "ess_dir (y/n): " << std::endl;
        {
          std::cin  >> tmp1;
          if (tmp1 == 'y')
            tmp[0] = 1; // essential dirichlet
          else if (tmp1 == 'n')
            tmp[0] = 0; // Nitsche
          else
            AssertThrow(false, ExcMessage("Invalid input for essential dirichlet flag. Use 'y' or 'n'."));
        }
        std::cout << "\nmortar (y/n): " << std::endl;
        {
          std::cin  >> tmp2;
          if (tmp2 == 'y')
          {
            tmp[1] = 1; // mortar
            std::cout << "\nmortar degree:" << std::endl;
            std::cin  >> tmp[2];
          } 
          else if (tmp2 == 'n')
          {
            tmp[1] = 0; // no mortar
            tmp[2] = 0; // no mortar degree if no mortar
          }  
          else
            AssertThrow(false, ExcMessage("Invalid input for mortar flag. Use 'y' or 'n'."));
        }
        std::cout << "\niterative method (0 for CG, 1 for GMRES): " << std::endl;
        std::cin  >> tmp[3];
      }
      // MPI_Barrier(MPI_COMM_WORLD);
      // Broadcast from root (rank 0) to all
      MPI_Bcast(&tmp[0], 4, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MixedStokesProblemDD<2> stokes(1, tmp[0], tmp[1], tmp[2], tmp[3]);
      MixedStokesProblemDD<2> mortars(1, tmp[0], tmp[1], tmp[2], tmp[3]);


      std::string name1("M0");
      std::string name2("M1");
      std::string name3("M2");
      std::string name4("M3");

      stokes.run(2, boundary_m2d, mesh_m2d, 1.e-10, name1, 100, 11);
      // mortars.run(5, boundary_m2d, mesh_m2d, 1.e-8, name1, 500, 11);
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

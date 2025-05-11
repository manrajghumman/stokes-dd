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

      MultithreadInfo::set_thread_limit(4);
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::cout << "Thread limit: " << MultithreadInfo::n_threads() << std::endl;

      // Mortar mesh parameters (non-matching checkerboard)
      std::vector<std::vector<unsigned int>> mesh_m2d(8);
      // mesh_m2d[0] = {2, 2}; // this is {x, y} for grid division
      // mesh_m2d[1] = {3, 3};
      // mesh_m2d[2] = {3, 3};
      // mesh_m2d[3] = {2, 2};//this one
      // mesh_m2d[4] = {1, 1};
      mesh_m2d[0] = {6, 6};
      mesh_m2d[1] = {6, 6};
      mesh_m2d[2] = {6, 6};
      mesh_m2d[3] = {6, 6};
      mesh_m2d[4] = {3, 3};
      mesh_m2d[5] = {2, 2};
      mesh_m2d[6] = {2, 2};
      mesh_m2d[7] = {2, 2};

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

      MixedStokesProblemDD<2> no_mortars(1);
      MixedStokesProblemDD<2> mortars(1, 1, 1);

      std::string name1("M0");
      std::string name2("M1");
      std::string name3("M2");
      std::string name4("M3");

      // no_mortars.run(5, boundary_m2d, mesh_m2d, 1.e-10, name1, 500, 11);
      mortars.run(5, boundary_m2d, mesh_m2d, 1.e-14, name1, 500, 11);
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

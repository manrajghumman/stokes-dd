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
          mesh_m2d[i] = {1, 1};
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

      // MixedStokesProblemDD(const unsigned int degree,
      //                  const bool ess_dir_flag          = 0, // 0 means Nitche, 1 means essential dir
      //                  const bool mortar_flag           = 0, // 0 means no mortar, 1 means mortar
      //                  const unsigned int mortar_degree = 0
      //                  const unsigned int iter_meth_flag= 0); // 0 means CG, 1 means GMRES
      ParameterHandler prm;
      StokesParameterReader param(prm);
      param.read_parameters("../parameters_stokes.prm");
      const unsigned int dim = prm.get_integer("dim");
      const unsigned int degree = prm.get_integer("degree");
      AssertThrow(dim == 2 || dim == 3, ExcMessage("Dimension must be either 2 or 3."));
      const unsigned int refinements = prm.get_integer("refinements");
      unsigned int qdegree = prm.get_integer("quad_degree"); // default quadrature degree, can be changed later
      // const std::vector<unsigned int> initial_grid = prm.get_integer_list("initial_grid");
      // AssertThrow(initial_grid.size() == dim, ExcMessage("Initial grid size must match the dimension."));
      prm.enter_subsection("External Boundary Conditions");
      std::vector<unsigned int> boundary_m2d;
      const bool ess_dir_flag = prm.get("essential_dirichlet") == "yes" ? 1 : 0;
      switch (dim)
      {
        case 2:
        {
          // Get the string and split into a vector of strings
          std::vector<std::string> bvec 
                      = Utilities::split_string_list(prm.get("boundary_2D")); 
          // Convert each string to a char
          for (const std::string &s : bvec)
            if (s[0] == 'N')
              boundary_m2d.push_back(1); // Neumann boundary
            else if (s[0] == 'D')
              boundary_m2d.push_back(0); // Dirichlet boundary
            else
              AssertThrow(false, ExcMessage("Invalid boundary condition type. Use 'N' for Neumann or 'D' for Dirichlet."));
          // Ensure the boundary vector has exactly 4 elements
          AssertThrow(boundary_m2d.size() == 4, ExcMessage("Boundary vector must have exactly 4 elements/edges for 2D."));
          break;
        }
        case 3:
        {
          // Get the string and split into a vector of strings
          std::vector<std::string> bvec 
                      = Utilities::split_string_list(prm.get("boundary_3D")); 
          // Convert each string to a char
          for (const std::string &s : bvec)
            if (s[0] == 'N')
              boundary_m2d.push_back(1); // Neumann boundary
            else if (s[0] == 'D')
              boundary_m2d.push_back(0); // Dirichlet boundary
            else
              AssertThrow(false, ExcMessage("Invalid boundary condition type. Use 'N' for Neumann or 'D' for Dirichlet."));
          // Ensure the boundary vector has exactly 4 elements
          AssertThrow(boundary_m2d.size() == 6, ExcMessage("Boundary vector must have exactly 6 elements/faces for 3D."));
          break;
        }
        default:
          AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
      }
      prm.leave_subsection();
      prm.enter_subsection("Mortar");
      const bool mortar_flag = prm.get("mortar") == "yes" ? 1 : 0;
      bool cont_mortar_flag;
      unsigned int mortar_degree;
      if (mortar_flag)
        {
          cont_mortar_flag = prm.get("continuous_mortar") == "yes" ? 1 : 0;
          mortar_degree = prm.get_integer("mortar_degree");
        }
      else
      {
        cont_mortar_flag = 1; // if no mortar, continuous mortar is default
        mortar_degree = 0; // if no mortar, mortar degree is 0
      }
      prm.leave_subsection();
      prm.enter_subsection("Solver");
      unsigned int iter_meth_flag;
      if (prm.get("iterative_method") == "CG")
        iter_meth_flag = 0; // CG
      else if (prm.get("iterative_method") == "GMRES")
        iter_meth_flag = 1; // GMRES
      else
        AssertThrow(false, ExcMessage("Invalid iterative method flag. Use 'CG' or 'GMRES'."));
      const unsigned int max_iter = prm.get_integer("max_iter");
      const double tol = prm.get_double("tol");
      prm.leave_subsection();
      prm.enter_subsection("Print Interface Matrix");
      const bool print_interface_matrix_flag = prm.get("print_interface_matrix") == "yes" ? 1 : 0;
      prm.leave_subsection();

      std::string name1("M0");
      std::string name2("M1");
      std::string name3("M2");
      std::string name4("M3");
      switch(dim)
      {
        case 2:
        {
          MixedStokesProblemDD<2> stokes(degree, 
                                         ess_dir_flag, 
                                         mortar_flag, 
                                         mortar_degree, 
                                         iter_meth_flag, 
                                         cont_mortar_flag, 
                                         print_interface_matrix_flag);
          stokes.run(refinements, 
                     boundary_m2d, 
                     mesh_m2d, 
                     tol, 
                     name1, 
                     max_iter, 
                     qdegree);
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

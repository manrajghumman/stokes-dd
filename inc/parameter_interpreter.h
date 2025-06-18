/*
file interpreting the parameters for various problems
*/
#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>
#include <vector>
#include <string>

namespace Interpreter
{
    /**
     * @brief Struct to interpret parameters for Stokes DD problem.
     * 
     * This struct interprets parameters read from a parameter file 
     * for a Stokes problem in either 2D or 3D, including boundary conditions,
     * mortar settings, and solver options.
     */
    // InterpretStokesDD is a struct that interprets parameters for a Stokes DD problem.
    // read from the parameter file parameters_stokes.prm
    using namespace dealii;
    using namespace dd_stokes;

    struct StokesDD
    {
        unsigned int dim;
        unsigned int degree;
        unsigned int refinements;
        unsigned int qdegree;
        std::vector<unsigned int> boundary_m2d;
        std::vector<std::vector<unsigned int>> mesh_m2d;
        std::vector<std::vector<unsigned int>> mesh_m3d;
        bool ess_dir_flag;
        bool mortar_flag;
        bool cont_mortar_flag;
        unsigned int mortar_degree;
        unsigned int iter_meth_flag;
        unsigned int max_iter;
        double tol;
        bool print_interface_matrix_flag;

        void interpret_parameters(dealii::ParameterHandler &prm,
                                  const unsigned int & this_mpi,
                                  const unsigned int &n_processes)
        {
            dim = prm.get_integer("dim");
            degree = prm.get_integer("degree");
            AssertThrow(dim == 2 || dim == 3, ExcMessage("Dimension must be either 2 or 3."));
            refinements = prm.get_integer("refinements");
            qdegree = prm.get_integer("quad_degree"); // default quadrature degree, can be changed later
            // const std::vector<unsigned int> initial_grid = prm.get_integer_list("initial_grid");
            // AssertThrow(initial_grid.size() == dim, ExcMessage("Initial grid size must match the dimension."));

            switch (dim)
                {
                    case 2:
                    {
                        // Get the string and split into a vector of strings
                        std::vector<int> bvec = Utilities::string_to_int(
                                        Utilities::split_string_list(prm.get("initial_grid_2D"))); 
                        AssertThrow(bvec.size() == dim, 
                                    ExcMessage("Initial grid size must match the dimension."));
                        // Initialize mesh_m2d with 2D grid sizes
                        // Uniform initial mesh for all processes
                        for (unsigned int i = 0; i < n_processes + 1; ++i)
                            mesh_m2d.push_back({static_cast<unsigned int>(bvec[0]), static_cast<unsigned int>(bvec[1])}); 
                        break;
                    }
                    case 3:
                    {
                        // Get the string and split into a vector of strings
                        std::vector<int> bvec = Utilities::string_to_int(
                                        Utilities::split_string_list(prm.get("initial_grid_3D"))); 
                        AssertThrow(bvec.size() == dim, 
                                    ExcMessage("Initial grid size must match the dimension."));
                        // Initialize mesh_m2d with 3D grid sizes
                        // Uniform initial mesh for all processes
                        for (unsigned int i = 0; i < n_processes + 1; ++i)
                            mesh_m3d.push_back({static_cast<unsigned int>(bvec[0]), 
                                static_cast<unsigned int>(bvec[1]), static_cast<unsigned int>(bvec[2])}); 
                        break;
                    }
                    default:
                        AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
                }
            
            prm.enter_subsection("External Boundary Conditions");
            {
                ess_dir_flag = prm.get("essential_dirichlet") == "yes" ? 1 : 0;
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
                                AssertThrow(false, ExcMessage("Invalid boundary condition type." 
                                    " Use 'N' for Neumann or 'D' for Dirichlet."));
                        // Ensure the boundary vector has exactly 4 elements
                        AssertThrow(boundary_m2d.size() == 4, 
                                ExcMessage("Boundary vector must have exactly 4 elements/edges for 2D."));
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
                                AssertThrow(false, 
                                    ExcMessage("Invalid boundary condition type."
                                                " Use 'N' for Neumann or 'D' for Dirichlet."));
                        // Ensure the boundary vector has exactly 4 elements
                        AssertThrow(boundary_m2d.size() == 6, 
                                    ExcMessage("Boundary vector must have exactly 6 elements/faces for 3D."));
                        break;
                    }
                    default:
                        AssertThrow(false, ExcMessage("Dimension must be either 2 or 3."));
                }
            }
            prm.leave_subsection();

            prm.enter_subsection("Mortar");
            {
                mortar_flag = prm.get("mortar") == "yes" ? 1 : 0;
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
            }
            prm.leave_subsection();

            prm.enter_subsection("Solver");
            {
                if (prm.get("iterative_method") == "CG")
                    iter_meth_flag = 0; // CG
                else if (prm.get("iterative_method") == "GMRES")
                    iter_meth_flag = 1; // GMRES
                else
                    AssertThrow(false, 
                        ExcMessage("Invalid iterative method flag. Use 'CG' or 'GMRES'."));
                max_iter = prm.get_integer("max_iter");
                tol = prm.get_double("tol");
            }
            prm.leave_subsection();

            prm.enter_subsection("Print Interface Matrix");
            {
                print_interface_matrix_flag = prm.get("print_interface_matrix") == "yes" ? 1 : 0;
            }
            prm.leave_subsection();
        }



    };
}   // namespace Interpreter
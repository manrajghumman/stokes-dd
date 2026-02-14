/* ---------------------------------------------------------------------
 * Declaration of MixedStokesProblemDD class template
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Ghumman, University of Pittsburgh, 2024
 * and Manu Jayadharan, Northwestern University, 2024
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017
 */

#ifndef STOKES_MFEDD_STOKES_MFEDD_H
#define STOKES_MFEDD_STOKES_MFEDD_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
// #include "utilities.h"
#include "projector.h"

namespace dd_stokes
{
  using namespace dealii;

  // Mixed Stokes Domain Decomposition class template
  template <int dim>
  class MixedStokesProblemDD
  {
  public:
  MixedStokesProblemDD(const unsigned int degree,
                       const bool ess_dir_flag                = 0,
                       const bool mortar_flag                 = 0,
                       const unsigned int mortar_degree       = 0,
                       const unsigned int iter_meth_flag      = 0,
                       const bool cont_mortar_flag            = 0,
                       const bool print_interface_matrix_flag = 0);

    void
    run(const unsigned int                            refine,
        const std::vector<unsigned int>              &boundary_def,
        std::vector<std::vector<unsigned int>> &reps,
        double                                        tol,
        std::string                                   name,
        unsigned int                                  maxiter,
        unsigned int                                  quad_degree);


  private:
    MPI_Comm   mpi_communicator;
    MPI_Status mpi_status;

    // Projector::Projector<dim> P_coarse2fine;
    // Projector::Projector<dim> P_fine2coarse;

    void
    make_grid_and_dofs(const std::vector<unsigned int> &boundary_def);

    void
    assemble_system();

    void
    get_interface_dofs();

    void
    assemble_rhs_star(FEFaceValues<dim> &fe_face_values);

    void
    solve_bar();

    void
    solve_star();

    // void
    // compute_multiscale_basis();
    void
    interface_matrix_column(BlockVector<double>     &local_flux_change,
                            unsigned int            &side,
                            const Quadrature<dim-1> &quad,
                            FEFaceValues<dim>       &fe_face_values,
                            std::vector<double>     &column);
    double
    inner_product_l2(BlockVector<double>     &vec1,
                     BlockVector<double>     &vec2,
                     unsigned int            &side,
                     const Quadrature<dim-1> &quad,
                     FEFaceValues<dim>       &fe_face_values);
    void
    print_interface_matrix(Projector::Projector<dim> P_coarse2fine,
                           Projector::Projector<dim> P_fine2coarse,
                           unsigned int &cycle);

    void
    local_gmres(const unsigned int &maxiter, unsigned int &cycle);

    void
    local_cg(const unsigned int &maxiter, unsigned int &cycle);

    // double
    // compute_interface_error(Function<dim> &exact_solution);

    void
    compute_errors(const unsigned int &cycle,
                   std::vector<std::vector<unsigned int>> &reps);

    void 
    write_dof_locations(const std::string &  filename) const;

    void 
    write_dof_locations_mortar(const std::string &  filename) const;

    void
    output_dof_results(const unsigned int cycle) const;

    void
    output_dof_results_mortar(const unsigned int cycle) const;

    void
    output_results(const unsigned int cycle) const;

    //For implementing GMRES
    void
	  givens_rotation(double v1, double v2, double &cs, double &sn);

    void
	  apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
    							unsigned int k_iteration);

    void
	  back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y);

    double vect_norm(std::vector<double> v);

    double
    inner_product_mortar(std::vector<std::vector<double>> &v_1,
                          std::vector<std::vector<double>> &v_2,
                          FEFaceValues<dim> &fe_face_values_mortar);


    unsigned int       gmres_iteration;
    double             residual;
    // Number of subdomains in the computational domain
    std::vector<unsigned int> n_domains;

    // FE degree and DD parameters
    const unsigned int degree;
    const bool ess_dir_flag;
    const bool mortar_flag;
    const bool cont_mortar_flag; // Flag for continuous mortar
    const bool print_interface_matrix_flag; // Flag to print interface matrix
    const unsigned int mortar_degree;
    const unsigned int iter_meth_flag;
    unsigned int       cg_iteration;
    double             tolerance;
    unsigned int       qdegree;
    double h;
    double h_old;
    double u_l2_error;
    double p_l2_error;
    double u_l2_error_old;
    double p_l2_error_old;
    double u_l2_error_total;
    double p_l2_error_total;
    double u_l2_error_old_total;
    double p_l2_error_old_total;
    

    // Neighbors and interface information
    std::vector<int>                       neighbors;
    std::vector<unsigned int>              faces_on_interface;
    std::vector<unsigned int>              faces_on_interface_mortar;
    std::vector<std::vector<unsigned int>> interface_dofs;
    std::vector<std::vector<unsigned int>> interface_dofs_fe;
    std::vector<double>                    interface_dofs_total;
    std::vector<std::vector<unsigned int>> interface_dofs_find_neumann;
    std::vector<unsigned int>              repeated_dofs;
    std::vector<unsigned int>              repeated_dofs_neumann;
    std::vector<unsigned int>              repeated_dofs_neumann_corner;

    unsigned long n_velocity_interface;
    unsigned long n_velocity_interface_fe;
    int           interface_dofs_size;

    // Subdomain coordinates (assuming logically rectangular blocks)
    Point<dim> p1;
    Point<dim> p2;

    // Fine triangulation
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    // Mortar triangulation
    Triangulation<dim> triangulation_mortar;
    FESystem<dim>      fe_mortar;
    DoFHandler<dim>    dof_handler_mortar;

    // Star and bar problem data structures
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> system_matrix_star;
    SparseDirectUMFPACK       A_direct;
    SparseDirectUMFPACK       B_direct;
    std::shared_ptr<SparseDirectUMFPACK> A_preconditioner;
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_mortar;
    AffineConstraints<double> constraints_star;
    BlockSparsityPattern      preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;
    FullMatrix<double>        interface_matrix;


    BlockVector<double> solution_bar_stokes;
    BlockVector<double> solution_star_stokes;
    BlockVector<double> solution;
    BlockVector<double> exact_solution_at_nodes;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_fe;
    std::vector<BlockVector<double>> exact_normal_stress_at_nodes_mortar;
    BlockVector<double> system_rhs_bar_stokes;
    BlockVector<double> system_rhs_star_stokes;
    std::vector<BlockVector<double>> interface_fe_function;
    std::vector<BlockVector<double>> interface_fe_function_fe;

    

    // Mortar data structures
    std::vector<BlockVector<double>>   interface_fe_function_mortar;
    std::vector<BlockVector<double>>   interface_fe_function_mortar_fe;
    BlockVector<double>                solution_bar_mortar;
    BlockVector<double>                solution_star_mortar;
    std::vector<BlockVector<double>>   multiscale_basis;


    // Output extra
    ConditionalOStream pcout;
    ConvergenceTable   convergence_table;
    ConvergenceTable   convergence_table_total;
    TimerOutput        computing_timer;
    


  };
} // namespace dd_stokes

#endif // STOKES_MFEDD_STOKES_MFEDD_H

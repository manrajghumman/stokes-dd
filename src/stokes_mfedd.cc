/* ---------------------------------------------------------------------
 * Implementation of the MixedStokesProblemDD class
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Northwestern University, 2024.
 * based on the Eldar Khattatov's Elasticity DD implementation from 2017.
 */

// Internals
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/sparse_direct.h>
#include <Eigen/Dense>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// Extra for MPI and mortars
#include <deal.II/base/timer.h>
#include <deal.II/numerics/fe_field_function.h>
// C++
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
// Utilities, data, etc.
#include "../inc/data.h"
#include "../inc/stokes_mfedd.h"
#include "../inc/utilities.h"
#include "../inc/interface.h"
#include "../inc/files.h"
#include "../inc/plot_interface.h"


namespace dd_stokes
{
  using namespace dealii;

  // MixedStokesDD class constructor
  template <int dim>
  MixedStokesProblemDD<dim>::MixedStokesProblemDD(
    const unsigned int degree,
    const bool ess_dir_flag,
    const bool mortar_flag,
    const unsigned int mortar_degree,
    const unsigned int iter_meth_flag,
    const bool cont_mortar_flag,
    const bool print_interface_matrix_flag)
    : degree(degree)
    , ess_dir_flag(ess_dir_flag)
    , mortar_flag(mortar_flag)
    , mortar_degree(mortar_degree)
    , iter_meth_flag(iter_meth_flag)
    , cont_mortar_flag((mortar_flag == 1 && mortar_degree == 0) ? false: cont_mortar_flag) // if mortar_degree is 0, use RT<dim> (0)
    , print_interface_matrix_flag(print_interface_matrix_flag)
    , mpi_communicator(MPI_COMM_WORLD)
    , P_coarse2fine(false)
    , P_fine2coarse(false)
    , n_domains(dim, 0)//vector of type unsigned int initialized to size dim with entries = 0
    , gmres_iteration(0)
	  , cg_iteration(0)
    , fe(FE_Q<dim>(degree+1),//fe for velocity
         dim,
         FE_Q<dim>(degree),//fe for pressure
         1)
    , dof_handler(triangulation)
    , dof_handler_mortar(triangulation_mortar)
    ,fe_mortar([mortar_degree, cont_mortar_flag]() -> FESystem<dim> {
      if (mortar_degree > 0 && cont_mortar_flag)
        return FESystem<dim>(FE_Q<dim>(mortar_degree), dim, FE_Nothing<dim>(), 1);
      else if (mortar_degree == 0 || !cont_mortar_flag)
        return FESystem<dim>(FE_RaviartThomas<dim>(mortar_degree), 1, FE_Nothing<dim>(), 1);
      else
        return FESystem<dim>(FE_Nothing<dim>(), dim, FE_Nothing<dim>(), 1);
    }()) // Conditional initialization using a lambda function
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  // MixedStokesProblemDD::make_grid_and_dofs
  template <int dim>
  void
  MixedStokesProblemDD<dim>::make_grid_and_dofs(const std::vector<unsigned int> &boundary_def)
  {
    TimerOutput::Scope t(computing_timer, "Make grid and DoFs");

    system_matrix.clear();
    
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);

    // Find neighbors
    neighbors.resize(GeometryInfo<dim>::faces_per_cell, 0);
    //initializes neighbours which is vector type int to size = faces per cell
    //GeometryInfo<dim>::faces_per_cell is a dimension independent way
    //of telling the faces per cell which is 2*dim
    find_neighbors(dim, this_mpi, n_domains, neighbors);
    //this is a function in utilities.h, it finds the neighbours 
    //for each subdomain processor

    // Make interface data structures
    faces_on_interface.resize(GeometryInfo<dim>::faces_per_cell, 0);
    faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell, 0);

    // Label interface faces and count how many of them there are per interface
    mark_interface_faces<dim>(triangulation, neighbors, p1, p2, faces_on_interface);
    if (mortar_flag)
      mark_interface_faces(
        triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);
    //this is a function in utilities.h

    std::vector<unsigned int> boundary_cond(2*dim);
    pcout << "boundary condition:";
    for (unsigned int i = 0; i < boundary_def.size(); ++i)
    {
      if (boundary_def[i] == 0)
        boundary_cond[i] = 0;
      else 
        boundary_cond[i] = 7;
      if (boundary_cond[i] == 7)
        pcout << " N";
      else if (boundary_cond[i] == 0)
        pcout << " D";
      else
        AssertThrow(false, ExcMessage("Invalid boundary condition type. Use 0 for Dirichlet or 7 for Neumann."));
    }
    pcout << "" << std::endl;
     for (const auto &cell : triangulation.active_cell_iterators()){
          for (const auto &face : cell->face_iterators()){
            if (face->center()[dim - 1] == 0)
              if (boundary_cond[0] != 0)
                face->set_all_boundary_ids(boundary_cond[0]);
            if (face->center()[dim - 2] == 1)
              if (boundary_cond[1] != 0)
                face->set_all_boundary_ids(boundary_cond[1]);
            if (face->center()[dim - 1] == 1)
              if (boundary_cond[2] != 0)
                face->set_all_boundary_ids(boundary_cond[2]);
            if (face->center()[dim - 2] == 0)
              if (boundary_cond[3] != 0)
                face->set_all_boundary_ids(boundary_cond[3]);
          }
        }
    

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    if (mortar_flag)
    {
      dof_handler_mortar.distribute_dofs(fe_mortar);
      DoFRenumbering::component_wise(dof_handler_mortar);
    }
      

    if (ess_dir_flag)
      {
        // Set up Dirichlet boundary conditions
        {
          constraints.clear();

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(dim);
          DoFTools::make_hanging_node_constraints(dof_handler, constraints);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                  0,
                                                  BoundaryValues<dim>(),
                                                  constraints,
                                                  fe.component_mask(velocities));// distributing velocity constraints eg dirichlet values
        }
        constraints.close();
        {
          constraints_star.clear();

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(dim);
          DoFTools::make_hanging_node_constraints(dof_handler, constraints_star);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                  0,
                                                  Functions::ZeroFunction<dim>(fe.n_components()),
                                                  constraints_star,
                                                  fe.component_mask(velocities));// distributing velocity constraints eg dirichlet values
        }
        constraints_star.close(); 
      }
    
    

    std::vector<types::global_dof_index> dofs_per_component =
            DoFTools::count_dofs_per_fe_component (dof_handler);

    // dofs_per_component.size() = dim x dim + dim + dim
    unsigned int n_u = 0, n_p = 0;
    // Enable for dim = 2, 3 right now only works for dim = 2
    for (unsigned int i = 0; i < dim; ++i)
      {
        n_u += dofs_per_component[i];
      }
    n_p = dofs_per_component[dim];

    n_velocity_interface = n_u;

    /**
     * @brief Constructs a BlockDynamicSparsityPattern object with 2x2 blocks.
     *
     * This object is used to store the sparsity pattern of a block matrix,
     * where each block can have a different sparsity pattern. The constructor
     * initializes a 2x2 block structure.
     */
    BlockDynamicSparsityPattern dsp(2, 2);

    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.collect_sizes();

    DoFTools::make_sparsity_pattern(dof_handler, dsp);//->this takes too much memory
    // in moderately-sized 3d problems
    // We instead use the following (see step-22)

    // {
    //   Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    //   for (unsigned int c = 0; c < dim + 1; ++c)
    //     for (unsigned int d = 0; d < dim + 1; ++d)
    //       if (!((c == dim) && (d == dim)))
    //         coupling[c][d] = DoFTools::always;
    //       else
    //         coupling[c][d] = DoFTools::none;

    //   DoFTools::make_sparsity_pattern(
    //     dof_handler, coupling, dsp, constraints, false);
      
      sparsity_pattern.copy_from(dsp);
    // }


    // Initialize system matrix
    system_matrix.reinit(sparsity_pattern);
    system_matrix_star.reinit(sparsity_pattern);

    // Reinit solution and RHS vectors
    solution_bar_stokes.reinit(2);
    solution_bar_stokes.block(0).reinit(n_u);
    solution_bar_stokes.block(1).reinit(n_p);
    solution_bar_stokes.collect_sizes();
    solution_bar_stokes = 0;

    // Reinit solution and RHS vectors
    solution_star_stokes.reinit(solution_bar_stokes);
    solution_star_stokes = 0;

    system_rhs_bar_stokes.reinit(solution_bar_stokes);
    system_rhs_bar_stokes = 0;

    system_rhs_star_stokes.reinit(solution_bar_stokes);
    system_rhs_star_stokes = 0;

    if (mortar_flag)
      {
        std::vector<types::global_dof_index> dofs_per_component_mortar =
            DoFTools::count_dofs_per_fe_component (dof_handler_mortar);

        unsigned int n_u_mortar = 0, n_p_mortar = 0;
        // Enable for dim = 2, 3 right now only works for dim = 2
        if (cont_mortar_flag && mortar_degree > 0)
          for (unsigned int i = 0; i < dim; ++i)
            n_u_mortar += dofs_per_component_mortar[i];
        else if (!cont_mortar_flag) // the case of RT elements
          n_u_mortar = dofs_per_component_mortar[0];

        n_p_mortar = dofs_per_component_mortar[dim];

        n_velocity_interface = n_u_mortar;
        n_velocity_interface_fe = n_u;

        solution_bar_mortar.reinit(2);
        solution_bar_mortar.block(0).reinit(n_u_mortar);
        solution_bar_mortar.block(1).reinit(n_p_mortar);
        solution_bar_mortar.collect_sizes();
        solution_bar_mortar = 0;

        solution_star_mortar.reinit(solution_bar_mortar);
        solution_star_mortar = 0;
        // solution_star_mortar.block(0).reinit(n_u_mortar);
        // solution_star_mortar.block(1).reinit(n_p_mortar);
        // solution_star_mortar.collect_sizes();
      }


    
    pcout << "N velocity dofs: " << n_u
    <<" | N pressure dofs: "<< n_p
    << std::endl;

    // exact_solution_at_nodes.reinit(solution_bar_stokes);
    exact_normal_stress_at_nodes_fe.resize(GeometryInfo<dim>::faces_per_cell);
    exact_normal_stress_at_nodes_mortar.resize(GeometryInfo<dim>::faces_per_cell);

    for (unsigned int face = 0;
             face < GeometryInfo<dim>::faces_per_cell;
             ++face)
      {
        exact_normal_stress_at_nodes_fe[face].reinit(solution_bar_stokes);
        exact_normal_stress_at_nodes_fe[face] = 0;
        NormalStressTensor_Exact<dim> exact_lambda(face);
        VectorTools::interpolate(dof_handler, exact_lambda, exact_normal_stress_at_nodes_fe[face]);
        exact_normal_stress_at_nodes_fe[face].block(0) *= -1;

        if (mortar_flag && cont_mortar_flag)
        {
          exact_normal_stress_at_nodes_mortar[face].reinit(solution_bar_mortar);
          exact_normal_stress_at_nodes_mortar[face] = 0;
          VectorTools::interpolate(dof_handler_mortar, exact_lambda, 
                    exact_normal_stress_at_nodes_mortar[face]);
          exact_normal_stress_at_nodes_mortar[face].block(0) *= -1;
        }
      }
  }


  // MixedStokesProblemDD - assemble_system
  template <int dim>
  void
  MixedStokesProblemDD<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "Assemble system");
    // const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    // const unsigned int this_mpi =
    //   Utilities::MPI::this_mpi_process(mpi_communicator);
    
    system_matrix         = 0;
    system_rhs_bar_stokes = 0;
    const double gamma    = 100;

    QGauss<dim> quadrature_formula(degree+2);
    QGauss<dim - 1> face_quadrature_formula(degree+2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors | update_gradients |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();


    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_face_dof_indices;
    local_face_dof_indices.resize(fe.dofs_per_face);

    const RightHandSide<dim>    right_hand_side;
    const RightHandSideG<dim>    right_hand_side_g;
    std::vector<Tensor<1, dim>> rhs_f_values(n_q_points, Tensor<1, dim>());
    StressTensor_Exact<dim> stress_tensor;
    BoundaryValues<dim> boundary_values;

    std::vector<double> rhs_g_values(n_q_points);
    std::vector<Tensor<2,dim>> stress_tensor_values(n_face_q_points, Tensor<2,dim>());

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>>          grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix                = 0;
        local_rhs                   = 0;
        double h_K                  = cell->diameter(); // std::pow(2, 0.5);      

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_f_values);
        right_hand_side_g.value_list(fe_values.get_quadrature_points(),
                                   rhs_g_values);                                       

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            { //compute at one place decrease computations
              symgrad_phi_u[k] =
                fe_values[velocities].symmetric_gradient(k, q);
              div_phi_u[k] = fe_values[velocities].divergence(k, q);
              phi_u[k]     = fe_values[velocities].value(k, q);
              phi_p[k]     = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i) // for test functions
            {
              for (unsigned int j = 0; j <= i; ++j)
                {
                  local_matrix(i, j) +=
                    (2 * (symgrad_phi_u[j] * symgrad_phi_u[i]) // 2* (D(u):D(v))
                    - phi_p[j] * div_phi_u[i]                 // - (p, div(v))
                    - div_phi_u[j] * phi_p[i])                // - (div(u), w)
                    * fe_values.JxW(q);                        // dx
                }
              local_rhs(i) += phi_u[i]            
                              * rhs_f_values[q]     // (F, v)
                              * fe_values.JxW(q)
                              -phi_p[i]            
                              * rhs_g_values[q]     // -(G, w)
                              * fe_values.JxW(q); // dx
            }
        }
        
        //Entering boundary integrals
        for (const auto &face : cell->face_iterators())
        {
          if (face->at_boundary())
          { 
            fe_face_values.reinit(cell, face);
            if (face->boundary_id() == 7) //Entering Neumann Condition
            {
              stress_tensor.value_list(fe_face_values.get_quadrature_points(), 
                                  stress_tensor_values);  
              for (unsigned int q_point = 0; q_point < n_face_q_points;
                   ++q_point)
                {
                  for (int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]     = fe_face_values[velocities].value(k, q_point);
                  }
                  const Tensor<1,dim> neumann_value =
                    (stress_tensor_values[q_point] *
                     fe_face_values.normal_vector(q_point));

                  for (int i = 0; i < dofs_per_cell; ++i)
                    local_rhs(i) +=
                      (phi_u[i] *                      
                       neumann_value) *                // + (gn, v)
                       fe_face_values.JxW(q_point);    // dx
                }
            }//Ending Neumann Condition
            if (face->boundary_id() == 0 && ess_dir_flag == 0)//Entering Dirichlet Condition
            {
              for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                Tensor<1,dim> gd; 
                Vector<double> tmp(dim);
                boundary_values.vector_value(fe_face_values.get_quadrature_points()[q], tmp);
                for (unsigned int i = 0; i < dim; ++i)
                    gd[i] = tmp[i];
                for (int k = 0; k < dofs_per_cell; ++k)
                {
                  symgrad_phi_u[k] =
                    fe_face_values[velocities].symmetric_gradient(k, q);
                  phi_u[k]     = fe_face_values[velocities].value(k, q);
                  grad_phi_u[k]     = fe_face_values[velocities].gradient(k, q);
                  phi_p[k]     = fe_face_values[pressure].value(k, q);
                }
                for (int i = 0; i < dofs_per_cell; ++i)
                {
                  for (int j = 0; j <= i; ++j)
                  {
                    local_matrix(i,j) += (-2 * (symgrad_phi_u[j] 
                                        * fe_face_values.normal_vector(q))
                                        * phi_u[i] // -2 <D(u).n, v>
                                        -2 * phi_u[j]
                                        * (symgrad_phi_u[i] 
                                          * fe_face_values.normal_vector(q)) // -2 <u, D(v).n>
                                          + gamma * phi_u[j]
                                          * phi_u[i] / h_K // + gamma <u,v> / h_k
                                          + phi_p[j]
                                          * (fe_face_values.normal_vector(q)
                                          * phi_u[i]) // + <p, v.n>
                                          + phi_u[j] 
                                          * fe_face_values.normal_vector(q)
                                          * phi_p[i]) // - <u.n, w>
                                          * fe_face_values.JxW(q);
                  }
                  local_rhs(i) += (-2 * gd * (symgrad_phi_u[i] 
                                * fe_face_values.normal_vector(q)) // -2 <gd, D(v).n>
                                + gamma * gd * phi_u [i] / h_K // gamma * <gd, v>/h
                                + gd * fe_face_values.normal_vector(q)
                                * phi_p[i]) // - <gd.n, w>
                                * fe_face_values.JxW(q);
                }
              }
            }//Ending Dirichlet Condition
          }
        }


        //Matrix is symmetric copying from other half
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              local_matrix(i, j) = local_matrix(j, i);

        cell->get_dof_indices(local_dof_indices);
        if (ess_dir_flag)
        {
          constraints.distribute_local_to_global(local_matrix,
                                                 local_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs_bar_stokes);
        }
        else
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                local_matrix(i, j));
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            system_rhs_bar_stokes(local_dof_indices[i]) += local_rhs(i);
        }
        
      }
  }

  
  
  
  // MixedStokesProblemDD - initialize the interface data structure
  template <int dim>
  void
  MixedStokesProblemDD<dim>::get_interface_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Get interface DoFs");
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<types::global_dof_index>());
    interface_dofs_fe.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<types::global_dof_index>());
    interface_dofs_find_neumann.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<types::global_dof_index>());
    interface_dofs_total.resize(0);

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    std::vector<types::global_dof_index> local_face_dof_indices, local_face_dof_indices_fe;

    typename DoFHandler<dim>::active_cell_iterator cell, endc, cell_fe, endc_fe;
    unsigned int side = 0;
    
    if (mortar_flag == 0)
      {
        cell = dof_handler.begin_active(), endc = dof_handler.end();
        local_face_dof_indices.resize(fe.dofs_per_face);
      }
    else
      {
        //first for fe/fine grid
        cell_fe = dof_handler.begin_active(), endc_fe = dof_handler.end();
        local_face_dof_indices_fe.resize(fe.dofs_per_face);
        for (; cell_fe != endc_fe; ++cell_fe)
        {
          for (unsigned int face_n = 0;
              face_n < n_faces_per_cell;
              ++face_n)
            if (cell_fe->at_boundary(face_n) &&
                (cell_fe->face(face_n)->boundary_id() != 0 && cell_fe->face(face_n)->boundary_id() !=7))
              {
                cell_fe->face(face_n)->get_dof_indices(local_face_dof_indices_fe, 0);
                side = cell_fe->face(face_n)->boundary_id() - 1;
                for (auto el : local_face_dof_indices_fe)
                {
                  if (el < n_velocity_interface_fe){
                    if (std::find (interface_dofs_fe[side].begin(), interface_dofs_fe[side].end(), el) 
                                                                  == interface_dofs_fe[side].end()){
                                                                      interface_dofs_fe[side].push_back(el);
                                                                  }
                  }
                }
              }
        }
        // now for mortar/coarse grid
        cell = dof_handler_mortar.begin_active(), endc = dof_handler_mortar.end();
        local_face_dof_indices.resize(fe_mortar.dofs_per_face);//it is shrunk to the mortar size but still has old elements
      }

    for (; cell != endc; ++cell)
      {
        for (unsigned int face_n = 0;
             face_n < n_faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0 && cell->face(face_n)->boundary_id() !=7))
            {
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
              side = cell->face(face_n)->boundary_id() - 1;
              for (auto el : local_face_dof_indices)
              {
                if (el < n_velocity_interface){
                    if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                                                  == interface_dofs[side].end()){
                                                                      interface_dofs[side].push_back(el);
                                                                      // interface_dofs_total.push_back(el);
                                                                  }
                    
                    if (std::find (interface_dofs_find_neumann[side].begin(), interface_dofs_find_neumann[side].end(), el) 
                                                                  == interface_dofs_find_neumann[side].end()){
                                                                      interface_dofs_find_neumann[side].push_back(el);
                                                                  }
                }
              }
            }
      }
    // compute the total interface dofs for this subdomain
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        if (std::find(interface_dofs_total.begin(), interface_dofs_total.end(), 
            interface_dofs[side][i]) == interface_dofs_total.end())
          interface_dofs_total.push_back(interface_dofs[side][i]);
    
    // if (mortar_flag == 0)
    // { 
    //   // Extracting the dirichlet and neumann dofs on the outside bdry shared between subdomains
    //   find_interface_dofs_dirichlet<dim>(dof_handler, interface_dofs, local_face_dof_indices, 
    //     n_velocity_interface, neighbors, repeated_dofs);
    //   find_interface_dofs_neumann<dim>(dof_handler, interface_dofs, local_face_dof_indices, 
    //     n_velocity_interface, neighbors, repeated_dofs_neumann);
    // }
    // else
    // {
    //   // Extracting the dirichlet and neumann dofs on the outside bdry shared between subdomains
    //   find_interface_dofs_dirichlet<dim>(dof_handler_mortar, interface_dofs, local_face_dof_indices, 
    //     n_velocity_interface, neighbors, repeated_dofs);
    //   find_interface_dofs_neumann<dim>(dof_handler_mortar, interface_dofs, local_face_dof_indices, 
    //     n_velocity_interface, neighbors, repeated_dofs_neumann);
    // }

    // // Extracting the neumann dofs on the interface corner point shared between subdomains  
    // find_interface_dofs_neumann_corner<dim>(interface_dofs_find_neumann, 
    //                       repeated_dofs_neumann_corner);
    
    compute_interface_dofs_size<dim>(interface_dofs_total, 
                                     mpi_communicator, 
                                     this_mpi, 
                                     interface_dofs_size);
    pcout << "N interface dofs: " << interface_dofs_size << std::endl;

    // if (this_mpi == 1)
    //   for (int side = 0; side < n_faces_per_cell; ++side)
    //     if (neighbors[side] >= 0)
    //     {
    //       std::cout << "\ninterface_dofs[side].size() = " << interface_dofs[side].size() << std::endl;
    //       for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
    //         std::cout << "\ninterface_dofs[" << side << "][" << i << "] = " << interface_dofs[side][i] << std::endl;   
    //       for (unsigned int i = 0; i < interface_dofs_total.size(); ++ i)
    //         std::cout << "\ninterface_dofs_total[" << i << "] = " << interface_dofs_total[i] << std::endl;    
    //     }
  }


  // MixedStokesProblemDD - assemble RHS of star problems
  template <int dim>
  void
  MixedStokesProblemDD<dim>::assemble_rhs_star(
    FEFaceValues<dim> &fe_face_values)
  {
    // TimerOutput::Scope t(computing_timer, "Assemble RHS star");
    system_rhs_star_stokes = 0;

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);

    // if (this_mpi == 1)
    //   std::cout << "Assembling RHS star for this_mpi = " << this_mpi << std::endl;

    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    std::vector<Tensor<1, dim>>          interface_values(n_face_q_points, Tensor<1, dim>());
    std::vector<Tensor<1, dim>>          tmp(n_face_q_points, Tensor<1, dim>());
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    int side;
    // if (this_mpi == 1)
    //   std::cout << "Assembling RHS star for this_mpi = " << this_mpi << std::endl;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        local_rhs                   = 0;
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int face_n = 0;face_n < n_faces_per_cell;++face_n)
          {
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() != 0) && (cell->face(face_n)->boundary_id() !=7)  )
              {
                //Need to reorder faces deal.ii iterators 
                //use a different ordering from freefem ... only works for 2D
                Assert(dim == 2,
                       ExcMessage("This function is only implemented for dim = 2."));
                fe_face_values.reinit(cell, face_n);
                if (face_n == 0)
                  side = 3;
                else if (face_n == 1)
                  side = 1;
                else if (face_n == 2)
                  side = 0;
                else 
                  side = 2;
                // std::cout << "interface_fe_function[side].block(0).size(): " << interface_fe_function[side].size() << " this_mpi, side = " << this_mpi << ", " << side << std::endl;
                // std::cout << "\ninterface_values = " << interface_values.size() << std::endl;
                fe_face_values[velocities].get_function_values(
                  interface_fe_function[side], interface_values);
                for (unsigned int q_point = 0; q_point < n_face_q_points;
                   ++q_point)
                {
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]  = fe_face_values[velocities].value(k, q_point); 
                  }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i){
                    local_rhs(i) +=
                      - (phi_u[i] * get_normal_direction(cell->face(face_n)->boundary_id()-1)*  // phi_v_i(x_q)
                       interface_values[q_point]) *    // Tn_i = lambda
                       fe_face_values.JxW(q_point);    // dx
                  }
                }
              }  
          }
        if (ess_dir_flag)
          {
            constraints_star.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs_star_stokes);
          }
        else
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              system_rhs_star_stokes(local_dof_indices[i]) += local_rhs(i);
          }
      }
  }


  // MixedStokesProblemDD::solvers
  template <int dim>
  void MixedStokesProblemDD<dim>::solve_bar()
  {
      TimerOutput::Scope t(computing_timer, "Solve bar");
    {
      A_direct.initialize (system_matrix);
      A_direct.vmult(solution_bar_stokes, system_rhs_bar_stokes);
      if (ess_dir_flag)
          constraints.distribute (solution_bar_stokes);
    }
  }


  template <int dim>
  void MixedStokesProblemDD<dim>::solve_star()
  {
      // TimerOutput::Scope t(computing_timer, "Solve star");
    {
      A_direct.vmult (solution_star_stokes, system_rhs_star_stokes);
      if (ess_dir_flag)
        constraints_star.distribute (solution_star_stokes);
    }
  }


  template <int dim>
  void
  MixedStokesProblemDD<dim>::print_interface_matrix(unsigned int &cycle)
  {
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    TimerOutput::Scope t(computing_timer, "Print interface matrix");
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    if ((n_processes > 2) && (n_processes % 2 == 0))
      AssertThrow(false,
                  ExcMessage("This function is only implemented for 2 or odd MPI processes."));
    AffineConstraints<double>   constraints;
    QGauss<dim - 1>    quad(qdegree);
    FEFaceValues<dim>  fe_face_values(fe,
                                     quad,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    long                n_interface_dofs = 0;
    FullMatrix<double>  local_matrix;

    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    
    n_interface_dofs = interface_dofs_total.size(); // number of interface dofs per subdomain
    interface_matrix.reinit(interface_dofs_size, interface_dofs_size);
    local_matrix.reinit(n_interface_dofs, n_interface_dofs);
    interface_matrix = 0;
    local_matrix = 0;

    // for (auto vec : interface_dofs)
    //   for (auto el : vec)
    //     n_interface_dofs += 1;
    std::vector<BlockVector<double>> lambda_basis;
    // lambda_basis.resize(n_interface_dofs);

    BlockVector<double> tmp_basis;
    BlockVector<double> local_flux_change;
    
    if (mortar_flag)
    {
      tmp_basis.reinit(solution_bar_mortar);
      local_flux_change.reinit(solution_bar_mortar);
    }
    else
    {
      tmp_basis.reinit(solution_bar_stokes);
      local_flux_change.reinit(solution_bar_stokes);
    }
    for (unsigned int face = 0; face < n_faces_per_cell; ++face)
      {
        interface_data_receive[face].resize(interface_dofs[face].size(),0); // intialize for all faces/sides!
        interface_data_send[face].resize(interface_dofs[face].size(), 0);
      }
    // std::cout << "\nhello1"<< " this_mpi = " << this_mpi << std::endl;
    // interface_fe_function[side].reinit(solution_bar_stokes);
    unsigned int ind = 0;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          {
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
            {
              std::fill(interface_data_receive[face].begin(), 
                        interface_data_receive[face].end(), 0.0);
              std::fill(interface_data_send[face].begin(), 
                        interface_data_send[face].end(), 0.0);
            }
            
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0)
                interface_fe_function[face] = 0;

            local_flux_change = 0;

            // std::cout << "\nhello2, side = " << side << " ind = " << ind << " this_mpi = " << this_mpi << std::endl;
            tmp_basis                          = 0;
            tmp_basis[interface_dofs[side][i]] = 1.0;
            if (mortar_flag)
            {
              project_mortar(P_coarse2fine,
                            dof_handler_mortar,
                            tmp_basis,
                            quad,
                            constraints_mortar,
                            neighbors,
                            dof_handler,
                            interface_fe_function[side]);
                  
              interface_fe_function[side].block(1) = 0;
            }
            else
              interface_fe_function[side] = tmp_basis;
            // std::cout << "\nhello2.5, side = " << side << " ind = " << ind << " this_mpi = " << this_mpi << std::endl;
            // std::cout << "interface_fe_function[side].size(): " << interface_fe_function[side].size() 
            //           << " this_mpi, side, ind = " << this_mpi << ", " << side << ", " << ind 
            //           << std::endl;
            // if (this_mpi == 1 && ind == 1)
            //   for (unsigned int j = 0; j < interface_fe_function[side].size(); ++j)
            //     {
            //       std::cout << "interface_fe_function[side][j]: " << interface_fe_function[side][j] << ", j = " << j 
            //                 << " this_mpi, side, ind = " << this_mpi << ", " << side << ", " << ind 
            //                 << std::endl;
            //     }
            assemble_rhs_star(fe_face_values);
            // std::cout << "\nhello2.51, side = " << side << " ind = " << ind << " this_mpi = " << this_mpi << std::endl;
            solve_star();
            // std::cout << "\nhello2.6, side = " << side << " ind = " << ind << " this_mpi = " << this_mpi << std::endl;
            if (mortar_flag)
            {
              project_mortar(P_fine2coarse,
                          dof_handler,
                          solution_star_stokes,
                          quad,
                          constraints_mortar,
                          neighbors,
                          dof_handler_mortar,
                          solution_star_mortar);
            
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  interface_data_send[side][i] = get_normal_direction(side) *
                                                    solution_star_mortar[interface_dofs[side][i]];
              }
            else
              for (unsigned int face = 0; face < n_faces_per_cell; ++face)
                if (neighbors[face] >= 0)
                  for (unsigned int i = 0; i < interface_dofs[face].size(); ++i)
                    interface_data_send[face][i] = get_normal_direction(face) *
                                                  solution_star_stokes[interface_dofs[face][i]];
              // for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              //   interface_data_send[side][i] = get_normal_direction(side) *
              //                                     solution_star_stokes[interface_dofs[side][i]];
            // std::cout << "\nhello3, side = " << side << " ind = " << ind << " this_mpi = " << this_mpi << std::endl
            MPI_Send(&interface_data_send[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      this_mpi,
                      mpi_communicator);
            MPI_Recv(&interface_data_receive[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      neighbors[side],
                      mpi_communicator,
                      &mpi_status);
            // std::cout << "\nhello4, side = " << side <<" ind = " << ind << " this_mpi = " << this_mpi << std::endl;
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              local_flux_change[interface_dofs[side][i]] = - (interface_data_send[side][i] +
                                                                  interface_data_receive[side][i]);
            for (unsigned int face = 0; face < n_faces_per_cell; ++face)
              if (neighbors[face] >= 0 && side != face)
                for (unsigned int i = 0; i < interface_dofs[face].size(); ++i)
                  local_flux_change[interface_dofs[face][i]] = - interface_data_send[face][i]; // the contribution from the other side
                                                                                          // is zero since lambda = 0 on the other side
            // needs a method later to add the local entries to the interface matrix
            for (unsigned int i = 0; i < n_interface_dofs; ++i)
              local_matrix(i,ind) += local_flux_change[interface_dofs_total[i]];
            ind += 1;
            pcout << "\r print interface matrix: " << ind << std::flush;
          }
    // if (this_mpi == 1)
    //   {
    //     std::cout << "\n local_matrix before exchange: \n" << std::endl;
    //     local_matrix.print(std::cout);
    //   }
    copy_matrix_local_to_global<dim>(local_matrix,
                                      interface_dofs,
                                      interface_dofs_size,
                                      this_mpi,
                                      n_processes,
                                      mpi_communicator,         
                                      interface_matrix);
    // if (this_mpi == 0)
    //   interface_matrix.print(std::cout);
    // if (this_mpi == 0)
    // {
    //   std::cout << "this_mpi = " << this_mpi
    //             << "\n before printing interface_matrix: \n" << std::endl;
    //             interface_matrix.print(std::cout);
    // }
    if (this_mpi == 0)
    {
      std::ofstream file;
      std::string dir = "../output/interface_data/";
      //name file
      if (mortar_flag)
            dir = dir + "mortar";
      else
            dir = dir + "/fe";
      file.open(dir + "/interface_matrix" + "_" 
                    + Utilities::int_to_string(cycle, 1) + ".txt", 
                    std::ios::out | std::ios::trunc);
      //insert data from interface_matrix into file
      for (unsigned int i = 0; i < interface_dofs_size; ++i)
        {
          for (unsigned int j = 0; j < interface_dofs_size; ++j)
            file << interface_matrix[i][j] << " ";
          file << "\n";
        }
      file.close();
    }
  }


  //Functions for GMRES:-------------------


  //finding the l2 norm of a std::vector<double> vector
  template <int dim>
  double
  MixedStokesProblemDD<dim>::vect_norm(std::vector<double> v){
  	double result = 0;
  	for(unsigned int i=0; i<v.size(); ++i){
  		result+= v[i]*v[i];
  	}
  	return sqrt(result);

  }
  //Calculating the given rotation matrix
  template <int dim>
  void
  MixedStokesProblemDD<dim>::givens_rotation(double v1, double v2, double &cs, double &sn){

  	if(abs(v1)<1e-25){
  		cs=0;
  		sn=1;
  	}
  	else{
  		double t = sqrt(v1*v1 + v2*v2);
  		cs = abs(v1)/t;
  		sn=cs*v2/v1;
  	}


  }

  //Applying givens rotation to H column
  template <int dim>
  void
  MixedStokesProblemDD<dim>::apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
                                                   unsigned int k_iteration){
	  int k=k_iteration;
  	assert(h.size()>k+1); //size should be k+2
  	double temp;
  	for( int i=0; i<=k-1; ++i){
  		temp= cs[i]* h[i]+ sn[i]*h[i+1];
  		h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1];
  		h[i] = temp;
  	}
  	assert(h.size()==k+2);
  	//update the next sin cos values for rotation
  	double cs_k=0, sn_k=0;
  	 givens_rotation(h[k],h[k+1],cs_k,sn_k);


  	 //Eliminate H(i+1,i)
  	 h[k] = cs_k*h[k] + sn_k*h[k+1];
  	 h[k+1] = 0.0;
  	 //adding cs_k and sn_k as cs(k) and sn(k)
  	 cs.push_back(cs_k);
  	 sn.push_back(sn_k);
  }


  template <int dim>
  void
  MixedStokesProblemDD<dim>::back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y){
  	 int k = beta.size()-1;
  	 assert(y.size()==beta.size()-1);
  	 for(int i=0; i<y.size();i++)
  		 y[i]=0;
  	for( int i =k-1; i>=0;i-- ){
  		y[i]= beta[i]/H[i][i];
  		for( int j = i+1; j<=k-1;j++){
  			y[i]-= H[j][i]*y[j]/H[i][i];
  		}
  	}

  }

  //local GMRES function.
  template <int dim>
  void
  MixedStokesProblemDD<dim>::local_gmres(const unsigned int &maxiter, unsigned int &cycle)
  {
    TimerOutput::Scope t(computing_timer, "Local CG");

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    interface_fe_function.resize(n_faces_per_cell);
    interface_fe_function_fe.resize(n_faces_per_cell);
    interface_fe_function_mortar.resize(n_faces_per_cell);
    interface_fe_function_mortar_fe.resize(n_faces_per_cell);

    std::vector<std::vector<double>> lambda(n_faces_per_cell);
    std::vector<std::vector<double>> lambda_fe(n_faces_per_cell);

    std::vector<std::ofstream> file(n_faces_per_cell);
    std::vector<std::ofstream> file_exact(n_faces_per_cell);
    std::vector<std::ofstream> file_residual(n_faces_per_cell);

    std::vector<std::ofstream> file_y(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_y(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_y(n_faces_per_cell);

    std::ofstream file_residual_total;
    std::ofstream file_residual_total_mortar;

    //for plotting data storage
    std::vector<std::vector<double>> plot(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact(n_faces_per_cell);
    std::vector<std::vector<double>> plot_y(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_y(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_y(n_faces_per_cell);

    name_files<dim>(this_mpi, 0, cycle, neighbors, file, file_exact, file_residual,
                      file_y, file_exact_y, file_residual_y, file_residual_total);

    // for mortar will be empty if mortar_flag is false
    std::vector<std::ofstream> file_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_mortar(n_faces_per_cell);

    std::vector<std::ofstream> file_y_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_y_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_y_mortar(n_faces_per_cell);

    //for plotting data storage
    std::vector<std::vector<double>> plot_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_y_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_y_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_y_mortar(n_faces_per_cell);

    name_files<dim>(this_mpi, 1, cycle, neighbors, file_mortar, file_exact_mortar, file_residual_mortar,
                    file_y_mortar, file_exact_y_mortar, file_residual_y_mortar, file_residual_total_mortar);

    solve_bar();
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        {
              interface_data_receive[side].resize(interface_dofs[side].size(),
                                                  0);
              interface_data_send[side].resize(interface_dofs[side].size(), 0);
              interface_data[side].resize(interface_dofs[side].size(), 0);
              interface_fe_function[side].reinit(solution_bar_stokes);
              interface_fe_function_fe[side].reinit(solution_bar_stokes);
        }
    if (print_interface_matrix_flag)
      print_interface_matrix(cycle); // important to keep it after solve_bar() and after initializing interface_fe_function


    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);


    // dealii::AffineConstraints<double>   constraints;
    FEFaceValues<dim> fe_face_values(fe,
                                    quad,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
    if (mortar_flag == 1)
    { 
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          interface_fe_function_mortar[side].reinit(solution_bar_mortar);
          interface_fe_function_mortar_fe[side].reinit(solution_bar_mortar);
        }
      project_mortar(P_fine2coarse,
                    dof_handler,
                    solution_bar_stokes,
                    quad,
                    constraints_mortar,
                    neighbors,
                    dof_handler_mortar,
                    solution_bar_mortar);
    }
    //GMRES structures and parameters
    std::vector<double>	sn;
    std::vector<double>	cs;
    std::vector<std::vector<double>>	H;
  //      std::vector<double> error_iter_side(n_faces_per_cell); //saves error in each iteration
    std::vector<double> e_all_iter; //error will be saved here after each iteration
    std::vector<double>	Beta; //beta for each side
    double combined_error_iter =0; //sum of error_iter_side

    std::vector<std::vector<double>> r(n_faces_per_cell); //to be deleted probably: p?
    std::vector<double> r_norm_side(n_faces_per_cell,0);
    std::vector<std::vector<std::vector<double>>>	Q_side(n_faces_per_cell) ;
    std::vector<std::vector<double>>  Ap(n_faces_per_cell);

    //defing q  to push_back to Q (reused in Arnoldi algorithm)
    std::vector<std::vector<double>> q(n_faces_per_cell);


    double l0 = 0.0;
    // CG with rhs being 0 and initial guess lambda = 0
    for (unsigned side = 0; side < n_faces_per_cell; ++side)

      if (neighbors[side] >= 0)
        {

          // Something will be here to initialize lambda correctly, right now it
          // is just zero
          Ap[side].resize(interface_dofs[side].size(), 0);
          lambda[side].resize(interface_dofs[side].size(), 0);

          q[side].resize(interface_dofs[side].size());
          r[side].resize(interface_dofs[side].size(), 0);
          std::vector<double> r_receive_buffer(r[side].size());

          // Right now it is effectively solution_bar - A\lambda (0)
          if (mortar_flag)
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] =  (get_normal_direction(side) * 
                              solution_bar_mortar[interface_dofs[side][i]]-
                            get_normal_direction(side) * l0);
            }
          }
          else
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] =  get_normal_direction(side) * 
                              solution_bar_stokes[interface_dofs[side][i]]-
                           get_normal_direction(side) * l0;
            }
          }

          MPI_Send(&r[side][0],
                  r[side].size(),
                  MPI_DOUBLE,
                  neighbors[side],
                  this_mpi,
                  mpi_communicator);
          MPI_Recv(&r_receive_buffer[0],
                  r_receive_buffer.size(),
                  MPI_DOUBLE,
                  neighbors[side],
                  neighbors[side],
                  mpi_communicator,
                  &mpi_status);

          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] += r_receive_buffer[i];
            }
          r_norm_side[side] = vect_norm(r[side]);
        }


    //Calculating r-norm(same as b-norm)-----
    double r_norm =0;
    for(unsigned int side=0; side<n_faces_per_cell;++side)
      if (neighbors[side] >= 0)
        r_norm+=r_norm_side[side]*r_norm_side[side];
    double r_norm_buffer =0;
    MPI_Allreduce(&r_norm,
        &r_norm_buffer,
      1,
      MPI_DOUBLE,
      MPI_SUM,
            mpi_communicator);
    r_norm = sqrt(r_norm_buffer);
    //end -----------of calclatig r-norm------------------

    //Making the first element of matrix Q[side] same as r_side[side/r_norm
    for(unsigned int side=0; side<n_faces_per_cell;++side)
          if (neighbors[side] >= 0){
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                            q[side][i]= r[side][i]/r_norm ;
            //adding q[side] as first element of Q[side]
            Q_side[side].push_back(q[side]);
          }



    //end----------- of caluclating first element of Q[side]-----------------
    e_all_iter.push_back(1.0);
    Beta.push_back(r_norm);






    std::vector<double> y;

    unsigned int k_counter = 0; //same as the count of the iteration
    while (k_counter < maxiter)
      {


        //////------solving the  star problem to find AQ(k)---------------------

        //Performing the Arnoldi algorithm
        //interface data will be given as Q_side[side][k_counter];
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            interface_data[side]=Q_side[side][k_counter];

        if (mortar_flag)
        {
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              interface_fe_function_mortar[side][interface_dofs[side][i]] = 
                interface_data[side][i];
          
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            {
              project_mortar(P_coarse2fine,
                            dof_handler_mortar,
                            interface_fe_function_mortar[side],
                            quad,
                            constraints_mortar,
                            neighbors,
                            dof_handler,
                            interface_fe_function[side]);
              interface_fe_function[side].block(1) = 0;//this is probably because of the projection
            }
        }
        else
        {
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              interface_fe_function[side][interface_dofs[side][i]] = 
                interface_data[side][i];
        }

        // interface_fe_function.block(1) = 0;
        assemble_rhs_star(fe_face_values);
        solve_star();
        gmres_iteration++;

        if (mortar_flag)
          project_mortar(P_fine2coarse,
                        dof_handler,
                        solution_star_stokes,
                        quad,
                        constraints_mortar,
                        neighbors,
                        dof_handler_mortar,
                        solution_star_mortar);

        //defing q  to push_back to Q (Arnoldi algorithm)
        //defing h  to push_back to H (Arnoldi algorithm)
        std::vector<double> h(k_counter+2,0);


        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            {
              // Create vector of u\dot n to send
              if (mortar_flag)
              {
                for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  interface_data_send[side][i] =
                    get_normal_direction(side) *
                    solution_star_mortar[interface_dofs[side][i]];
              }
              else
              {
                for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  interface_data_send[side][i] =
                    get_normal_direction(side) *
                    solution_star_stokes[interface_dofs[side][i]];
              }
              

              MPI_Send(&interface_data_send[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      this_mpi,
                      mpi_communicator);
              MPI_Recv(&interface_data_receive[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      neighbors[side],
                      mpi_communicator,
                      &mpi_status);

              // Compute Ap and with it compute alpha
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                {
                  Ap[side][i] = - (interface_data_send[side][i] +
                                  interface_data_receive[side][i]);


                }

              q[side].resize(Ap[side].size(),0);
              assert(Ap[side].size()==Q_side[side][k_counter].size());
              q[side] = Ap[side];
              for(unsigned int i=0; i<=k_counter; ++i)
              {
                        for(unsigned int j=0; j<q[side].size();++j){
                          h[i]+=q[side][j]*Q_side[side][i][j];
                          }

              }


            } //////-----------end of loop over side, q is calculated as AQ[][k] and ARnoldi Algorithm continued-------------------------------



        //Arnoldi Algorithm continued
              //combining summing h[i] over all subdomains
              std::vector<double> h_buffer(k_counter+2,0);

            MPI_Allreduce(&h[0],
                &h_buffer[0],
          k_counter+2,
          MPI_DOUBLE,
          MPI_SUM,
          mpi_communicator);

            h=h_buffer;
            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              if (neighbors[side] >= 0)
                for(unsigned int i=0; i<=k_counter; ++i)
                  for(unsigned int j=0; j<q[side].size();++j){
                    q[side][j]-=h[i]*Q_side[side][i][j];
                  }//end first loop for arnolod algorithm
            double h_dummy = 0;

            //calculating h(k+1)=norm(q) as summation over side,subdomains norm_squared(q[side])
            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                            if (neighbors[side] >= 0)
                              h_dummy+=vect_norm(q[side])*vect_norm(q[side]);
            double h_k_buffer=0;

            MPI_Allreduce(&h_dummy,
                            &h_k_buffer,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
            h[k_counter+1]=sqrt(h_k_buffer);
                  

            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              if (neighbors[side] >= 0){
                for(unsigned int i=0;i<q[side].size();++i)
                  q[side][i]/=h[k_counter+1];
              //Pushing back q[side] h to Q  as Q(k+1)
              Q_side[side].push_back(q[side]);
              }

              //Pushing back  h to H as H(k)}
            H.push_back(h);
        //---end of Arnoldi Algorithm

        //Eliminating the last element in H ith row and updating the rotation matrix.
        apply_givens_rotation(H[k_counter],cs,sn,
                    k_counter);
        //Updating the residual vector
        Beta.push_back(-sn[k_counter]*Beta[k_counter]);
        Beta[k_counter]*=cs[k_counter];

        //Combining error at kth iteration
        combined_error_iter=abs(Beta[k_counter+1])/r_norm;


        //saving the combined error at each iteration
        e_all_iter.push_back(combined_error_iter);
        
        //Calculating the result from H ,Q_side and Beta
        //Finding y which has size k_counter using back sove function
        y.resize(k_counter+1,0);
        assert(Beta.size()==k_counter+2); // gives error if exceed maxiter, do break if size is too much and going to exceed maxiter
        back_solve(H,Beta,y);

        // reset lambda to zero, x = 0, lambda = x + Q*y
        for (int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              lambda[side][i] = 0;

        // lambda = 0; // reset lambda to zero, x = 0, lambda = x + Q*y
  
        //updating X(lambda) to get the final lambda value before solving the final star problem
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            for (unsigned int i = 0; i < interface_data[side].size(); ++i)
              for(unsigned int j=0; j<=k_counter; ++j)
                lambda[side][i] += Q_side[side][j][i]*y[j];
        
        if (mortar_flag)
        {
          // for mortar grid
          plot_approx_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
          neighbors, lambda, plot_mortar, plot_y_mortar, file_mortar, file_y_mortar);
          if (cont_mortar_flag)
            plot_exact_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
              neighbors, exact_normal_stress_at_nodes_mortar, plot_exact_mortar, plot_exact_y_mortar, file_exact_mortar, file_exact_y_mortar);
          // plot_residual_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
          //   neighbors, r, plot_residual_mortar, plot_residual_y_mortar, file_residual_mortar, file_residual_y_mortar);  // residual plotting is not working yet 
          // for fe grid
          // first prepare lambda on fe grid
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            {
              lambda_fe[side].resize(interface_dofs_fe[side].size(), 0);
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                interface_fe_function_mortar_fe[side][interface_dofs[side][i]] = lambda[side][i];
              // project mortar to fe grid
              project_mortar(P_coarse2fine,
                            dof_handler_mortar,
                            interface_fe_function_mortar_fe[side],
                            quad,
                            constraints_mortar,
                            neighbors,
                            dof_handler,
                            interface_fe_function_fe[side]);
              for (unsigned int i = 0; i < interface_dofs_fe[side].size(); ++i)
                lambda_fe[side][i] = interface_fe_function_fe[side][interface_dofs_fe[side][i]];
            }
          // then plot it
          plot_approx_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
            neighbors, lambda_fe, plot, plot_y, file, file_y);
          plot_exact_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
            neighbors, exact_normal_stress_at_nodes_fe, plot_exact, plot_exact_y, file_exact, file_exact_y);
          // plot_residual_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
          //   neighbors, r, plot_residual, plot_residual_y, file_residual, file_residual_y);
        }
        else
        {
          plot_approx_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
            neighbors, lambda, plot, plot_y, file, file_y);
          plot_exact_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
            neighbors, exact_normal_stress_at_nodes_fe, plot_exact, plot_exact_y, file_exact, file_exact_y);
          // plot_residual_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
          //   neighbors, r, plot_residual, plot_residual_y, file_residual, file_residual_y);
        }

        residual = combined_error_iter;

        pcout << "\r  ..." << gmres_iteration
              << " iterations completed, (residual = " << combined_error_iter
              << ")..." << std::flush;
        // Exit criterion
        if (combined_error_iter < tolerance)
          {
            pcout << "\n  GMRES converges in " << gmres_iteration << " iterations!\n";
            break;
          }
        else if(k_counter>maxiter-2)
          pcout << "\n  GMRES doesn't converge after  " << k_counter << " iterations!\n";
        else if(k_counter == interface_dofs_size-1)
          {
            pcout << "\n  GMRES doesn't converge after " << interface_dofs_size << " iterations!\n";
            break;
          }


        
        //maxing interface_data_receive and send zero so it can be used is solving for Ap(or A*Q([k_counter]).
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          {
            interface_data_receive[side].resize(interface_dofs[side].size(), 0);
            interface_data_send[side].resize(interface_dofs[side].size(), 0);
          }

        Ap.resize(n_faces_per_cell);
        k_counter++;
      }//end of the while loop(k_counter<max iteration)
    //plot the total residual for mortar and non-mortar cases
    if (mortar_flag)
      plot_total_residual<dim>(e_all_iter, file_residual_total_mortar);
    else
      plot_total_residual<dim>(e_all_iter, file_residual_total);
    for (unsigned int side=0; side < n_faces_per_cell; ++side)
    {
      file[side].close(); // close the file
      file_residual[side].close(); // has the total residual
      file_exact[side].close();
      file_y[side].close(); // close the file
      file_residual_y[side].close(); // useless empty file
      file_exact_y[side].close();
      //for mortar will be empty if mortar_flag is false
      file_mortar[side].close(); // close the file
      file_residual_mortar[side].close();
      file_exact_mortar[side].close(); 
      file_y_mortar[side].close(); // close the file
      file_residual_y_mortar[side].close(); // useless empty file
      file_exact_y_mortar[side].close();
      file_residual_total_mortar.close();
      file_residual_total.close();
    }

    //we can replace lambda here and just add interface_data(skip one step below)
    if (mortar_flag)
    {
      interface_data = lambda;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          interface_fe_function_mortar[side][interface_dofs[side][i]] = 
            interface_data[side][i];
      
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        { 
          project_mortar(P_coarse2fine,
                        dof_handler_mortar,
                        interface_fe_function_mortar[side],
                        quad,
                        constraints_mortar,
                        neighbors,
                        dof_handler,
                        interface_fe_function[side]);

          interface_fe_function[side].block(1) = 0; 
        }
    }
    else
    {
      interface_data = lambda;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function[side][interface_dofs[side][i]] = interface_data[side][i];
    }

    assemble_rhs_star(fe_face_values);
    solve_star();
    solution.reinit(solution_bar_stokes);
    solution = solution_bar_stokes;
    solution.sadd(1.0, solution_star_stokes);
    pcout<<"finished local_gmres"<<"\n";
  }


  template <int dim>
  void
  MixedStokesProblemDD<dim>::local_cg(const unsigned int &maxiter, unsigned int &cycle)
  {
    TimerOutput::Scope t(computing_timer, "Local CG");

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    // const unsigned int n_processes =
    //   Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    

    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    interface_fe_function.resize(n_faces_per_cell);
    interface_fe_function_fe.resize(n_faces_per_cell);
    interface_fe_function_mortar.resize(n_faces_per_cell);
    interface_fe_function_mortar_fe.resize(n_faces_per_cell);
    std::vector<double> residual_vector;

    std::vector<std::vector<double>> lambda(n_faces_per_cell);
    std::vector<std::vector<double>> lambda_fe(n_faces_per_cell);

    std::vector<std::ofstream> file(n_faces_per_cell);
    std::vector<std::ofstream> file_exact(n_faces_per_cell);
    std::vector<std::ofstream> file_residual(n_faces_per_cell);

    std::vector<std::ofstream> file_y(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_y(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_y(n_faces_per_cell);

    std::ofstream file_residual_total;
    std::ofstream file_residual_total_mortar;

    //for plotting data storage
    std::vector<std::vector<double>> plot(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact(n_faces_per_cell);
    std::vector<std::vector<double>> plot_y(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_y(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_y(n_faces_per_cell);

    name_files<dim>(this_mpi, 0, cycle, neighbors, file, file_exact, file_residual,
                      file_y, file_exact_y, file_residual_y, file_residual_total);
                      
    // for mortar will be empty if mortar_flag is false
    std::vector<std::ofstream> file_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_mortar(n_faces_per_cell);

    std::vector<std::ofstream> file_y_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_exact_y_mortar(n_faces_per_cell);
    std::vector<std::ofstream> file_residual_y_mortar(n_faces_per_cell);

    //for plotting data storage
    std::vector<std::vector<double>> plot_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_y_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_residual_y_mortar(n_faces_per_cell);
    std::vector<std::vector<double>> plot_exact_y_mortar(n_faces_per_cell);

    name_files<dim>(this_mpi, 1, cycle, neighbors, file_mortar, file_exact_mortar, file_residual_mortar,
                    file_y_mortar, file_exact_y_mortar, file_residual_y_mortar, file_residual_total_mortar);


    solve_bar();
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        {
              interface_data_receive[side].resize(interface_dofs[side].size(),
                                                  0);
              interface_data_send[side].resize(interface_dofs[side].size(), 0);
              interface_data[side].resize(interface_dofs[side].size(), 0);
              interface_fe_function[side].reinit(solution_bar_stokes);
              interface_fe_function_fe[side].reinit(solution_bar_stokes);
              // interface_fe_function_mortar[side].reinit(solution_bar_mortar);
        }
    if (print_interface_matrix_flag)
      print_interface_matrix(cycle); // important to keep it after solve_bar() and initializing interface_fe_function


    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);
    pcout << "qdegree = " << qdegree << std::endl;
    

    // dealii::AffineConstraints<double>   constraints;
    FEFaceValues<dim> fe_face_values(fe,
                                     quad,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    if (mortar_flag == 1)
    { 
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        {
          interface_fe_function_mortar[side].reinit(solution_bar_mortar);
          interface_fe_function_mortar_fe[side].reinit(solution_bar_mortar);
        }
      
      project_mortar(P_fine2coarse,
                    dof_handler,
                    solution_bar_stokes,
                    quad,
                    constraints_mortar,
                    neighbors,
                    dof_handler_mortar,
                    solution_bar_mortar);
    }

    // CG structures and parameters
    std::vector<double> alpha_side(n_faces_per_cell, 0),
      alpha_side_d(n_faces_per_cell, 0), beta_side(n_faces_per_cell, 0),
      beta_side_d(n_faces_per_cell, 0);
    std::vector<double> alpha(2, 0), beta(2, 0);

    std::vector<std::vector<double>> r(n_faces_per_cell), p(n_faces_per_cell);
    std::vector<std::vector<double>> Ap(n_faces_per_cell);

    double l0 = 0.0;
    // CG with rhs being u_bar and initial guess lambda = 0
    for (unsigned side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        {
          // Something will be here to initialize lambda correctly, right now it
          // is just zero
          Ap[side].resize(interface_dofs[side].size(), 0);
          lambda[side].resize(interface_dofs[side].size(), 0);
          // lambda_fe[side].resize(interface_dofs_fe[side].size(), 0);

          r[side].resize(interface_dofs[side].size(), 0);
          std::vector<double> r_receive_buffer(r[side].size());

          // Right now it is effectively solution_bar - A\lambda (0)
          if (mortar_flag)
            {
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              {
                r[side][i] = get_normal_direction(side) * 
                                solution_bar_mortar[interface_dofs[side][i]]-
                             get_normal_direction(side) * l0;
              }
            }
          else
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] = get_normal_direction(side) * 
                              solution_bar_stokes[interface_dofs[side][i]]-
                           get_normal_direction(side) * l0;
            }
          }

          MPI_Send(&r[side][0],// start sending starting from this element
                   r[side].size(),// send these many of them
                   MPI_DOUBLE,// type of elements being sent
                   neighbors[side],// being sent to this processor
                   this_mpi,// tag of message
                   mpi_communicator);// the communication world/mail office
          MPI_Recv(&r_receive_buffer[0],// start receiving/loading the input from here
                   r_receive_buffer.size(),// these many inputs will be received
                   MPI_DOUBLE,// type of inputs being received
                   neighbors[side],// receiving message from this processor
                   neighbors[side],// tag associated to the message
                   mpi_communicator,// the communication world/mail office
                   &mpi_status);

          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] += r_receive_buffer[i];
            }
        }
                  
    p = r;

    double normB    = 0;
    double normRold = 0;

    // if (this_mpi == 0)
    //   for (int side = 0; side < n_faces_per_cell; ++side)
    //     if (neighbors[side] >= 0)
    //     {
    //       std::cout << "\ninterface_dofs[side].size() = " << interface_dofs[side].size() << std::endl;
    //       for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
    //         std::cout << "\ninterface_dofs[" << side << "][" << i << "] = " << interface_dofs[side][i] << std::endl;       
    //     }

    unsigned int iteration_counter = 0;
    while (iteration_counter < maxiter)
    {
      alpha[0] = 0.0;
      alpha[1] = 0.0;
      beta[0]  = 0.0;
      beta[1]  = 0.0;

      iteration_counter++;
      interface_data = p;

      if (mortar_flag)
      {
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function_mortar[side][interface_dofs[side][i]] = 
              interface_data[side][i];
        
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
          {
            project_mortar(P_coarse2fine,
                          dof_handler_mortar,
                          interface_fe_function_mortar[side],
                          quad,
                          constraints_mortar,
                          neighbors,
                          dof_handler,
                          interface_fe_function[side]);
            interface_fe_function[side].block(1) = 0;//this is probably because of the projection
            // pcout << "did we get here?? 1" << std::endl;
          }
      }
      else
      {
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function[side][interface_dofs[side][i]] = 
              interface_data[side][i];
      }
      assemble_rhs_star(fe_face_values);
      solve_star();
      gmres_iteration++;

      //Interface data for plotting, currently works only in 2 dim
      if (mortar_flag)
      {
        // for mortar grid
        plot_approx_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
        neighbors, lambda, plot_mortar, plot_y_mortar, file_mortar, file_y_mortar);
        if (cont_mortar_flag)
          plot_exact_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
            neighbors, exact_normal_stress_at_nodes_mortar, plot_exact_mortar, 
            plot_exact_y_mortar, file_exact_mortar, file_exact_y_mortar);
        plot_residual_function<dim>(this_mpi, 1, mortar_degree, interface_dofs, 
          neighbors, r, plot_residual_mortar, plot_residual_y_mortar, file_residual_mortar, file_residual_y_mortar);  
        // for fe grid
        // first prepare lambda on fe grid
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
          {
            lambda_fe[side].resize(interface_dofs_fe[side].size(), 0);
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              interface_fe_function_mortar_fe[side][interface_dofs[side][i]] = lambda[side][i];
            // project mortar to fe grid
            project_mortar(P_coarse2fine,
                          dof_handler_mortar,
                          interface_fe_function_mortar_fe[side],
                          quad,
                          constraints_mortar,
                          neighbors,
                          dof_handler,
                          interface_fe_function_fe[side]);
            for (unsigned int i = 0; i < interface_dofs_fe[side].size(); ++i)
              lambda_fe[side][i] = interface_fe_function_fe[side][interface_dofs_fe[side][i]];
          }
        // then plot it
        plot_approx_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
          neighbors, lambda_fe, plot, plot_y, file, file_y);
        plot_exact_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
          neighbors, exact_normal_stress_at_nodes_fe, plot_exact, plot_exact_y, file_exact, file_exact_y);
        plot_residual_function<dim>(this_mpi, 0, mortar_degree, interface_dofs_fe, 
          neighbors, r, plot_residual, plot_residual_y, file_residual, file_residual_y);
      }
      else
      {
        plot_approx_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
          neighbors, lambda, plot, plot_y, file, file_y);
        plot_exact_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
          neighbors, exact_normal_stress_at_nodes_fe, plot_exact, plot_exact_y, file_exact, file_exact_y);
        plot_residual_function<dim>(this_mpi, 0, mortar_degree, interface_dofs, 
          neighbors, r, plot_residual, plot_residual_y, file_residual, file_residual_y);
      }

      if (mortar_flag)
        project_mortar(P_fine2coarse,
                      dof_handler,
                      solution_star_stokes,
                      quad,
                      constraints_mortar,
                      neighbors,
                      dof_handler_mortar,
                      solution_star_mortar);

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          {
            alpha_side[side]   = 0;
            alpha_side_d[side] = 0;
            beta_side[side]    = 0;
            beta_side_d[side]  = 0;

            // Create vector of u\dot n to send
            if (mortar_flag)
            {
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              interface_data_send[side][i] = get_normal_direction(side) *
                                                solution_star_mortar[interface_dofs[side][i]];
            }
            else
            {
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              interface_data_send[side][i] = get_normal_direction(side) *
                                                solution_star_stokes[interface_dofs[side][i]];
            }


            // constant_Ap_two<dim>(interface_dofs, neighbors, 
            //         repeated_dofs, repeated_dofs_neumann, interface_data_send);

            // average_Ap_two<dim>(interface_dofs, neighbors, 
            //         repeated_dofs, repeated_dofs_neumann, interface_data_send);
              
            // average_Ap_three<dim>(interface_dofs, neighbors, 
            //         repeated_dofs, repeated_dofs_neumann, interface_data_send);
                    
            MPI_Send(&interface_data_send[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      this_mpi,
                      mpi_communicator);
            MPI_Recv(&interface_data_receive[side][0],
                      interface_dofs[side].size(),
                      MPI_DOUBLE,
                      neighbors[side],
                      neighbors[side],
                      mpi_communicator,
                      &mpi_status);

            // Compute Ap and with it compute alpha
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                Ap[side][i] = - (interface_data_send[side][i] +
                                interface_data_receive[side][i]);

                alpha_side[side] += r[side][i] * r[side][i];
                alpha_side_d[side] += p[side][i] * Ap[side][i];
              }
          }
        

      // Fancy some lambdas, huh?
      std::for_each(alpha_side.begin(), alpha_side.end(), [&](double n) {
        alpha[0] += n;
      });
      std::for_each(alpha_side_d.begin(), alpha_side_d.end(), [&](double n) {
        alpha[1] += n;
      });
      std::vector<double> alpha_buffer(2, 0);

      MPI_Allreduce(&alpha[0], //sending this data
                    &alpha_buffer[0], //receiving the result here
                    2, //number of elements in alpha and alpha_buffer = 1+1
                    MPI_DOUBLE, //type of each element
                    MPI_SUM, //adding all elements received
                    mpi_communicator);


      alpha = alpha_buffer;

      if (gmres_iteration == 1)
        normB = alpha[0];

      normRold = alpha[0];

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          {
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              {
                lambda[side][i] += (alpha[0] * p[side][i]) / alpha[1];
                r[side][i] -= (alpha[0] * Ap[side][i]) / alpha[1];
              }

            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              beta_side[side] += r[side][i] * r[side][i];
          }

      // if (cycle==0){
      //   residual = fabs(alpha[0] / normB);
      //   pcout << gmres_iteration
      //         << " iterations completed, (residual = " << fabs(alpha[0] / normB)
      //         << ")" << std::endl;
      //   // break;
      // }

      if (cycle>=0)
      {
        pcout << "\r  ..." << gmres_iteration
              << " iterations completed, (residual = " << fabs(alpha[0] / normB)
              << ")..." << std::flush;
        residual = fabs(alpha[0] / normB);
      }
      residual_vector.push_back(residual);
      // Exit criterion
      if (fabs(alpha[0]) / normB < tolerance)
        {
          residual = fabs(alpha[0] / normB);
          pcout << "\n  CG converges in " << gmres_iteration << " iterations!\n";
          break;
        }
      
      if (std::isnan(fabs(alpha[0]) / normB))
      {
        residual = fabs(alpha[0] / normB);
        pcout << "\n  residual is not a number! at " << gmres_iteration << " iteration\n";
        break;
      }
      
      if ((fabs(alpha[0]) / normB >= tolerance) && gmres_iteration == maxiter)
      {
        residual = fabs(alpha[0] / normB);
        pcout << "\n  CG exceeds maxiter = " << gmres_iteration << " iterations!\n";
        break;
      }

      std::for_each(beta_side.begin(), beta_side.end(), [&](double n) {
        beta[0] += n;
      });
      double beta_buffer = 0;

      MPI_Allreduce(
        &beta[0], &beta_buffer, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      beta[0] = beta_buffer;
      beta[1] = normRold;

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        {
          if (neighbors[side] >= 0)
            for (unsigned int i = 0; i < interface_data[side].size(); ++i)
              p[side][i] = r[side][i] + (beta[0] / beta[1]) * p[side][i];

          interface_data_receive[side].resize(interface_dofs[side].size(), 0);
          interface_data_send[side].resize(interface_dofs[side].size(), 0);

          Ap.resize(n_faces_per_cell);
        }
    }
    if (mortar_flag)
      plot_total_residual<dim>(residual_vector, file_residual_total_mortar);
    else
      plot_total_residual<dim>(residual_vector, file_residual_total);
    for (unsigned int side=0; side < n_faces_per_cell; ++side)
    {
      file[side].close(); // close the file
      file_residual[side].close();
      file_exact[side].close();
      file_y[side].close(); // close the file
      file_residual_y[side].close();
      file_exact_y[side].close();
      //for mortar will be empty if mortar_flag is false
      file_mortar[side].close(); // close the file
      file_residual_mortar[side].close();
      file_exact_mortar[side].close();
      file_y_mortar[side].close(); // close the file
      file_residual_y_mortar[side].close();
      file_exact_y_mortar[side].close();
      file_residual_total.close();
      file_residual_total_mortar.close();
    }

    if (mortar_flag)
    {
      interface_data = lambda;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
          interface_fe_function_mortar[side][interface_dofs[side][i]] = 
            interface_data[side][i];
      
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
        { 
          project_mortar(P_coarse2fine,
                        dof_handler_mortar,
                        interface_fe_function_mortar[side],
                        quad,
                        constraints_mortar,
                        neighbors,
                        dof_handler,
                        interface_fe_function[side]);

          interface_fe_function[side].block(1) = 0; 
        }
    }
    else
    {
      interface_data = lambda;
      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function[side][interface_dofs[side][i]] = interface_data[side][i];
    }

    assemble_rhs_star(fe_face_values);
    solve_star();
    solution.reinit(solution_bar_stokes);
    solution = solution_bar_stokes;// u = u_bar
    solution.sadd(1.0, solution_star_stokes);// u = 1*u+u_star
  }



  // MixedStokesProblemDD::compute_interface_error
  // template <int dim>
  // double
  // MixedStokesProblemDD<dim>::compute_interface_error(
  //   Function<dim> &exact_solution)
  // {
  //   system_rhs_star_stokes = 0;

  //   QGauss<dim - 1>   quad(qdegree);
  //   QGauss<dim - 1>   project_quad(qdegree);
  //   FEFaceValues<dim> fe_face_values(fe,
  //                                    quad,
  //                                    update_values | update_normal_vectors |
  //                                      update_quadrature_points |
  //                                      update_JxW_values);

  //   const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
  //   const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  //   // const unsigned int dofs_per_cell_mortar = fe_mortar.dofs_per_cell;

  //   Vector<double>                       local_rhs(dofs_per_cell);
  //   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  //   std::vector<FEValuesExtractors::Vector> stresses(
  //     dim, FEValuesExtractors::Vector());
  //   for (unsigned int d = 0; d < dim; ++d)
  //     {
  //       const FEValuesExtractors::Vector tmp_stress(d * dim);
  //       stresses[d].first_vector_component = tmp_stress.first_vector_component;
  //     }

  //   std::vector<std::vector<Tensor<1, dim>>> interface_values(
  //     dim, std::vector<Tensor<1, dim>>(n_face_q_points));
  //   std::vector<std::vector<Tensor<1, dim>>> solution_values(
  //     dim, std::vector<Tensor<1, dim>>(n_face_q_points));
  //   std::vector<Vector<double>> displacement_values(n_face_q_points,
  //                                                   Vector<double>(dim));

  //   // Assemble rhs for star problem with data = u - lambda_H on interfaces
  //   typename DoFHandler<dim>::active_cell_iterator cell =
  //                                                    dof_handler.begin_active(),
  //                                                  endc = dof_handler.end();
  //   for (; cell != endc; ++cell)
  //     {
  //       local_rhs = 0;
  //       cell->get_dof_indices(local_dof_indices);

  //       for (unsigned int face_n = 0;
  //            face_n < GeometryInfo<dim>::faces_per_cell;
  //            ++face_n)
  //         if (cell->at_boundary(face_n) &&
  //             cell->face(face_n)->boundary_id() != 0)
  //           {
  //             fe_face_values.reinit(cell, face_n);

  //             for (unsigned int d_i = 0; d_i < dim; ++d_i)
  //               fe_face_values[stresses[d_i]].get_function_values(
  //                 interface_fe_function, interface_values[d_i]);

  //             exact_solution.vector_value_list(
  //               fe_face_values.get_quadrature_points(), displacement_values);

  //             for (unsigned int q = 0; q < n_face_q_points; ++q)
  //               for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //                 {
  //                   Tensor<2, dim> sigma;
  //                   Tensor<2, dim> interface_lambda;
  //                   for (unsigned int d_i = 0; d_i < dim; ++d_i)
  //                     fe_face_values[stresses[d_i]].get_function_values(
  //                       interface_fe_function, interface_values[d_i]);

  //                   Tensor<1, dim> sigma_n =
  //                     sigma * fe_face_values.normal_vector(q);
  //                   for (unsigned int d_i = 0; d_i < dim; ++d_i)
  //                     local_rhs(i) +=
  //                       fe_face_values[stresses[d_i]].value(i, q) *
  //                       fe_face_values.normal_vector(q) *
  //                       (displacement_values[q][d_i] -
  //                        interface_values[d_i][q] *
  //                          get_normal_direction(
  //                            cell->face(face_n)->boundary_id() - 1) *
  //                          fe_face_values.normal_vector(q)) *
  //                       fe_face_values.JxW(q);
  //                 }
  //           }

  //       for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //         system_rhs_star_stokes(local_dof_indices[i]) += local_rhs(i);
  //     }

  //   // Solve star problem with data given by p - lambda_h
  //   solve_star();

  //   double res = 0;

  //   FEFaceValues<dim> fe_face_values_mortar(fe_mortar,
  //                                           quad,
  //                                           update_values |
  //                                             update_normal_vectors |
  //                                             update_quadrature_points |
  //                                             update_JxW_values);

  //   // Compute the discrete interface norm
  //   cell = dof_handler_mortar.begin_active(), endc = dof_handler_mortar.end();
  //   for (; cell != endc; ++cell)
  //     {
  //       for (unsigned int face_n = 0;
  //            face_n < GeometryInfo<dim>::faces_per_cell;
  //            ++face_n)
  //         if (cell->at_boundary(face_n) &&
  //             cell->face(face_n)->boundary_id() != 0)
  //           {
  //             fe_face_values_mortar.reinit(cell, face_n);

  //             for (unsigned int d_i = 0; d_i < dim; ++d_i)
  //               {
  //                 fe_face_values_mortar[stresses[d_i]].get_function_values(
  //                   solution_star_mortar, solution_values[d_i]);
  //                 fe_face_values_mortar[stresses[d_i]].get_function_values(
  //                   interface_fe_function_mortar, interface_values[d_i]);
  //               }

  //             exact_solution.vector_value_list(
  //               fe_face_values_mortar.get_quadrature_points(),
  //               displacement_values);

  //             for (unsigned int q = 0; q < n_face_q_points; ++q)
  //               for (unsigned int d_i = 0; d_i < dim; ++d_i)
  //                 res += fabs(fe_face_values_mortar.normal_vector(q) *
  //                             solution_values[d_i][q] *
  //                             (displacement_values[q][d_i] -
  //                              fe_face_values_mortar.normal_vector(q) *
  //                                interface_values[d_i][q] *
  //                                get_normal_direction(
  //                                  cell->face(face_n)->boundary_id() - 1)) *
  //                             fe_face_values_mortar.JxW(q));
  //           }
  //     }

  //   return sqrt(res);
  // }


  // // MixedStokesProblemDD::compute_errors
  // template <int dim>
  // void
  // MixedStokesProblemDD<dim>::compute_errors(const unsigned int &cycle)
  // {
  //   TimerOutput::Scope t(computing_timer, "Compute Errors");

  //   const ComponentSelectFunction<dim> rotation_mask(dim * dim + dim,
  //                                                    dim * dim + dim +
  //                                                      0.5 * dim * (dim - 1));
  //   const ComponentSelectFunction<dim> displacement_mask(
  //     std::make_pair(dim * dim, dim * dim + dim),
  //     dim * dim + dim + 0.5 * dim * (dim - 1));
  //   const ComponentSelectFunction<dim> stress_mask(std::make_pair(0, dim * dim),
  //                                                  dim * dim + dim +
  //                                                    0.5 * dim * (dim - 1));
  //   ExactSolution<dim>                 exact_solution;

  //   // Vectors to temporarily store cellwise errros
  //   Vector<double> cellwise_errors(triangulation.n_active_cells());
  //   Vector<double> cellwise_norms(triangulation.n_active_cells());

  //   // Vectors to temporarily store cellwise componentwise div errors
  //   Vector<double> cellwise_div_errors(triangulation.n_active_cells());
  //   Vector<double> cellwise_div_norms(triangulation.n_active_cells());

  //   // Define quadrature points to compute errors at
  //   QGauss<dim> quadrature(degree + 5);

  //   // This is used to show superconvergence at midcells
  //   QGauss<dim> quadrature_super(1);

  //   // Since we want to compute the relative norm
  //   BlockVector<double> zerozeros(1, solution_star_stokes.size());

  //   // Rotation error and norm
  //   VectorTools::integrate_difference(dof_handler,
  //                                     solution,
  //                                     exact_solution,
  //                                     cellwise_errors,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &rotation_mask);
  //   const double p_l2_error = cellwise_errors.l2_norm();

  //   VectorTools::integrate_difference(dof_handler,
  //                                     zerozeros,
  //                                     exact_solution,
  //                                     cellwise_norms,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &rotation_mask);
  //   const double p_l2_norm = cellwise_norms.l2_norm();

  //   // Displacement error and norm
  //   VectorTools::integrate_difference(dof_handler,
  //                                     solution,
  //                                     exact_solution,
  //                                     cellwise_errors,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &displacement_mask);
  //   const double u_l2_error = cellwise_errors.l2_norm();

  //   VectorTools::integrate_difference(dof_handler,
  //                                     zerozeros,
  //                                     exact_solution,
  //                                     cellwise_norms,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &displacement_mask);
  //   const double u_l2_norm = cellwise_norms.l2_norm();

  //   // Displacement error and norm at midcells
  //   VectorTools::integrate_difference(dof_handler,
  //                                     solution,
  //                                     exact_solution,
  //                                     cellwise_errors,
  //                                     quadrature_super,
  //                                     VectorTools::L2_norm,
  //                                     &displacement_mask);
  //   const double u_l2_mid_error = cellwise_errors.l2_norm();

  //   VectorTools::integrate_difference(dof_handler,
  //                                     zerozeros,
  //                                     exact_solution,
  //                                     cellwise_norms,
  //                                     quadrature_super,
  //                                     VectorTools::L2_norm,
  //                                     &displacement_mask);
  //   const double u_l2_mid_norm = cellwise_norms.l2_norm();

  //   // Stress L2 error and norm
  //   VectorTools::integrate_difference(dof_handler,
  //                                     solution,
  //                                     exact_solution,
  //                                     cellwise_errors,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &stress_mask);
  //   const double s_l2_error = cellwise_errors.l2_norm();

  //   VectorTools::integrate_difference(dof_handler,
  //                                     zerozeros,
  //                                     exact_solution,
  //                                     cellwise_norms,
  //                                     quadrature,
  //                                     VectorTools::L2_norm,
  //                                     &stress_mask);

  //   const double s_l2_norm = cellwise_norms.l2_norm();

  //   // Stress Hdiv seminorm
  //   cellwise_errors = 0;
  //   cellwise_norms  = 0;
  //   for (int i = 0; i < dim; ++i)
  //     {
  //       const ComponentSelectFunction<dim> stress_component_mask(
  //         std::make_pair(i * dim, (i + 1) * dim),
  //         dim * dim + dim + 0.5 * dim * (dim - 1));

  //       VectorTools::integrate_difference(dof_handler,
  //                                         solution,
  //                                         exact_solution,
  //                                         cellwise_div_errors,
  //                                         quadrature,
  //                                         VectorTools::Hdiv_seminorm,
  //                                         &stress_component_mask);
  //       cellwise_errors += cellwise_div_errors;

  //       VectorTools::integrate_difference(dof_handler,
  //                                         zerozeros,
  //                                         exact_solution,
  //                                         cellwise_div_norms,
  //                                         quadrature,
  //                                         VectorTools::Hdiv_seminorm,
  //                                         &stress_component_mask);
  //       cellwise_norms += cellwise_div_norms;
  //     }

  //   const double s_hd_error = cellwise_errors.l2_norm();
  //   const double s_hd_norm  = cellwise_norms.l2_norm();

  //   double l_int_error = 1, l_int_norm = 1;

  //   if (mortar_flag)
  //     {
  //       DisplacementBoundaryValues<dim> displ_solution;
  //       l_int_error = compute_interface_error(displ_solution);

  //       interface_fe_function        = 0;
  //       interface_fe_function_mortar = 0;
  //       l_int_norm                   = compute_interface_error(displ_solution);
  //     }

  //   double send_buf_num[6] = {s_l2_error,
  //                             s_hd_error,
  //                             u_l2_error,
  //                             u_l2_mid_error,
  //                             p_l2_error,
  //                             l_int_error};
  //   double send_buf_den[6] = {
  //     s_l2_norm, s_hd_norm, u_l2_norm, u_l2_mid_norm, p_l2_norm, l_int_norm};

  //   double recv_buf_num[6] = {0, 0, 0, 0, 0, 0};
  //   double recv_buf_den[6] = {0, 0, 0, 0, 0, 0};

  //   MPI_Reduce(&send_buf_num[0],
  //              &recv_buf_num[0],
  //              6,
  //              MPI_DOUBLE,
  //              MPI_SUM,
  //              0,
  //              mpi_communicator);
  //   MPI_Reduce(&send_buf_den[0],
  //              &recv_buf_den[0],
  //              6,
  //              MPI_DOUBLE,
  //              MPI_SUM,
  //              0,
  //              mpi_communicator);

  //   for (unsigned int i = 0; i < 6; ++i)
  //     recv_buf_num[i] = recv_buf_num[i] / recv_buf_den[i];

  //   if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  //     {
  //       convergence_table.add_value("cycle", cycle);
  //       convergence_table.add_value("# CG", gmres_iteration);
  //       convergence_table.add_value("Stress,L2", recv_buf_num[0]);
  //       convergence_table.add_value("Stress,Hdiv", recv_buf_num[1]);
  //       convergence_table.add_value("Displ,L2", recv_buf_num[2]);
  //       convergence_table.add_value("Displ,L2mid", recv_buf_num[3]);
  //       convergence_table.add_value("Rotat,L2", recv_buf_num[4]);

  //       if (mortar_flag)
  //         convergence_table.add_value("Lambda,Int", recv_buf_num[6]);
  //     }
  // }


  // // MixedStokesProblemDD::output_results
  // template <int dim>
  // void
  // MixedStokesProblemDD<dim>::output_results(const unsigned int &cycle,
  //                                           const unsigned int &refine,
  //                                           const std::string & name)
  // {
  //   TimerOutput::Scope t(computing_timer, "Output results");
  //   unsigned int       n_processes =
  //     Utilities::MPI::n_mpi_processes(mpi_communicator);
  //   unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);


  //   std::vector<std::string> solution_names;
  //   std::string              rhs_name = "rhs";

  //   switch (dim)
  //     {
  //       case 2:
  //         solution_names.push_back("s11");
  //         solution_names.push_back("s12");
  //         solution_names.push_back("s21");
  //         solution_names.push_back("s22");
  //         solution_names.push_back("u");
  //         solution_names.push_back("v");
  //         solution_names.push_back("p");
  //         break;

  //       case 3:
  //         solution_names.push_back("s11");
  //         solution_names.push_back("s12");
  //         solution_names.push_back("s13");
  //         solution_names.push_back("s21");
  //         solution_names.push_back("s22");
  //         solution_names.push_back("s23");
  //         solution_names.push_back("s31");
  //         solution_names.push_back("s32");
  //         solution_names.push_back("s33");
  //         solution_names.push_back("u");
  //         solution_names.push_back("v");
  //         solution_names.push_back("w");
  //         solution_names.push_back("p1");
  //         solution_names.push_back("p2");
  //         solution_names.push_back("p3");
  //         break;

  //       default:
  //         Assert(false, ExcNotImplemented());
  //     }

  //   std::vector<DataComponentInterpretation::DataComponentInterpretation>
  //     data_component_interpretation(
  //       dim * dim + dim + 0.5 * dim * (dim - 1) - 1,
  //       DataComponentInterpretation::component_is_part_of_vector);

  //   switch (dim)
  //     {
  //       case 2:
  //         data_component_interpretation.push_back(
  //           DataComponentInterpretation::component_is_scalar);
  //         break;

  //       case 3:
  //         data_component_interpretation.push_back(
  //           DataComponentInterpretation::component_is_part_of_vector);
  //         break;

  //       default:
  //         Assert(false, ExcNotImplemented());
  //         break;
  //     }

  //   DataOut<dim> data_out_star;
  //   data_out_star.add_data_vector(dof_handler,
  //                                 solution,
  //                                 solution_names,
  //                                 data_component_interpretation);
  //   data_out_star.build_patches(degree);
  //   std::ofstream output("solution" + name + "_p" +
  //                        Utilities::to_string(this_mpi) + "-" +
  //                        Utilities::to_string(cycle) + ".vtu");
  //   data_out_star.write_vtu(output);


  //   if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  //     {
  //       convergence_table.set_precision("Stress,L2", 3);
  //       convergence_table.set_precision("Stress,Hdiv", 3);
  //       convergence_table.set_precision("Displ,L2", 3);
  //       convergence_table.set_precision("Displ,L2mid", 3);
  //       convergence_table.set_precision("Rotat,L2", 3);

  //       convergence_table.set_scientific("Stress,L2", true);
  //       convergence_table.set_scientific("Stress,Hdiv", true);
  //       convergence_table.set_scientific("Displ,L2", true);
  //       convergence_table.set_scientific("Displ,L2mid", true);
  //       convergence_table.set_scientific("Rotat,L2", true);

  //       convergence_table.set_tex_caption("# CG", "\\# cg");
  //       convergence_table.set_tex_caption(
  //         "Stress,L2", "$ \\|\\sigma - \\sigma_h\\|_{L^2} $");
  //       convergence_table.set_tex_caption(
  //         "Stress,Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^2} $");
  //       convergence_table.set_tex_caption("Displ,L2",
  //                                         "$ \\|u - u_h\\|_{L^2} $");
  //       convergence_table.set_tex_caption("Displ,L2mid",
  //                                         "$ \\|Qu - u_h\\|_{L^2} $");
  //       convergence_table.set_tex_caption("Rotat,L2",
  //                                         "$ \\|p - p_h\\|_{L^2} $");

  //       convergence_table.evaluate_convergence_rates(
  //         "# CG", ConvergenceTable::reduction_rate_log2);
  //       convergence_table.evaluate_convergence_rates(
  //         "Stress,L2", ConvergenceTable::reduction_rate_log2);
  //       convergence_table.evaluate_convergence_rates(
  //         "Stress,Hdiv", ConvergenceTable::reduction_rate_log2);
  //       convergence_table.evaluate_convergence_rates(
  //         "Displ,L2", ConvergenceTable::reduction_rate_log2);
  //       convergence_table.evaluate_convergence_rates(
  //         "Displ,L2mid", ConvergenceTable::reduction_rate_log2);
  //       convergence_table.evaluate_convergence_rates(
  //         "Rotat,L2", ConvergenceTable::reduction_rate_log2);

  //       if (mortar_flag)
  //         {
  //           convergence_table.set_precision("Lambda,Int", 3);
  //           convergence_table.set_scientific("Lambda,Int", true);
  //           convergence_table.set_tex_caption("Lambda,Int",
  //                                             "$ \\|p - \\lambda_H\\|_{d_H} $");
  //           convergence_table.evaluate_convergence_rates(
  //             "Lambda,Int", ConvergenceTable::reduction_rate_log2);
  //         }

  //       if (cycle == refine - 1)
  //         {
  //           std::ofstream error_table_file(
  //             "error" + name +
  //             std::to_string(
  //               Utilities::MPI::n_mpi_processes(mpi_communicator)) +
  //             "domains.tex");
  //           convergence_table.write_text(std::cout);
  //           convergence_table.write_tex(error_table_file);
  //         }
  //     }
  // }



  template <int dim>
  void MixedStokesProblemDD<dim>::compute_errors(const unsigned int &cycle, 
                                                 std::vector<std::vector<unsigned int>> &reps) 
  {
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);
    const ComponentSelectFunction<dim> v_x_mask(0, dim + 1);
    const ComponentSelectFunction<dim> v_y_mask(1, dim + 1);

    ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors(triangulation.n_active_cells());

    QTrapezoid<1>  q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 2);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &v_x_mask);
    
    const double u_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::H1_seminorm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &velocity_mask);
    const double ux_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::H1_seminorm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &v_y_mask);
    const double uy_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::H1_seminorm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &pressure_mask);
    const double p_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::H1_seminorm);

    // std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
    //           << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
    int tmp;
    for (int i = 0; i<n_processes; ++i)
    {
      MPI_Barrier(mpi_communicator);
      if (this_mpi == i)
      {
        std::cout << "this_mpi = " << this_mpi 
                  << ", Errors: ||e_p||_L2 = " << p_l2_error
                  << ",   ||e_u||_L2 = " << u_l2_error 
                  << "\n" << std::flush;
      }
    }
    int mortar_dofs = 0;
    for (int i = 0; i<interface_dofs.size(); ++i)
    {
      mortar_dofs += interface_dofs[i].size();
    }
    mortar_dofs = mortar_dofs / 2;
    const unsigned int n_active_cells = triangulation.n_active_cells();
    // const unsigned int mortar_dofs = interface_dofs.size();
    const unsigned int n_dofs         = dof_handler.n_dofs();
    // const unsigned int n_dofs_mortar  = dof_handler_mortar.n_dofs();

    double order_u;
    double order_p;
    double order_u_total;
    double order_p_total;
    MPI_Allreduce(&u_l2_error, //sending this data
      &u_l2_error_total, //receiving the result here
      1, //number of elements in alpha and alpha_buffer = 1+1
      MPI_DOUBLE, //type of each element
      MPI_SUM, //adding all elements received
      mpi_communicator);
    MPI_Allreduce(&p_l2_error, //sending this data
      &p_l2_error_total, //receiving the result here
      1, //number of elements in alpha and alpha_buffer = 1+1
      MPI_DOUBLE, //type of each element
      MPI_SUM, //adding all elements received
      mpi_communicator);
    // if (cycle == 0)
    // {
    h = reps[this_mpi][0]; // right now this is fine since we take ration in computing order
    h = 1.0/(h*2); // will be used to compute u_order and p_order
    int n;
    tmp = std::min(n_domains[0] * reps[this_mpi][0], n_domains[1] * reps[this_mpi][1]);
    MPI_Allreduce(&tmp, //sending this data
      &n, //receiving the result here
      1, //number of elements in alpha and alpha_buffer = 1+1
      MPI_INT, //type of each element
      MPI_MIN, //find min of element received
      mpi_communicator);
      // std::cout << "this_mpi = " << this_mpi << "reps[this_mpi][0] = " << reps[this_mpi][0] << std::endl;
    // }
    // else
    // {
    //   h = h * 2;
    //   // std::cout << "this_mpi = " << this_mpi << "h = " << h << std::endl;
    // }
    if (cycle > 0)
    {
      order_u = std::log(u_l2_error / u_l2_error_old) / std::log(h / h_old);
      order_p = std::log(p_l2_error / p_l2_error_old) / std::log(h / h_old);
      order_u_total = std::log(u_l2_error_total / u_l2_error_old_total) / std::log(h / h_old);
      order_p_total = std::log(p_l2_error_total / p_l2_error_old_total) / std::log(h / h_old);
    }
    else
    {
      order_u = 0;
      order_p = 0;
      order_u_total = 0;
      order_p_total = 0;
    }
    // int interface_dofs_size; 
    // int tmp;
    // if (dim == 2)
    // {
    //   if (this_mpi % 2 == 0)
    //     tmp = 0;
    //   else
    //     tmp = interface_dofs_total.size();
    //   MPI_Allreduce(&tmp, //sending this data
    //     &interface_dofs_size, //receiving the result here
    //     1, //number of elements in alpha and alpha_buffer = 1+1
    //     MPI_INT, //type of each element
    //     MPI_SUM, //adding all elements received
    //     mpi_communicator);
    // }
    // else
    //   throw std::runtime_error("dim = 3 not yet implemented!");

    double cond;
    double symm;
    if (print_interface_matrix_flag)
    {
      // compute condition number of interface matrix
      unsigned int N = interface_matrix.m();
      Eigen::MatrixXd matrix(N, N);
      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          matrix(i, j) = interface_matrix(i, j);
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
      const auto& singular_values = svd.singularValues();
      cond = singular_values(0) / singular_values(singular_values.size() - 1);
      // check if the matrix is symmetric
      Eigen::MatrixXd emat(N, N);
      emat = matrix - matrix.transpose();
      symm = emat.norm();
    }
    //   double cond = interface_matrix.condition_number();
    
    // convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("h", "1/"+Utilities::int_to_string(n));
    // convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("interface_dofs", interface_dofs_size);
    // convergence_table.add_value("dofs_m", mortar_dofs);
    // convergence_table.add_value("dofs_m", n_dofs_mortar);
    convergence_table.add_value("u_L2", u_l2_error);
    convergence_table.add_value("u_order", order_u);
    convergence_table.add_value("p_L2", p_l2_error);
    convergence_table.add_value("p_order", order_p);
    if (iter_meth_flag == 1)
      convergence_table.add_value("gmres_iter", gmres_iteration);
    else
      convergence_table.add_value("cg_iter", gmres_iteration);
    convergence_table.add_value("residual", residual);
    // convergence_table.add_value("u_H1", u_H1_error);
    // convergence_table.add_value("p_H1", p_H1_error);
    // convergence_table.add_value("ux_H1", ux_H1_error);
    // convergence_table.add_value("uy_H1", uy_H1_error);

    convergence_table_total.add_value("h", "1/"+Utilities::int_to_string(n));
    // convergence_table_total.add_value("dofs", n_dofs);
    convergence_table_total.add_value("interface_dofs", interface_dofs_size);
    // convergence_table_total.add_value("dofs_m", mortar_dofs);
    convergence_table_total.add_value("u_L2", u_l2_error_total);
    convergence_table_total.add_value("u_order", order_u_total);
    convergence_table_total.add_value("p_L2", p_l2_error_total);
    convergence_table_total.add_value("p_order", order_p_total);
    if (iter_meth_flag == 1)
      convergence_table_total.add_value("gmres_iter", gmres_iteration);
    else
      convergence_table_total.add_value("cg_iter", gmres_iteration);
    if (print_interface_matrix_flag)
    {
      convergence_table_total.add_value("cond(S)", cond);
      convergence_table_total.add_value("||S-S'||", symm);
    }
    convergence_table_total.add_value("residual", residual);

    u_l2_error_old = u_l2_error;
    p_l2_error_old = p_l2_error;
    h_old = h;
    u_l2_error_old_total = u_l2_error_total;
    p_l2_error_old_total = p_l2_error_total;
  }


  template <int dim>
  void 
  MixedStokesProblemDD<dim>::write_dof_locations(const std::string &  filename) const
  {
    const FEValuesExtractors::Vector velocities(0);
    std::map<types::global_dof_index, Point<dim>> dof_location_map =
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler,
                                          fe.component_mask(velocities));

    std::ofstream dof_location_file(filename);
    DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                 dof_location_map);
  }

  template <int dim>
  void
  MixedStokesProblemDD<dim>::output_dof_results(const unsigned int cycle) const
  {
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    std::ofstream mesh_file("../output/gnuplot_data/mesh-"
      +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
          +".gnuplot");
    GridOut().write_gnuplot(triangulation, mesh_file);

    if (cont_mortar_flag)
      write_dof_locations("../output/gnuplot_data/dof_loc-"
        +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
        +".gnuplot");
  }

  template <int dim>
  void 
  MixedStokesProblemDD<dim>::write_dof_locations_mortar(const std::string &  filename) const
  {
    const FEValuesExtractors::Vector velocities(0);
    std::map<types::global_dof_index, Point<dim>> dof_location_map=
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler_mortar,
                                          fe_mortar.component_mask(velocities));

    std::ofstream dof_location_file(filename);
    DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                dof_location_map);
  }

  template <int dim>
  void
  MixedStokesProblemDD<dim>::output_dof_results_mortar(const unsigned int cycle) const
  {
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    std::ofstream mesh_file("../output/gnuplot_data/mesh_mortar-"
      +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
          +".gnuplot");
    GridOut().write_gnuplot(triangulation_mortar, mesh_file);
    if (cont_mortar_flag)
      write_dof_locations_mortar("../output/gnuplot_data/dof_loc_mortar-"
        +Utilities::int_to_string(this_mpi)
        +Utilities::int_to_string(cycle)
        +".gnuplot");
  }

  template <int dim>
  void
  MixedStokesProblemDD<dim>::output_results(const unsigned int cycle) const
  {
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(
      "../output/paraview_data/sol_dd-" + Utilities::int_to_string(this_mpi)
      + "_" + Utilities::int_to_string(cycle, 2) 
      + ".vtk");
    data_out.write_vtk(output);
  }

  // MixedStokesProblemDD::run
  template <int dim>
  void
  MixedStokesProblemDD<dim>::run(

    const unsigned int                            refine,
    const std::vector<unsigned int>              &boundary_def,
    std::vector<std::vector<unsigned int>> &reps,
    double                                        tol,
    std::string                                   name,
    unsigned int                                  maxiter,
    unsigned int                                  quad_degree)
  {




    tolerance = tol;
    qdegree   = quad_degree;

    clean_files<dim>();// clean up old files and create output directory structure

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

    if (mortar_flag)
      {
        Assert(n_processes > 1,
               ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
        Assert(reps.size() >= n_processes + 1,
               ExcMessage("Some of the mesh parameters were not provided"));
      }

    for (unsigned int cycle = 0; cycle < refine; ++cycle)
      {
        gmres_iteration = 0;
        interface_dofs.clear();
        interface_dofs_fe.clear();
        interface_dofs_find_neumann.clear();

        if (cycle == 0)
          {
            // Partitioning into subdomains (simple bricks)
            find_divisors<dim>(n_processes, n_domains);

            // Dimensions of the domain (unit hypercube)
            std::vector<double> subdomain_dimensions(dim);
            for (unsigned int d = 0; d < dim; ++d)
              subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

            get_subdomain_coordinates<dim>(
              this_mpi, n_domains, subdomain_dimensions, p1, p2);

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[this_mpi],
                                                        p1,
                                                        p2);
            else
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[0],
                                                        p1,
                                                        p2);

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
                                                        reps[n_processes],
                                                        p1,
                                                        p2);

            
            // GridGenerator::subdivided_hyper_rectangle(triangulation,
            //                                             reps[3],
            //                                             p1,
            //                                             p2);
          }
        else
          {
            // triangulation.refine_global(1);

            // if (mortar_flag)
            // {
            //   triangulation_mortar.refine_global(1);
            //   pcout << "Mortar mesh has "
            //         << triangulation_mortar.n_active_cells() << " cells"
            //         << std::endl;
            // }

            // Partitioning into subdomains (simple bricks)
            find_divisors<dim>(n_processes, n_domains);

            // Dimensions of the domain (unit hypercube)
            std::vector<double> subdomain_dimensions(dim);
            for (unsigned int d = 0; d < dim; ++d)
              subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

            get_subdomain_coordinates<dim>(
              this_mpi, n_domains, subdomain_dimensions, p1, p2);
            
            for (unsigned long i = 0; i < reps[this_mpi].size(); ++i)
            {
              reps[this_mpi][i] = reps[this_mpi][i] * 2;
              reps[n_processes][i] = reps[n_processes][i] * 2;
            }

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[this_mpi],
                                                        p1,
                                                        p2);
            else
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[this_mpi],
                                                        p1,
                                                        p2);

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
                                                        reps[n_processes],
                                                        p1,
                                                        p2);
          }

        
        


        if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
          {
            pcout << "Making grid and DOFs..."
              << "\n";
            
            make_grid_and_dofs(boundary_def);

            pcout << "Assembling system..."
                  << "\n";
        
            assemble_system();
            pcout<<"assembled system!!"<<std::endl;
            solve_bar();
            solution = solution_bar_stokes;

            compute_errors(cycle, reps);

            // computing_timer.print_summary();
            computing_timer.reset();

            // output_results(cycle, refine, name);
            output_dof_results(cycle);
            pcout << "dof output written" << std::endl;
            output_results(cycle);
            pcout << "solution data output written" << std::endl;
            // output_interface_results(cycle);

            triangulation.clear();
            dof_handler.clear();
            // convergence_table.clear();
            faces_on_interface.clear();
            faces_on_interface_mortar.clear();
            interface_dofs.clear();
            interface_fe_function.clear();
            interface_fe_function_mortar.clear();
            if (mortar_flag)
              {
                triangulation_mortar.clear();
                P_fine2coarse.reset();
                P_coarse2fine.reset();
              }
            dof_handler_mortar.clear();
          }
        else
          {
            pcout << "Making grid and DOFs..."
              << "\n";
            make_grid_and_dofs(boundary_def);

            pcout << "Getting interface dofs" << std::endl;
            get_interface_dofs();
            pcout << "Assembling system..."
                  << "\n";
        
            assemble_system();
            pcout<<"Assembled system!!"<<std::endl;
            output_dof_results(cycle);
            pcout << "dof output written" << std::endl;
            if (mortar_flag)
            {
              output_dof_results_mortar(cycle);
                pcout << "dof mortar output written" << std::endl;
            }

            if (iter_meth_flag == 0)
            {
              pcout << "Starting CG iterations..."
                  << "\n";
              local_cg(maxiter, cycle);
            }
            else if (iter_meth_flag == 1)
            {
              pcout << "Starting GMRES iterations..."
                  << "\n";
              local_gmres(maxiter, cycle);
            }
            else
            {
              AssertThrow(false, ExcMessage("Unknown iterative method!"));
            }
            compute_errors(cycle, reps);

            MPI_Barrier(mpi_communicator);

            output_results(cycle);
            // computing_timer.print_summary();
            computing_timer.reset();
            pcout << "----------------------------------------------" << std::endl;
            
            triangulation.clear();
            dof_handler.clear();
            faces_on_interface.clear();
            faces_on_interface_mortar.clear();
            interface_dofs.clear();
            interface_dofs_fe.clear();
            interface_fe_function.clear();
            interface_fe_function_mortar.clear();
            if (mortar_flag)
              {
                triangulation_mortar.clear();
                P_fine2coarse.reset();
                P_coarse2fine.reset();
              }
            dof_handler_mortar.clear();
          }
      }

    print_individual_table_l2<dim>(convergence_table,
                                   iter_meth_flag,
                                   this_mpi,
                                   n_processes,
                                   mpi_communicator);
    print_total_table_l2<dim>(convergence_table_total,
                              iter_meth_flag,
                              print_interface_matrix_flag,
                              this_mpi);
                              
    convergence_table.clear();
    convergence_table_total.clear();
    faces_on_interface.clear();
    faces_on_interface_mortar.clear();
  }

  template class MixedStokesProblemDD<2>;
  // template class MixedStokesProblemDD<3>;
} // namespace dd_stokes

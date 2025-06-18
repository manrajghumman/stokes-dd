/* ---------------------------------------------------------------------
 * Utilities:
 *  - DD & mortar related - mesh subdivision, mortar projection
 *  - Printing the convergence tables
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 *         Manraj Singh Ghumman, University of Pittsburgh, 2023-2025
 */
#ifndef STOKES_MFEDD_UTILITIES_H
#define STOKES_MFEDD_UTILITIES_H

#include "projector.h"
#include <map>
#include <deal.II/lac/block_vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/convergence_table.h>

namespace dd_stokes
{
  using namespace dealii;

  // Domain decomposition related utilities
  // Function to get the true normal vector orientation (in deal.II they are all
  // positive by default)
  
  double
  get_normal_direction(const unsigned int &face)
  {
    std::vector<double> normals(6);
    normals[0] = -1;
    normals[1] = 1;
    normals[2] = 1;
    normals[3] = -1;
    normals[4] = -1;
    normals[5] = 1;

    return normals[face];
  }

  // Split the number into two divisors closest to sqrt (in order to partition
  // mesh into subdomains)
  void
  find_2_divisors(const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
  {
    double tmp = floor(sqrt(n_proc));
    while (true)
      {
        if (fmod(double(n_proc), tmp) == 0)
          {
            n_dom[0] = n_proc / tmp;
            n_dom[1] = tmp;
            break;
          }
        else
          {
            tmp -= 1;
          }
      }
  }

  // Split number into three divisors (Try to make it recursive better)
  template <int dim>
  void
  find_divisors(const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
  {
    Assert(dim == 2 || dim == 3, ExcNotImplemented());

    if (dim == 2)
      find_2_divisors(n_proc, n_dom);
    else if (dim == 3)
      {
        double tmp = floor(pow(n_proc, 1.0 / 3.0));
        while (true)
          {
            if (fmod(static_cast<double>(n_proc), tmp) == 0)
              {
                std::vector<unsigned int> two_divisors(2);
                unsigned int              two_proc = n_proc / tmp;
                find_2_divisors(two_proc, two_divisors);

                n_dom[0] = two_divisors[0];
                n_dom[1] = two_divisors[1];
                n_dom[2] = tmp;
                break;
              }
            else
              {
                tmp -= 1;
              }
          }
      }
  }

  // Compute the lower left and upper right coordinate of a block
  template <int dim>
  void
  get_subdomain_coordinates(const unsigned int &             this_mpi,
                            const std::vector<unsigned int> &n_doms,
                            const std::vector<double> &      dims,
                            Point<dim, double> &                     p1,
                            Point<dim, double> &                     p2)
  {
    switch (dim)
      {
        case 2:
          p1[0] = (this_mpi % n_doms[0]) * dims[0];
          p1[1] = dims[1] * (floor(double(this_mpi) / double(n_doms[0])));
          p2[0] = (this_mpi % n_doms[0]) * dims[0] + dims[0];
          p2[1] =
            dims[1] * (floor(double(this_mpi) / double(n_doms[0]))) + dims[1];
          break;

        case 3:
          p1[0] = (this_mpi % n_doms[0]) * dims[0];
          p1[1] = dims[1] * (floor(double(this_mpi % (n_doms[0] * n_doms[1])) /
                                   double(n_doms[0])));
          p1[2] =
            dims[2] * (floor(double(this_mpi) / double(n_doms[0] * n_doms[1])));
          p2[0] = (this_mpi % n_doms[0]) * dims[0] + dims[0];
          p2[1] = dims[1] * (floor(double(this_mpi % (n_doms[0] * n_doms[1])) /
                                   double(n_doms[0]))) +
                  dims[1];
          p2[2] = dims[2] *
                    (floor(double(this_mpi) / double(n_doms[0] * n_doms[1]))) +
                  dims[2];
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }
  }

  // Find neighboring subdomains
  // output: modifies the vector neighbours and outputs the neighbours 
  //         relative to this_mpi
  void
  find_neighbors(const int &                      dim,
                 const unsigned int &             this_mpi,
                 const std::vector<unsigned int> &n_doms,
                 std::vector<int> &               neighbors)
  {
    Assert(neighbors.size() == 2.0 * dim,
           ExcDimensionMismatch(neighbors.size(), 2.0 * dim));

    switch (dim)
      {
        case 2:
          if (this_mpi % n_doms[0] == 0) // x = a line
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = -1;
            }
          else if ((this_mpi + 1) % n_doms[0] ==
                   0 /*&& (this_mpi + 1) >= n_doms[0]*/) // x = b line
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = -1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
            }
          else
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
            }

          break;

        case 3:
          if (this_mpi % (n_doms[0] * n_doms[1]) == 0) // corner at origin
            {
              neighbors[0] = -1;
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi - n_doms[0] + 1) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at x=a
            {
              neighbors[0] = -1;
              neighbors[1] = -1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + n_doms[0]) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at y=a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = -1;
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + 1) % (n_doms[0] * n_doms[1]) ==
                   0) // corner at x=y=a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = -1;
              neighbors[2] = -1;
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % n_doms[0] == 0) // plane x = a
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = -1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if ((this_mpi + 1) % n_doms[0] == 0) // plane x = b
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = -1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % (n_doms[0] * n_doms[1]) <
                   n_doms[0]) // plane y = a
            {
              neighbors[0] = -1;
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else if (this_mpi % (n_doms[0] * n_doms[1]) >
                   n_doms[0] * n_doms[1] - n_doms[0]) // plane y = b
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = -1;
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          else
            {
              neighbors[0] = this_mpi - n_doms[0];
              neighbors[1] = this_mpi + 1;
              neighbors[2] = this_mpi + n_doms[0];
              neighbors[3] = this_mpi - 1;
              neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
              neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
            }
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }


    for (unsigned int i = 0; i < neighbors.size(); ++i)
      {
        if (neighbors[i] < 0)
          neighbors[i] = -1;
        else if (dim == 2 && neighbors[i] >= int(n_doms[0] * n_doms[1]))
          neighbors[i] = -1;
        else if (dim == 3 &&
                 neighbors[i] >= int(n_doms[0] * n_doms[1] * n_doms[2]))
          neighbors[i] = -1;
      }
  }

  template <int dim, typename Number, int spacedim = dim>
  void
  mark_interface_faces(const Triangulation<dim, spacedim> &tria,
                       const std::vector<int> &  neighbors,
                       const Point<dim, double> &        p1,
                       const Point<dim, double> &        p2,
                       std::vector<Number> &     faces_per_interface)
  {
    Assert(faces_per_interface.size() == neighbors.size(),
           ExcDimensionMismatch(faces_per_interface.size(), neighbors.size()));

    // Label boundaries
    // On unit hypercube for example:
    //  1 - plane y=0, 2 - plane x=1, 3 - plane y=1, 4 - plane x=0, 5 - plane
    //  z=0, 6 - plane z=1 for interfaces, 0 - for outside

    typename Triangulation<dim>::cell_iterator cell, endc;
    cell = tria.begin_active(), endc = tria.end();

    for (; cell != endc; ++cell)
      for (unsigned int face_number = 0;
           face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          // If left boundary of the subdomain in 2d or
          if (std::fabs(cell->face(face_number)->center()(0) - p1[0]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[3] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(4);
                  faces_per_interface[3] += 1;
                }
            }
          // If bottom boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(1) - p1[1]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[0] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(1);
                  faces_per_interface[0] += 1;
                }
            }
          // If right boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(0) - p2[0]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[1] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(2);
                  faces_per_interface[1] += 1;
                }
            }
          // If top boundary of the subdomain
          else if (std::fabs(cell->face(face_number)->center()(1) - p2[1]) <
                   1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[2] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(3);
                  faces_per_interface[2] += 1;
                }
            }
          else if (dim == 3 && std::fabs(cell->face(face_number)->center()(2) -
                                         p1[2]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[4] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(5);
                  faces_per_interface[4] += 1;
                }
            }
          else if (dim == 3 && std::fabs(cell->face(face_number)->center()(2) -
                                         p2[2]) < 1e-12)
            {
              // If it is outside boundary (no neighbor) or interface
              if (neighbors[5] < 0)
                cell->face(face_number)->set_boundary_id(0);
              else
                {
                  cell->face(face_number)->set_boundary_id(6);
                  faces_per_interface[5] += 1;
                }
            }
        }
  }

  template <int dim>
  void
  project_mortar(Projector::Projector<dim> &proj,
                 const DoFHandler<dim> &    dof1,
                 BlockVector<double> &      in_vec,
                 const Quadrature<dim - 1> &quad,
                 AffineConstraints<double> &         constraints,
                 const std::vector<int> &   neighbors,
                 const DoFHandler<dim> &    dof2,
                 BlockVector<double> &      out_vec)
  {
    out_vec = 0;

    Functions::FEFieldFunction<dim, BlockVector<double>>
                                              fe_interface_data(dof1, in_vec);
    std::map<types::global_dof_index, double> boundary_values_velocity;

    std::map<types::boundary_id, const Function<dim, double> *>  boundary_functions_velocity;

    constraints.clear();

    for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell;
         ++side)
      if (neighbors[side] >= 0)
        boundary_functions_velocity[side + 1] = &fe_interface_data;
    
    proj.project_boundary_values(dof2,
                                 boundary_functions_velocity,
                                 quad,
                                 constraints);

    constraints.close();
    constraints.distribute(out_vec);
  }

  template <int dim>
  void
  print_individual_table_l2(ConvergenceTable &convergence_table,
                            const unsigned int &iter_meth_flag,
                            const unsigned int &this_mpi,
                            const unsigned int &n_processes,
                            MPI_Comm &mpi_communicator)
  {
    convergence_table.set_precision("residual", 3);
    convergence_table.set_precision("h", 1);
    convergence_table.set_precision("u_L2", 3);
    convergence_table.set_precision("p_L2", 3);
    convergence_table.set_precision("u_order", 2);
    convergence_table.set_precision("p_order", 2);
    // convergence_table.set_precision("u_H1", 3);
    // convergence_table.set_precision("p_H1", 3);
    // convergence_table.set_precision("ux_H1", 3);
    // convergence_table.set_precision("uy_H1", 3);

    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("h", true);
    convergence_table.set_scientific("u_L2", true);
    convergence_table.set_scientific("p_L2", true);
    // convergence_table.set_scientific("u_order", true);
    // convergence_table.set_scientific("p_order", true);
    // convergence_table.set_scientific("u_H1", true);
    // convergence_table.set_scientific("p_H1", true);
    // convergence_table.set_scientific("ux_H1", true);
    // convergence_table.set_scientific("uy_H1", true);

    convergence_table.set_tex_caption("cells", "\\ cells");
    convergence_table.set_tex_caption("h", "\\ h");
    // convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("interface_dofs", "\\ dofs");
    if (iter_meth_flag == 1)
      convergence_table.set_tex_caption("gmres_iter", "gmres iter");
    else
      convergence_table.set_tex_caption("cg_iter", "cg iter");
    convergence_table.set_tex_caption("u_order", "u order");
    convergence_table.set_tex_caption("p_order", "p order");
    convergence_table.set_tex_caption("residual", "residual");
    convergence_table.set_tex_caption("u_L2", "$L^2$-error (u)");
    convergence_table.set_tex_caption("p_L2", "$L^2$-error (p)");
    // convergence_table.set_tex_caption("u_H1", "$H^1$-error (u)");
    // convergence_table.set_tex_caption("p_H1", "$H^1$-error (p)");
    // convergence_table.set_tex_caption("ux_H1", "$H^1$-error (ux)");
    // convergence_table.set_tex_caption("uy_H1", "$H^1$-error (uy)");

    convergence_table.set_tex_format("h", "r");
    convergence_table.set_tex_format("interface_dofs", "r");
    // convergence_table.set_tex_format("dofs_m", "r");

    // convergence_table.add_column_to_supercolumn("cycle", "n   h");
    // convergence_table.add_column_to_supercolumn("h", "n   h");
    
    // convergence_table.evaluate_convergence_rates(
    //       "u_L2", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "u_L2", "h", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "p_L2", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "p_L2", "h", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "u_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "u_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "p_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "p_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "ux_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "ux_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "uy_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "uy_H1", ConvergenceTable::reduction_rate_log2);

    // convergence_table.set_precision("u_L2-rate", 3);
    // convergence_table.set_precision("p_L2-rate", 3);
    // convergence_table.set_precision("u_H1-rate", 3);
    // convergence_table.set_precision("p_H1-rate", 3);
    // convergence_table.set_precision("ux_H1-rate", 3);
    // convergence_table.set_precision("uy_H1-rate", 3);

    // std::cout << std::endl;
    // convergence_table.write_text(std::cout);

    // std::string conv_filename = "convergence_mpi" + Utilities::int_to_string(this_mpi) + ".tex";

    // std::ofstream table_file(conv_filename);
    // convergence_table.write_tex(table_file);
  
    std::string conv_filename = "../output/convg_tables/table_individual_l2" + std::string(".tex");

    if (this_mpi == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes; ++i)
    {
      MPI_Barrier(mpi_communicator);
      if (this_mpi == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator);
    if (this_mpi == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }

  template <int dim>
  void
  print_individual_table_h1(ConvergenceTable &convergence_table,
                            const unsigned int &iter_meth_flag,
                            const unsigned int &this_mpi,
                            const unsigned int &n_processes,
                            MPI_Comm &mpi_communicator)
  {
    convergence_table.set_precision("h", 1);
    convergence_table.set_precision("u_H1", 3);
    convergence_table.set_precision("p_H1", 3);
    convergence_table.set_precision("ux_H1", 3);
    convergence_table.set_precision("uy_H1", 3);

    convergence_table.set_scientific("h", true);
    convergence_table.set_scientific("u_H1", true);
    convergence_table.set_scientific("p_H1", true);
    convergence_table.set_scientific("ux_H1", true);
    convergence_table.set_scientific("uy_H1", true);

    convergence_table.set_tex_caption("cells", "\\ cells");
    convergence_table.set_tex_caption("h", "\\ h");
    convergence_table.set_tex_caption("interface_dofs", "\\ dofs");
    if (iter_meth_flag == 1)
      convergence_table.set_tex_caption("gmres_iter", "gmres iter");
    else
      convergence_table.set_tex_caption("cg_iter", "cg iter");
    convergence_table.set_tex_caption("u_H1", "$H^1$-error (u)");
    convergence_table.set_tex_caption("p_H1", "$H^1$-error (p)");
    convergence_table.set_tex_caption("ux_H1", "$H^1$-error (ux)");
    convergence_table.set_tex_caption("uy_H1", "$H^1$-error (uy)");

    convergence_table.set_tex_format("h", "r");
    convergence_table.set_tex_format("interface_dofs", "r");
    // convergence_table.set_tex_format("dofs_m", "r");

    // convergence_table.add_column_to_supercolumn("cycle", "n   h");
    // convergence_table.add_column_to_supercolumn("h", "n   h");
    
    // convergence_table.evaluate_convergence_rates(
    //       "u_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "u_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "p_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "p_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "ux_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "ux_H1", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //       "uy_H1", ConvergenceTable::reduction_rate);
    // convergence_table.evaluate_convergence_rates(
    //       "uy_H1", ConvergenceTable::reduction_rate_log2);

    // convergence_table.set_precision("u_L2-rate", 3);
    // convergence_table.set_precision("p_L2-rate", 3);
    // convergence_table.set_precision("u_H1-rate", 3);
    // convergence_table.set_precision("p_H1-rate", 3);
    // convergence_table.set_precision("ux_H1-rate", 3);
    // convergence_table.set_precision("uy_H1-rate", 3);

    // std::cout << std::endl;
    // convergence_table.write_text(std::cout);

    // std::string conv_filename = "convergence_mpi" + Utilities::int_to_string(this_mpi) + ".tex";

    // std::ofstream table_file(conv_filename);
    // convergence_table.write_tex(table_file);
  
    std::string conv_filename = "../output/convg_tables/table_individual_h1" + std::string(".tex");

    if (this_mpi == 0) // latex header
    {
        std::ofstream table_file(conv_filename, std::ios::out);
        table_file << "\\documentclass[10pt]{report}\n"
                   << "\\usepackage{float}\n"
                   << "\\usepackage[margin=0.7in]{geometry}\n\n"
                   << "\\begin{document}\n\n";
        table_file.close();
    }
    for (unsigned int i = 0; i<n_processes; ++i)
    {
      MPI_Barrier(mpi_communicator);
      if (this_mpi == i)
      { 
        // Print to screen if you like, but messy for many processes
        // convergence_table.write_text(std::cout);

        std::ofstream table_file(conv_filename, std::ios::app);
        // Write a LaTeX comment and a section header before the table:
        table_file << "\n% ===== MPI Process " << this_mpi << " =====\n";
        table_file << "\\section*{MPI Process " << this_mpi << "}\n";
        convergence_table.write_tex(table_file, false);
        table_file << "\n"; // Optional: extra space after each table
      }
    }
    MPI_Barrier(mpi_communicator);
    if (this_mpi == 0)
    {
        std::ofstream table_file(conv_filename, std::ios::app);
        table_file << "\n\\end{document}\n";
    }
  }
  
  template <int dim>
  void
  print_total_table_l2(ConvergenceTable &convergence_table_total,
                       const unsigned int &iter_meth_flag,
                       const unsigned int &print_interface_matrix_flag,
                       const unsigned int &this_mpi)
  {
    // only one process prints this data
    if (this_mpi == 0)
    {
      convergence_table_total.set_precision("residual", 3);
      convergence_table_total.set_precision("h", 1);
      convergence_table_total.set_precision("u_L2", 3);
      convergence_table_total.set_precision("p_L2", 3);
      convergence_table_total.set_precision("u_order", 2);
      convergence_table_total.set_precision("p_order", 2);
      if (print_interface_matrix_flag)
      {
        convergence_table_total.set_precision("cond(S)", 3);
        convergence_table_total.set_precision("||S-S'||", 3);
      }

      convergence_table_total.set_scientific("residual", true);
      convergence_table_total.set_scientific("h", true);
      convergence_table_total.set_scientific("u_L2", true);
      convergence_table_total.set_scientific("p_L2", true);
      if (print_interface_matrix_flag)
      {
        convergence_table_total.set_scientific("cond(S)", true);
        convergence_table_total.set_scientific("||S-S'||", true);
      }

      // convergence_table_total.set_tex_caption("cells", "\\# cells");
      convergence_table_total.set_tex_caption("h", "\\ h");
      // convergence_table_total.set_tex_caption("dofs", "\\# dofs");
      convergence_table_total.set_tex_caption("interface_dofs", "\\ dofs");
      if (iter_meth_flag == 1)
        convergence_table_total.set_tex_caption("gmres_iter", "gmres iter");
      else
        convergence_table_total.set_tex_caption("cg_iter", "cg iter");
      convergence_table_total.set_tex_caption("u_order", "u order");
      convergence_table_total.set_tex_caption("p_order", "p order");
      convergence_table_total.set_tex_caption("residual", "residual");
      convergence_table_total.set_tex_caption("u_L2", "$L^2$-error (u)");
      convergence_table_total.set_tex_caption("p_L2", "$L^2$-error (p)");
      if (print_interface_matrix_flag)
        convergence_table_total.set_tex_caption("||S-S'||", "$||S-S'||$");

      convergence_table_total.set_tex_format("h", "r");
      convergence_table_total.set_tex_format("interface_dofs", "r");
      // convergence_table_total.set_tex_format("dofs_m", "r");

      convergence_table_total.write_text(std::cout);

      std::string conv_filename = std::string("../output/convg_table_total/convergence_total_l2") + ".tex";

      std::cout << "total error\n" << std::endl;
      std::ofstream table_file(conv_filename, std::ios::out);
      table_file << "\\documentclass[10pt]{report}\n"
                 << "\\usepackage{float}\n"
                 << "\\usepackage[margin=0.7in]{geometry}\n\n"
                 << "\\begin{document}\n\n";
      
      convergence_table_total.write_tex(table_file, false);
      table_file << "\n\\end{document}\n";
      table_file.close();
    }
  }

  template <int dim>
  void
  print_total_table_h1()
  {

  }

} // namespace dd_stokes

#endif // STOKES_MFEDD_UTILITIES_H

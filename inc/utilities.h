/* ---------------------------------------------------------------------
 * Utilities:
 *  - Elasticity related - physical tensors and asymmetry operators
 *  - DD & mortar related - mesh subdivision, mortar projection
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 */
#ifndef STOKES_MFEDD_UTILITIES_H
#define STOKES_MFEDD_UTILITIES_H

#include "projector.h"
#include <map>
#include <deal.II/lac/block_vector.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <filesystem>

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

// Function to get the true normal and tangent vector orientation (in deal.II they are all
// positive by default)
  // double
  // get_normal_direction(const unsigned int &face, unsigned int &dof)
  // {
  //   std::vector<std::vector<double>> normals(4, std::vector<double>(2));
  //   unsigned int coordinate;
  //   coordinate = dof % 2;
  //   normals[0][0] = 1;//tangent direction
  //   normals[0][1] = -1;//normal direction
  //   normals[1][0] = 1;
  //   normals[1][1] = 1;
  //   normals[2][0] = -1;
  //   normals[2][1] = 1;
  //   normals[3][0] = -1;
  //   normals[3][1] = -1;

  //   return normals[face][coordinate];
  // }
  // template<int,dim>
  // Tensor<1,2>
  // get_tangent_direction(Tensor<1,2> &normal_vector)
  // {
  //   Tensor<1,2> tangent;
  //   tangent[0] = -normal_vector[1];
  //   tangent[1] = normal_vector[0];
  //   // tangent[2] = -1;
  //   // tangent[3] = -1;

  //   return tangent;
  // }

  // std::vector<double>
  // tangent_direction(const unsigned int &face)
  // {
  //   std::vector<double> tangent(4);
  //   tangent[0] = 1;
  //   tangent[1] = 1;
  //   tangent[2] = -1;
  //   tangent[3] = -1;

  //   return tangent[face];
  // }

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

  /* 
  In output folder create output directories if not present
  and clean up old files if present.
  Will create .gitignore and .gitkeep in each subdirectory
  with .gitignore
  {
  *
  !.gitkeep
  }
  add any files you want to keep in .gitkeep
  this way we can keep the directory structure in git
  */
  template <int dim>
  void
  clean_files()
  {
    namespace fs = std::filesystem;
    // Check if output directory exists
    if (!fs::exists("../output/"))
      fs::create_directory("../output/");

    // interface_data directory
    {
      // Check if interface_data directory exists
    if (!fs::exists("../output/interface_data/"))
      fs::create_directory("../output/interface_data/");
    // Remove old files in the interface_data directory
    for (const auto &entry : fs::directory_iterator("../output/interface_data/"))
      {
        const auto &path = entry.path();
        std::string filename = path.filename().string();
        if (entry.is_regular_file() && filename[0] != '.')
          fs::remove(entry.path());
      }
    // If the directory does not exist, create it
    if (!fs::exists("../output/interface_data/"))
      fs::create_directory("../output/interface_data/");
    // If the directory exists, but is not a directory, throw an error
    if (fs::exists("../output/interface_data/") && !fs::is_directory("../output/interface_data/"))
      throw std::runtime_error("../output/interface_data/ is not a directory.");
    
    // create .gitkeep (empty file)
    std::ofstream(std::string("../output/interface_data") + "/.gitkeep").close();
    // create .gitignore
    std::ofstream gitignore(std::string("../output/interface_data") + "/.gitignore");
    gitignore << "*\n!.gitkeep\n";
    gitignore.close();
    }
    // paraview_data directory
    {
      // Check if paraview_data directory exists
    if (!fs::exists("../output/paraview_data/"))
      fs::create_directory("../output/paraview_data/");
    // Remove old files in the paraview_data directory
    for (const auto &entry : fs::directory_iterator("../output/paraview_data/"))
      {
        const auto &path = entry.path();
        std::string filename = path.filename().string();
        if (entry.is_regular_file() && filename[0] != '.')
          fs::remove(entry.path());
      }
    // If the directory does not exist, create it
    if (!fs::exists("../output/paraview_data/"))
      fs::create_directory("../output/paraview_data/");
    // If the directory exists, but is not a directory, throw an error
    if (fs::exists("../output/paraview_data/") && !fs::is_directory("../output/paraview_data/"))
      throw std::runtime_error("../output/paraview_data/ is not a directory.");

    // create .gitkeep (empty file)
    std::ofstream(std::string("../output/paraview_data") + "/.gitkeep").close();
    // create .gitignore
    std::ofstream gitignore(std::string("../output/paraview_data") + "/.gitignore");
    gitignore << "*\n!.gitkeep\n";
    gitignore.close();
    }

    //gnuplot_data directory
    {
      // Check if gnuplot_data directory exists
    if (!fs::exists("../output/gnuplot_data/"))
      fs::create_directory("../output/gnuplot_data/");
    // Remove old files in the gnuplot_data directory
    for (const auto &entry : fs::directory_iterator("../output/gnuplot_data/"))
      {
        const auto &path = entry.path();
        std::string filename = path.filename().string();
        if (entry.is_regular_file() && filename[0] != '.')
          fs::remove(entry.path());
      }
    // If the directory does not exist, create it
    if (!fs::exists("../output/gnuplot_data/"))
      fs::create_directory("../output/gnuplot_data/");
    // If the directory exists, but is not a directory, throw an error
    if (fs::exists("../output/gnuplot_data/") && !fs::is_directory("../output/gnuplot_data/"))
      throw std::runtime_error("../output/gnuplot_data/ is not a directory.");

    // create .gitkeep (empty file)
    std::ofstream(std::string("../output/gnuplot_data") + "/.gitkeep").close();
    // create .gitignore
    std::ofstream gitignore(std::string("../output/gnuplot_data") + "/.gitignore");
    gitignore << "*\n!.gitkeep\n";
    gitignore.close();
    }

    // convg_tables directory
    {
      // Check if convg_tables directory exists
    if (!fs::exists("../output/convg_tables/"))
      fs::create_directory("../output/convg_tables/");
    // Remove old files in the convg_tables directory
    for (const auto &entry : fs::directory_iterator("../output/convg_tables/"))
      {
        const auto &path = entry.path();
        std::string filename = path.filename().string();
        if (entry.is_regular_file() && filename[0] != '.')
          fs::remove(entry.path());
      }
    // If the directory does not exist, create it
    if (!fs::exists("../output/convg_tables/"))
      fs::create_directory("../output/convg_tables/");
    // If the directory exists, but is not a directory, throw an error
    if (fs::exists("../output/convg_tables/") && !fs::is_directory("../output/convg_tables/"))
      throw std::runtime_error("../output/convg_tables/ is not a directory.");

    // create .gitkeep (empty file)
    std::ofstream(std::string("../output/convg_tables") + "/.gitkeep").close();
    // create .gitignore
    std::ofstream gitignore(std::string("../output/convg_tables") + "/.gitignore");
    gitignore << "*\n!.gitkeep\n";
    gitignore.close();
    }

    // convg_table_total directory
    {
      // Check if convg_table_total directory exists
    if (!fs::exists("../output/convg_table_total/"))
      fs::create_directory("../output/convg_table_total/");
    // Remove old files in the convg_table_total directory
    for (const auto &entry : fs::directory_iterator("../output/convg_table_total/"))
      {
        const auto &path = entry.path();
        std::string filename = path.filename().string();
        if (entry.is_regular_file() && filename[0] != '.')
          fs::remove(entry.path());
      }
    // If the directory does not exist, create it
    if (!fs::exists("../output/convg_table_total/"))
      fs::create_directory("../output/convg_table_total/");
    // If the directory exists, but is not a directory, throw an error
    if (fs::exists("../output/convg_table_total/") && !fs::is_directory("../output/convg_table_total/"))
      throw std::runtime_error("../output/convg_table_total/ is not a directory.");

    // create .gitkeep (empty file)
    std::ofstream(std::string("../output/convg_table_total") + "/.gitkeep").close();
    // create .gitignore
    std::ofstream gitignore(std::string("../output/convg_table_total") + "/.gitignore");
    gitignore << "*\n!.gitkeep\n";
    gitignore.close();
    }
  }

  /*
  Opens a file with the given file(filename) 
  std::ios::trunc will erase previous  
  content if it exists if not std::ios::out
  will write it to file
  */

  template <int dim>
  void
  name_files(const unsigned int              &this_mpi,
            unsigned int                     &cycle,
            std::vector<int>                 &neighbors,
            std::vector<std::ofstream>       &file,
            std::vector<std::ofstream>       &file_exact,
            std::vector<std::ofstream>       &file_residual,
            std::vector<std::ofstream>       &file_y,
            std::vector<std::ofstream>       &file_exact_y,
            std::vector<std::ofstream>       &file_residual_y)
  {
   const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (int side=0; side < n_faces_per_cell; ++side){
        if (neighbors[side] >= 0)
        {
          file[side].open("../output/interface_data/lambda" + Utilities::int_to_string(this_mpi)
                      + "_" + Utilities::int_to_string(side, 1)+ "_" 
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 
          file_exact[side].open("../output/interface_data/lambda_exact" + Utilities::int_to_string(this_mpi)                    
                      + "_" + Utilities::int_to_string(side, 1) + "_"
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
          file_residual[side].open("../output/interface_data/residual" + Utilities::int_to_string(this_mpi)
                      + "_" + Utilities::int_to_string(side, 1) + "_"
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 

          file_y[side].open("../output/interface_data/lambda_y" + Utilities::int_to_string(this_mpi)
                      + "_" + Utilities::int_to_string(side, 1) + "_"
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
          file_exact_y[side].open("../output/interface_data/lambda_exact_y" + Utilities::int_to_string(this_mpi)
                      + "_" + Utilities::int_to_string(side, 1) + "_"
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
         file_residual_y[side].open("../output/interface_data/residual_y" + Utilities::int_to_string(this_mpi)
                      + "_" + Utilities::int_to_string(side, 1) + "_"
                      + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 
        }
      }
  }

  template <int dim>
  void
  plot_approx_function(const unsigned int      &this_mpi,
              const unsigned int               &mortar_flag,
              const unsigned int               &mortar_degree,
              std::vector<std::vector<unsigned int>> &interface_dofs,
              std::vector<int>                 &neighbors,
              std::vector<std::vector<double>> &lambda,
              std::vector<std::vector<double>> &plot,
              std::vector<std::vector<double>> &plot_y,
              std::vector<std::ofstream>       &file,
              std::vector<std::ofstream>       &file_y)
  { 
    unsigned int q;
    double tmp;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
     for (int side = 0; side < n_faces_per_cell; ++side){
            plot[side].clear();
            plot_y[side].clear();
          }
          q = 0;
          // int k1;
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            { 
              q = q+1;
              // k1 = 0;
              // pcout << "q value = " << q << std::endl;
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 0)//put x values
                { 
                  plot[side].push_back(lambda[side][i]);
                }
              }
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 1)//put y values
                {
                  plot_y[side].push_back(lambda[side][i]);
                }
              }
              if (!mortar_flag || mortar_degree == 2)
              {
                // Rearrange the x and y values plotted according to the correct dof
                int k = interface_dofs[side].size()/2;
                if (k % 2 == 1)
                {
                  for (int i = 1; i < k-1; i+=2)
                  {
                    tmp = plot[side][i];
                    plot[side][i] = plot[side][i + 1];
                    plot[side][i + 1] = tmp;

                    tmp = plot_y[side][i];
                    plot_y[side][i] = plot_y[side][i +1];
                    plot_y[side][i + 1] = tmp;  
                  }
                }
                else if (k % 2 == 0 & this_mpi % 2 == 0 )
                {
                  for (int i = 1; i < k-3; i+=2)
                  {
                    tmp = plot[side][i];
                    plot[side][i] = plot[side][i + 1];
                    plot[side][i + 1] = tmp;

                    tmp = plot_y[side][i];
                    plot_y[side][i] = plot_y[side][i +1];
                    plot_y[side][i + 1] = tmp;
                  }
                }
                else 
                {
                  for (int i = 0; i < k-2; i+=2)
                  {
                    tmp = plot[side][i];
                    plot[side][i] = plot[side][i + 1];
                    plot[side][i + 1] = tmp;

                    tmp = plot_y[side][i];
                    plot_y[side][i] = plot_y[side][i +1];
                    plot_y[side][i + 1] = tmp;
                  }
                }
              }
            }  

    for (int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            {
              for (const double& value : plot[side])
                file[side] << value << " ";
              file[side] << "\n";

              for (const double& value : plot_y[side])
                file_y[side] << value << " ";
              file_y[side] << "\n";
            }
  }


  template <int dim>
  void
  plot_exact_function( const unsigned int                &this_mpi,
              const unsigned int               &mortar_flag,
              const unsigned int               &mortar_degree,
              std::vector<std::vector<unsigned int>> &interface_dofs,
              std::vector<int>                 &neighbors,
              std::vector<BlockVector<double>> &exact_normal_stress_at_nodes,
              std::vector<std::vector<double>> &plot_exact,
              std::vector<std::vector<double>> &plot_exact_y,
              std::vector<std::ofstream>       &file_exact,
              std::vector<std::ofstream>       &file_exact_y)
  { 
    unsigned int q;
    double tmp;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (int side = 0; side < n_faces_per_cell; ++side){
            plot_exact[side].clear();
            plot_exact_y[side].clear();
          }
          q = 0;
          // int k1;
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            { 
              q = q+1;
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 0)//put x values
                { 
                  plot_exact[side].push_back(exact_normal_stress_at_nodes[side][interface_dofs[side][i]]);
                }
              }
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 1)//put y values
                {
                  plot_exact_y[side].push_back(exact_normal_stress_at_nodes[side][interface_dofs[side][i]]);
                }
              }
              if (!mortar_flag || mortar_degree == 2)
              {
                // Rearrange the x and y values plotted according to the correct dof
                int k = interface_dofs[side].size()/2;
                if (k % 2 == 1)
                {
                  for (int i = 1; i < k-1; i+=2)
                  {
                    tmp = plot_exact[side][i];
                    plot_exact[side][i] = plot_exact[side][i + 1];
                    plot_exact[side][i + 1] = tmp;

                    tmp = plot_exact_y[side][i];
                    plot_exact_y[side][i] = plot_exact_y[side][i + 1];
                    plot_exact_y[side][i + 1] = tmp;
                  }
                }
                else if (k % 2 == 0 & this_mpi % 2 == 0 )
                {
                  for (int i = 1; i < k-3; i+=2)
                  {
                    tmp = plot_exact[side][i];
                    plot_exact[side][i] = plot_exact[side][i + 1];
                    plot_exact[side][i + 1] = tmp;

                    tmp = plot_exact_y[side][i];
                    plot_exact_y[side][i] = plot_exact_y[side][i + 1];
                    plot_exact_y[side][i + 1] = tmp;
                  }
                }
                else 
                {
                  for (int i = 0; i < k-2; i+=2)
                  {
                    tmp = plot_exact[side][i];
                    plot_exact[side][i] = plot_exact[side][i + 1];
                    plot_exact[side][i + 1] = tmp;

                    tmp = plot_exact_y[side][i];
                    plot_exact_y[side][i] = plot_exact_y[side][i + 1];
                    plot_exact_y[side][i + 1] = tmp;
                  }
                }
              }
            }  

    for (int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            {
              for (const double& value : plot_exact[side])
                file_exact[side] << value << " ";
              file_exact[side] << "\n";

              for (const double& value : plot_exact_y[side])
                file_exact_y[side] << value << " ";
              file_exact_y[side] << "\n";

            }
  }

  template <int dim>
  void
  plot_residual_function(const unsigned int               &this_mpi,
                const unsigned int               &mortar_flag,
                const unsigned int               &mortar_degree,
                std::vector<std::vector<unsigned int>> &interface_dofs,
                std::vector<int>                 &neighbors,
                std::vector<std::vector<double>> &r,
                std::vector<std::vector<double>> &plot_residual,
                std::vector<std::vector<double>> &plot_residual_y,
                std::vector<std::ofstream>       &file_residual,
                std::vector<std::ofstream>       &file_residual_y)
  { 
    unsigned int q;
    double tmp;
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (int side = 0; side < n_faces_per_cell; ++side){
            plot_residual[side].clear();
            plot_residual_y[side].clear();
          }
          q = 0;
          // int k1;
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            { 
              q = q+1;
              // k1 = 0;
              // pcout << "q value = " << q << std::endl;
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 0)//put x values
                { 
                  plot_residual[side].push_back(r[side][i]);
                }
              }
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              { 
                if (i%2 == 1)//put y values
                {
                  plot_residual_y[side].push_back(r[side][i]);
                }
              }
              if (!mortar_flag || mortar_degree == 2)
              {
                // Rearrange the x and y values plotted according to the correct dof
                int k = interface_dofs[side].size()/2;
                if (k % 2 == 1)
                {
                  for (int i = 1; i < k-1; i+=2)
                  {
                    tmp = plot_residual[side][i];
                    plot_residual[side][i] = plot_residual[side][i + 1];
                    plot_residual[side][i + 1] = tmp;

                    tmp = plot_residual_y[side][i];
                    plot_residual_y[side][i] = plot_residual_y[side][i + 1];
                    plot_residual_y[side][i + 1] = tmp;    
                  }
                }
                else if (k % 2 == 0 & this_mpi % 2 == 0 )
                {
                  for (int i = 1; i < k-3; i+=2)
                  {
                    tmp = plot_residual[side][i];
                    plot_residual[side][i] = plot_residual[side][i + 1];
                    plot_residual[side][i + 1] = tmp;

                    tmp = plot_residual_y[side][i];
                    plot_residual_y[side][i] = plot_residual_y[side][i + 1];
                    plot_residual_y[side][i + 1] = tmp;    
                  }
                }
                else 
                {
                  for (int i = 0; i < k-2; i+=2)
                  {
                    tmp = plot_residual[side][i];
                    plot_residual[side][i] = plot_residual[side][i + 1];
                    plot_residual[side][i + 1] = tmp;

                    tmp = plot_residual_y[side][i];
                    plot_residual_y[side][i] = plot_residual_y[side][i + 1];
                    plot_residual_y[side][i + 1] = tmp;    
                  }
                }
              }
            }  
    
    for (int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
            {
              for (const double& value : plot_residual[side])
                file_residual[side] << value << " ";
              file_residual[side] << "\n";

              for (const double& value : plot_residual_y[side])
                file_residual_y[side] << value << " ";
              file_residual_y[side] << "\n";
            }
  }

  template <int dim>
  void 
  constant_residual_two (std::vector<std::vector<unsigned int>> &interface_dofs,
                        std::vector<int>                       &neighbors,
                        std::vector<unsigned int>              &repeated_dofs,
                        std::vector<unsigned int>              &repeated_dofs_neumann,
                        std::vector<std::vector<double>>       &r)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i){
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
                            interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = r[side][i+4]; 
            }
            else{
              r[side][i] = r[side][i+2]; 
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
                            interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = r[side][i+4]; 
            }
            else{
              r[side][i] = r[side][i+2]; 
            }
          }
        }
  }

   template <int dim>
  void 
  constant_Ap_two (std::vector<std::vector<unsigned int>>    &interface_dofs,
                  std::vector<int>                          &neighbors,
                  std::vector<unsigned int>                 &repeated_dofs,
                  std::vector<unsigned int>                 &repeated_dofs_neumann,
                  std::vector<std::vector<double>>          &interface_data_send)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
              interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = interface_data_send[side][i+4];
            }
            else
            {
              interface_data_send[side][i] = interface_data_send[side][i+2];
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
              interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = interface_data_send[side][i+4]; 
            }
            else 
            {
              interface_data_send[side][i] = interface_data_send[side][i+2]; 
            }
          }
        }
      }
  }

  template <int dim>
  void 
  average_residual_two (std::vector<std::vector<unsigned int>> &interface_dofs,
                        std::vector<int>                       &neighbors,
                        std::vector<unsigned int>              &repeated_dofs,
                        std::vector<unsigned int>              &repeated_dofs_neumann,
                        std::vector<std::vector<double>>       &r)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i){
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
                            interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = (r[side][i+4]+r[side][i])/2; 
              r[side][i+4] = r[side][i];
            }
            else{
              r[side][i] = (r[side][i+2]+r[side][i])/2; 
              r[side][i+2] = r[side][i];
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
                            interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = (r[side][i+4]+r[side][i])/2; 
              r[side][i+4] = r[side][i];
            }
            else{
              r[side][i] = (r[side][i+2]+r[side][i])/2; 
              r[side][i+2] = r[side][i];
            }
          }
        }
  }

   template <int dim>
  void 
  average_Ap_two (std::vector<std::vector<unsigned int>>    &interface_dofs,
                  std::vector<int>                          &neighbors,
                  std::vector<unsigned int>                 &repeated_dofs,
                  std::vector<unsigned int>                 &repeated_dofs_neumann,
                  std::vector<std::vector<double>>          &interface_data_send)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
              interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = (interface_data_send[side][i+4]+interface_data_send[side][i])/2;
              interface_data_send[side][i+4] = interface_data_send[side][i];
            }
            else
            {
              interface_data_send[side][i] = (interface_data_send[side][i+2]+interface_data_send[side][i])/2;
              interface_data_send[side][i+2] = interface_data_send[side][i];
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
              interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = (interface_data_send[side][i+4]+interface_data_send[side][i])/2; 
              interface_data_send[side][i+4] = interface_data_send[side][i];
            }
            else 
            {
              interface_data_send[side][i] = (interface_data_send[side][i+2]+interface_data_send[side][i])/2; 
              interface_data_send[side][i+2] = interface_data_send[side][i];
            }
          }
        }
      }
  }

  template <int dim>
  void 
  average_residual_three (std::vector<std::vector<unsigned int>> &interface_dofs,
                          std::vector<int>                       &neighbors,
                          std::vector<unsigned int>              &repeated_dofs,
                          std::vector<unsigned int>              &repeated_dofs_neumann,
                          std::vector<std::vector<double>>       &r)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i){
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
                            interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = (r[side][i+4]+r[side][i+2]+r[side][i])/3; 
              r[side][i+2] = r[side][i];
              r[side][i+4] = r[side][i];
            }
            else{
              r[side][i] = (r[side][i+2]+r[side][i]+r[side][i-4])/3; 
              r[side][i+2] = r[side][i];
              r[side][i-4] = r[side][i];
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
                            interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1){
              r[side][i] = (r[side][i+4]+r[side][i+2]+r[side][i])/3; 
              r[side][i+2] = r[side][i];
              r[side][i+4] = r[side][i];
            }
            else{
              r[side][i] = (r[side][i+2]+r[side][i]+r[side][i-4])/3; 
              r[side][i+2] = r[side][i];
              r[side][i-4] = r[side][i];
            }
          }
        }
  }

   template <int dim>
  void 
  average_Ap_three (std::vector<std::vector<unsigned int>>    &interface_dofs,
                    std::vector<int>                          &neighbors,
                    std::vector<unsigned int>                 &repeated_dofs,
                    std::vector<unsigned int>                 &repeated_dofs_neumann,
                    std::vector<std::vector<double>>          &interface_data_send)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
      {
        for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          if (std::find(repeated_dofs.begin(), repeated_dofs.end(), 
              interface_dofs[side][i]) != repeated_dofs.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = (interface_data_send[side][i+4]
                          +interface_data_send[side][i]+interface_data_send[side][i+2])/3;
              interface_data_send[side][i+4] = interface_data_send[side][i];
              interface_data_send[side][i+2] = interface_data_send[side][i];
            }
            else
            {
              interface_data_send[side][i] = (interface_data_send[side][i+2]
                          +interface_data_send[side][i]+interface_data_send[side][i-4])/3;
              interface_data_send[side][i+2] = interface_data_send[side][i];
              interface_data_send[side][i-4] = interface_data_send[side][i];
            }
          }

          if (std::find(repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), 
              interface_dofs[side][i]) != repeated_dofs_neumann.end())
          { 
            if (i == 0 || i == 1)
            { 
              interface_data_send[side][i] = (interface_data_send[side][i+4]
                            +interface_data_send[side][i]+interface_data_send[side][i+2])/3; 
              interface_data_send[side][i+4] = interface_data_send[side][i];
              interface_data_send[side][i+2] = interface_data_send[side][i];
            }
            else 
            {
              interface_data_send[side][i] = (interface_data_send[side][i+2]
                              +interface_data_send[side][i]+interface_data_send[side][i-4])/3; 
              interface_data_send[side][i+2] = interface_data_send[side][i];
              interface_data_send[side][i-4] = interface_data_send[side][i];
            }
          }
        }
      }
  }

  // Extracting the dirichlet dofs on the outside bdry shared between subdomains 
  template <int dim, int spacedim = dim>
  void 
  find_interface_dofs_dirichlet (DoFHandler<dim, spacedim>                           &dof_handler,
                           std::vector<std::vector<unsigned int>>    &interface_dofs,
                           std::vector<types::global_dof_index>      &local_face_dof_indices,
                           unsigned long                             &n_velocity_interface,
                           std::vector<int>                          &neighbors,
                           std::vector<unsigned int>                 &repeated_dofs)
  { 
    repeated_dofs.clear();
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (const auto &cell: dof_handler.active_cell_iterators())
    {
      for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() == 0))
            { 
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
              for (auto el: local_face_dof_indices)
                if (el < n_velocity_interface)
                  for (int side = 0; side < n_faces_per_cell;++side)
                    if (neighbors[side]>=0)
                      if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                      != interface_dofs[side].end())//enters this statement if local_dof_indices[i] belongs to interface_dofs[side]
                        if (std::find (repeated_dofs.begin(), repeated_dofs.end(), el) 
                                                                  == repeated_dofs.end())
                          {
                            repeated_dofs.push_back(el);
                            // if (this_mpi == 0)
                            // { 
                            //   std::cout << "repeated_dofs = " << el << std::endl;
                            // }
                          }
              }       
    }
    // std::cout << "repeated_dofs.size() = " << repeated_dofs.size() << std::endl; 
   
    //   for (int m = 0; m < repeated_dofs.size(); ++m)
    //     std::cout << "repeated_dofs = " << repeated_dofs[m] << std::endl;  
  }

  // Extracting the neumann dofs on the outside bdry shared between subdomains 
  template <int dim, int spacedim =  dim>
  void 
  find_interface_dofs_neumann (DoFHandler<dim, spacedim>                     &dof_handler,
                         std::vector<std::vector<unsigned int>>    &interface_dofs,
                         std::vector<types::global_dof_index>      &local_face_dof_indices,
                         unsigned long                             &n_velocity_interface,
                         std::vector<int>                          &neighbors,
                         std::vector<unsigned int>                 &repeated_dofs_neumann)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (const auto &cell: dof_handler.active_cell_iterators())
    {
        repeated_dofs_neumann.clear();  
        for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() == 7))
            { 
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
              for (auto el: local_face_dof_indices)
                if (el < n_velocity_interface)
                  for (int side = 0; side<n_faces_per_cell;++side)
                    if (neighbors[side]>=0)
                      if (std::find (interface_dofs[side].begin(), interface_dofs[side].end(), el) 
                                      != interface_dofs[side].end())//enters this statement if local_dof_indices[i] belongs to interface_dofs[side]
                        if (std::find (repeated_dofs_neumann.begin(), repeated_dofs_neumann.end(), el) 
                                                                  == repeated_dofs_neumann.end())
                          {
                            repeated_dofs_neumann.push_back(el);
                            // if (this_mpi == 0)
                            // { 
                            //   std::cout << "repeated_dofs_neumann = " << el << std::endl;
                            // }
                          }
              }
    }
    // std::cout << "repeated_dofs_neumann.size() = " << repeated_dofs_neumann.size() << std::endl; 
   
    //   for (int m = 0; m < repeated_dofs_neumann.size(); ++m)
    //     std::cout << "repeated_dofs_neumann = " << repeated_dofs_neumann[m] << std::endl; 
  }

  // Remove dirichlet dof on interface from interface_dofs
  template <int dim, int spacedim = dim>
  void 
  remove_interface_dirichlet_dofs (DoFHandler<dim, spacedim>                 &dof_handler,
                         std::vector<std::vector<unsigned int>>    &interface_dofs,
                         std::vector<types::global_dof_index>      &local_face_dof_indices,
                         unsigned long                             &n_velocity_interface)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    unsigned int side = 0;
    for (const auto &cell: dof_handler.active_cell_iterators())
      {
        for (unsigned int face_n = 0;
             face_n < n_faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() == 0))
            { 
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
              // pcout << "side = " << side << std::endl;
              side = face_n;
              // pcout << "side = " << side << std::endl;
              for (auto el : local_face_dof_indices){
                if (el < n_velocity_interface){
                  // pcout << "interface_dofs[side] = " << interface_dofs[side].size() << std::endl;
                  for (int i = 0; i<4; ++i){
                    interface_dofs[i].erase(std::remove(interface_dofs[i].begin(), 
                                        interface_dofs[i].end(), el), interface_dofs[i].end());
                  }
                }
              }
            }
      }
  }

   // Remove neumann dof on interface from interface_dofs
  template <int dim, int spacedim = dim>
  void 
  remove_interface_neumann_dofs (DoFHandler<dim, spacedim>                 &dof_handler,
                         std::vector<std::vector<unsigned int>>    &interface_dofs,
                         std::vector<types::global_dof_index>      &local_face_dof_indices,
                         unsigned long                             &n_velocity_interface)
  {
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    unsigned int side = 0;
    for (const auto &cell: dof_handler.active_cell_iterators())
      {
        for (unsigned int face_n = 0;
             face_n < n_faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              (cell->face(face_n)->boundary_id() == 7))
            { 
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);
              // pcout << "side = " << side << std::endl;
              side = face_n;
              // pcout << "side = " << side << std::endl;
              for (auto el : local_face_dof_indices){
                if (el < n_velocity_interface){
                  // pcout << "interface_dofs[side] = " << interface_dofs[side].size() << std::endl;
                  for (int i = 0; i<4; ++i){
                    interface_dofs[i].erase(std::remove(interface_dofs[i].begin(), 
                                        interface_dofs[i].end(), el), interface_dofs[i].end());
                  }
                }
              }
            }
      }
  }
  
  // Extracting the neumann dofs on the interface corner point shared between subdomains  
  template <int dim>
  void 
  find_interface_dofs_neumann_corner (std::vector<std::vector<unsigned int>>    &interface_dofs_find_neumann,
                            std::vector<unsigned int>                 &repeated_dofs_neumann_corner)
  { 
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    repeated_dofs_neumann_corner.clear();
    std::vector<unsigned int> tmp;
    tmp.clear();
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (interface_dofs_find_neumann[side].size() != 0)
        for (unsigned int i = 0; i < interface_dofs_find_neumann[side].size(); ++i)
        {
          if (std::find (tmp.begin(), tmp.end(), interface_dofs_find_neumann[side][i]) != tmp.end())
            repeated_dofs_neumann_corner.push_back(interface_dofs_find_neumann[side][i]);
          
          tmp.push_back(interface_dofs_find_neumann[side][i]);
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
    // Functions::FEFieldFunction<dim, DoFHandler<dim>, BlockVector<double>>
    //                                           fe_interface_data(dof1, in_vec);
    // std::map<types::global_dof_index, double> boundary_values_velocity;

    // typename FunctionMap<dim>::type boundary_functions_velocity;

    Functions::FEFieldFunction<dim, BlockVector<double>>
                                              fe_interface_data(dof1, in_vec);
    std::map<types::global_dof_index, double> boundary_values_velocity;

    std::map<types::boundary_id, const Function<dim, double> *>  boundary_functions_velocity;

    constraints.clear();

    for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell;
         ++side)
      if (neighbors[side] >= 0)
        boundary_functions_velocity[side + 1] = &fe_interface_data;
    
    // std::cout << boundary_functions_velocity.size() << std::endl;
    
    proj.project_boundary_values(dof2,
                                 boundary_functions_velocity,
                                 quad,
                                 constraints);

    constraints.close();
    constraints.distribute(out_vec);
  }


} // namespace dd_stokes

#endif // STOKES_MFEDD_UTILITIES_H

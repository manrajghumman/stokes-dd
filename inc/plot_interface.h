/* ---------------------------------------------------------------------
 * Utilities:
 *  - Stokes - related to plotting interface data
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2025 - 2026
 */
#ifndef STOKES_MFEDD_PLOT_INTERFACE_H
#define STOKES_MFEDD_PLOT_INTERFACE_H

// // #include <map>
// // #include <deal.II/lac/block_vector.h>

// // #include <deal.II/lac/sparse_direct.h>

// // #include <deal.II/lac/sparse_ilu.h>

// #include <filesystem>

namespace dd_stokes
{
  using namespace dealii;

  template <int dim>
  void
  plot_approx_function(const unsigned int      &this_mpi,
              const unsigned int               &write_mortar,
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
     for (unsigned int side = 0; side < n_faces_per_cell; ++side){
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
              if (!write_mortar || mortar_degree == 2)
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

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
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
  plot_exact_function( const unsigned int      &this_mpi,
              const unsigned int               &write_mortar,
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
    for (unsigned int side = 0; side < n_faces_per_cell; ++side){
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
              if (!write_mortar || mortar_degree == 2)
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

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
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
  plot_residual_function(const unsigned int      &this_mpi,
                const unsigned int               &write_mortar,
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
    for (unsigned int side = 0; side < n_faces_per_cell; ++side){
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
              if (!write_mortar || mortar_degree == 2)
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
    
    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
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
  plot_total_residual(std::vector<double>          &error,
                      std::ofstream                &file_residual)
  {
    // const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (const auto& value : error)
      file_residual << value << " ";
    // for (int side = 0; side < n_faces_per_cell; ++side)
    //   if (neighbors[side] >= 0)
    //     for (const double& value : error)
    //       file_residual[side] << value << " ";
  }

} // namespace dd_stokes

#endif // STOKES_MFEDD_PLOT_INTERFACE_H
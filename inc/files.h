/* ---------------------------------------------------------------------
 * Utilities:
 *  - General - related to creating, naming and deleting 
 *              files and folders for output
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2025 - 2026
 */
#ifndef STOKES_MFEDD_FILES_H
#define STOKES_MFEDD_FILES_H

#include <filesystem>

namespace dd_stokes
{
  using namespace dealii;

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
      // Check if interface_data/mortar sub directory exists
      if (!fs::exists("../output/interface_data/mortar"))
        fs::create_directory("../output/interface_data/mortar");
      // Check if interface_data/fe sub directory exists
      if (!fs::exists("../output/interface_data/fe"))
        fs::create_directory("../output/interface_data/fe");
      // Remove old files in the interface_data directory
      for (const auto &entry : fs::directory_iterator("../output/interface_data/fe"))
        {
          const auto &path = entry.path();
          std::string filename = path.filename().string();
          if (entry.is_regular_file() && filename[0] != '.')
            fs::remove(entry.path());
        }
      for (const auto &entry : fs::directory_iterator("../output/interface_data/mortar"))
        {
          const auto &path = entry.path();
          std::string filename = path.filename().string();
          if (entry.is_regular_file() && filename[0] != '.')
            fs::remove(entry.path());
      }
      // // If the directory exists, but is not a directory, throw an error
      // if (fs::exists("../output/interface_data/") && !fs::is_directory("../output/interface_data/"))
      //   throw std::runtime_error("../output/interface_data/ is not a directory.");
      
      {
        // create .gitkeep (empty file)
        std::ofstream(std::string("../output/interface_data/mortar") + "/.gitkeep").close();
        // create .gitignore
        std::ofstream gitignore1(std::string("../output/interface_data/mortar") + "/.gitignore");
        gitignore1 << "*\n!.gitkeep\n";
        gitignore1.close();
      }
      {
        // create .gitkeep (empty file)
        std::ofstream(std::string("../output/interface_data/fe") + "/.gitkeep").close();
        // create .gitignore
        std::ofstream gitignore2(std::string("../output/interface_data/fe") + "/.gitignore");
        gitignore2 << "*\n!.gitkeep\n";
        gitignore2.close();
      }
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
    // // If the directory does not exist, create it
    // if (!fs::exists("../output/paraview_data/"))
    //   fs::create_directory("../output/paraview_data/");
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
    // // If the directory does not exist, create it
    // if (!fs::exists("../output/gnuplot_data/"))
    //   fs::create_directory("../output/gnuplot_data/");
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
    // // If the directory does not exist, create it
    // if (!fs::exists("../output/convg_tables/"))
    //   fs::create_directory("../output/convg_tables/");
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
    // // If the directory does not exist, create it
    // if (!fs::exists("../output/convg_table_total/"))
    //   fs::create_directory("../output/convg_table_total/");
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
  Opens a file with the given file(filename), std::ios::trunc will erase previous  
  content if it exists if not std::ios::out will write it to file.
  mortar_flag is used to determine if the file is for mortar or not.
  If you are not using mortar, write only fe files otherwise write 
  mortar files (called twice in cg and gmres if mortar_flag = 1).
  */

  template <int dim>
  void
  name_files(const unsigned int               &this_mpi,
             const bool                       &mortar_flag, //not mortar flag basically do you want to create mortar files or not
             unsigned int                     &cycle,
             std::vector<int>                 &neighbors,
             std::vector<std::ofstream>       &file,
             std::vector<std::ofstream>       &file_exact,
             std::vector<std::ofstream>       &file_residual,
             std::vector<std::ofstream>       &file_y,
             std::vector<std::ofstream>       &file_exact_y,
             std::vector<std::ofstream>       &file_residual_y,
             std::ofstream                    &file_residual_total)
  {
   const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    for (unsigned int side=0; side < n_faces_per_cell; ++side)
    {
      std::string dir = "../output/interface_data/";
      if (mortar_flag)
        dir = dir + "/mortar";
      else
        dir = dir + "/fe";
      if (neighbors[side] >= 0)
      {
        file[side].open(dir + "/lambda" + Utilities::int_to_string(this_mpi)
                    + "_" + Utilities::int_to_string(side, 1)+ "_" 
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 
        file_exact[side].open(dir + "/lambda_exact" + Utilities::int_to_string(this_mpi)                    
                    + "_" + Utilities::int_to_string(side, 1) + "_"
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
        file_residual[side].open(dir + "/residual" + Utilities::int_to_string(this_mpi)
                    + "_" + Utilities::int_to_string(side, 1) + "_"
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 

        file_y[side].open(dir + "/lambda_y" + Utilities::int_to_string(this_mpi)
                    + "_" + Utilities::int_to_string(side, 1) + "_"
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
        file_exact_y[side].open(dir + "/lambda_exact_y" + Utilities::int_to_string(this_mpi)
                    + "_" + Utilities::int_to_string(side, 1) + "_"
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
        file_residual_y[side].open(dir + "/residual_y" + Utilities::int_to_string(this_mpi)
                    + "_" + Utilities::int_to_string(side, 1) + "_"
                    + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc); 
      }
    }
    std::string dir = "../output/interface_data/";
    if (mortar_flag)
      dir = dir + "/mortar";
    else
      dir = dir + "/fe";
    // std::cout << dir << std::endl;
    file_residual_total.open(dir + "/residual_total" + Utilities::int_to_string(cycle, 1) + ".txt", std::ios::out | std::ios::trunc);
  }
  

} // namespace dd_stokes

#endif // STOKES_MFEDD_FILES_H
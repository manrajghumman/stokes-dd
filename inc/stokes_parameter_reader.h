// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 Manraj Singh Ghumman
//
// This file is part for reading parameters in parameters_stokes.prm.
//
// ---------------------------------------------------------------------

#ifndef STOKES_PARAMETER_READER_H
#define STOKES_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
// #include <deal.II/base/parsed_function.h>

namespace dd_stokes
{
  using namespace dealii;

class StokesParameterReader : public Subscriptor
  {
  public:
    StokesParameterReader(ParameterHandler &paramhandler) : prm(paramhandler) {}
    inline void read_parameters(const std::string);
  private:
    inline void declare_parameters();
    ParameterHandler &prm;
  };

  inline void StokesParameterReader::declare_parameters()
  {
    prm.declare_entry("dim", "0",
                      Patterns::Integer());
    prm.declare_entry("degree", "0",
                      Patterns::Integer());
    prm.declare_entry("refinements", "1",
                      Patterns::Integer());
    prm.declare_entry("quad_degree", "11",
                      Patterns::Integer());
    prm.declare_entry("initial_grid_2D", "1, 1",
                        Patterns::Anything(),
                        "Initial grid size for 2D (x, y) "
                        "in number of cells per direction");
    prm.declare_entry("initial_grid_3D", "1, 1, 1",
                        Patterns::Anything(),
                        "Initial grid size for 3D (x, y, z) "
                        "in number of cells per direction");

    prm.enter_subsection("External Boundary Conditions");
    {
        prm.declare_entry("essential_dirichlet", "no",
                      Patterns::Selection("yes|no"));
        prm.declare_entry("boundary_2D", "N, D, D, D",
                      Patterns::Anything(),
                    "Boundary types (N=Neumann, D=Dirichlet)");
        prm.declare_entry("boundary_3D", "N, D, D, D, D, D",
                      Patterns::Anything(),
                    "Boundary types (N=Neumann, D=Dirichlet)");
    }
    prm.leave_subsection();

    prm.enter_subsection("Mortar");
    {
        prm.declare_entry("mortar", "no",
                      Patterns::Selection("yes|no"));
        prm.declare_entry("continuous_mortar", "yes",
                      Patterns::Selection("yes|no"));
        prm.declare_entry("mortar_degree", "2",
                      Patterns::Integer());
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
        prm.declare_entry("iterative_method", "GMRES",
                      Patterns::Selection("CG|GMRES"));
        prm.declare_entry("max_iter", "1000",
                      Patterns::Integer());
        prm.declare_entry("tol", "1e-8",
                      Patterns::Double(0.0));
    }
    prm.leave_subsection();

    prm.enter_subsection("Print Interface Matrix");
    {
        prm.declare_entry("print_interface_matrix", "no",
                      Patterns::Selection("yes|no"));
    }
    prm.leave_subsection();
  }

  inline void StokesParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }
}

#endif //STOKES_PARAMETER_READER_H

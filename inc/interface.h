/* ---------------------------------------------------------------------
 * Utilities:
 *  - Stokes related - related to printing the interface matrix
 * ---------------------------------------------------------------------
 *
 * Author: Manraj Singh Ghumman, University of Pittsburgh, 2025 - 2026
 */
#ifndef STOKES_MFEDD_INTERFACE_H
#define STOKES_MFEDD_INTERFACE_H

namespace dd_stokes
{
  using namespace dealii;

  template <int dim>
  void
  compute_interface_dofs_size(std::vector<double> &interface_dofs_total,
                              const MPI_Comm      &mpi_communicator,
                              const unsigned int  &this_mpi,
                              int                 &interface_dofs_size)
  {
    interface_dofs_size = 0;
    int tmp = 0;

    // If dim == 2, we can use MPI_Allreduce to get the total size of the interface dofs
    // If dim == 3, this is not implemented yet
    if (dim == 2)
    {
      if (this_mpi % 2 == 0)
        tmp = 0;
      else
        tmp = interface_dofs_total.size();
      MPI_Allreduce(&tmp, //sending this data
        &interface_dofs_size, //receiving the result here
        1, //number of elements in alpha and alpha_buffer = 1+1
        MPI_INT, //type of each element
        MPI_SUM, //adding all elements received
        mpi_communicator);
    }
    else
      throw std::runtime_error("dim = 3 not yet implemented!");
  }



  template <int dim>
  void
  copy_matrix_local_to_global(FullMatrix<double>                      &local_matrix,
                              std::vector<std::vector<unsigned int>>  &interface_dofs,
                              int                                     &interface_dofs_size,
                              const unsigned int                      &this_mpi,
                              const unsigned int                      &n_processes,
                              const MPI_Comm                          &mpi_communicator,        
                              FullMatrix<double>                      &interface_matrix)
  {
    // This function is not implemented yet
    // It should copy the local matrix to the global matrix
    // for the case of dim == 3
    if (dim == 3)
      AssertThrow(false, 
        ExcMessage("copy_matrix_local_to_global not implemented for dim = 3!"));
    
    // only 1 interface
    if (n_processes == 2)
    {
      interface_matrix = local_matrix;
      // if (this_mpi == 0)
      // {
      //   std::cout << "\n local_matrix for this_mpi = 0: \n" << std::endl;
      //   local_matrix.print(std::cout);
      // }
      // if (this_mpi == 1)
      // {
      //   std::cout << "\n local_matrix for this_mpi = 1: \n" << std::endl;
      //   local_matrix.print(std::cout);
      // }
    }
    else if (n_processes % 2 == 1) // when prime number of subdomains, all interfaces are vertical interfaces
    {
      int per_interface_dofs, global_row, global_col;
      // global_row = (this_mpi - 1) * per_interface_dofs;
      // global_col = global_row;
      if (this_mpi == 1)
      { 
        per_interface_dofs = interface_dofs[1].size();
        /*
        First interchange R1 <---> R2 and R3 <---> R4
        i.e. eg ordering of interfaces for n_processes = 5
          0 R2 1 R1 2 R4 3 R3 4  ----> 0 R1 1 R2 2 R3 3 R4 4
        */
        // std::cout << "\n local_matrix before exchange: \n" << std::endl;
        // local_matrix.print(std::cout);
        FullMatrix<double> local_matrix_copy;
        local_matrix_copy.reinit(per_interface_dofs * 2, per_interface_dofs * 2);
        // exchange columns 0, 1; 2, 3; 4, 5; ... using local_matrix_copy
        for (int k = 0; k < 2 * per_interface_dofs; ++k)
          if (k % 2 == 0) //even columns
            for (int i = 0; i < 2 * per_interface_dofs; ++i)
            {
              local_matrix_copy(i, k+1) = local_matrix(i, k);
              local_matrix_copy(i, k) = local_matrix(i, k+1);
            }
        // exchange rows 0, 1; 2, 3; 4, 5; ... and put back in local_matrix
        for (int k = 0; k < 2 * per_interface_dofs; ++k)
          if (k % 2 == 0) //even rows
            for (int j = 0; j < 2 * per_interface_dofs; ++j)
            {
              local_matrix(k+1, j) = local_matrix_copy(k, j);
              local_matrix(k, j) = local_matrix_copy(k+1, j);
            }
        // std::cout << "\n local_matrix after exchange: \n" << std::endl;
        // local_matrix.print(std::cout);
        int row, col;
        global_row = (this_mpi - 1) * per_interface_dofs;
        global_col = global_row;
        // std::cout << "\n initial interface_matrix: \n" << std::endl;
        // interface_matrix.print(std::cout);
        /* 
        (1,1) submatrix
        enter the flux differences on the left interface 
        for basis eg lambda = (1,0,...,0) on the left interface
        */
        row = per_interface_dofs;
        col = row;
        for (int j = 0; j < col; ++j)
          for (int i = 0; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j);
        
        // std::cout << "\n after (1,1) block interface_matrix: \n" << std::endl;
        // interface_matrix.print(std::cout);
        /* 
        (2,2) submatrix
        enter the flux differences on the right interface 
        for basis eg lambda = (1,0,...,0) on the right interface
        */
        row = 2 * per_interface_dofs;
        col = row;
        for (int j = per_interface_dofs; j < col; ++j)
          for (int i = per_interface_dofs; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j); 

        // std::cout << "\n after (2,2) block interface_matrix: \n" << std::endl;
        // interface_matrix.print(std::cout);
        /* 
        (1,2) submatrix
        first enter the flux differences from the left interface 
        for lambda = (1,0,...,0) on the right interface
        */
        row = per_interface_dofs;
        col = 2 * per_interface_dofs;
        for (int j = per_interface_dofs; j < col; ++j)
          for (int i = 0; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j); 

        // std::cout << "\n after (1,2) block interface_matrix: \n" << std::endl;
        // interface_matrix.print(std::cout);
        /* 
        (2,1) submatrix
        now enter the flux differences from the right interface 
        for lambda = (1,0,...,0) on the left interface
        */
        row = 2 * per_interface_dofs;
        col = per_interface_dofs;
        for (int j = 0; j < col; ++j)
          for (int i = per_interface_dofs; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j); 

        // std::cout << "\n after (2,1) block interface_matrix: \n" << std::endl;
        // interface_matrix.print(std::cout);
      }
      else if (this_mpi < n_processes - 1 && this_mpi > 1) // if not the second last subdomain
      {
        per_interface_dofs = interface_dofs[1].size();
        /*
        First interchange R1 <---> R2 and R3 <---> R4
        i.e. eg ordering of interfaces for n_processes = 5
          0 R2 1 R1 2 R4 3 R3 4  ----> 0 R1 1 R2 2 R3 3 R4 4
        */
        FullMatrix<double> local_matrix_copy;
        local_matrix_copy.reinit(per_interface_dofs * 2, per_interface_dofs * 2);
        // exchange columns 0, 1; 2, 3; 4, 5; ... using local_matrix_copy
        for (int k = 0; k < 2 * per_interface_dofs; ++k)
          if (k % 2 == 0) //even columns
            for (int i = 0; i < 2 * per_interface_dofs; ++i)
            {
              local_matrix_copy(i, k+1) = local_matrix(i, k);
              local_matrix_copy(i, k) = local_matrix(i, k+1);
            }
        // exchange rows 0, 1; 2, 3; 4, 5; ... and put back in local_matrix
        for (int k = 0; k < 2 * per_interface_dofs; ++k)
          if (k % 2 == 0) //even rows
            for (int j = 0; j < 2 * per_interface_dofs; ++j)
            {
              local_matrix(k+1, j) = local_matrix_copy(k, j);
              local_matrix(k, j) = local_matrix_copy(k+1, j);
            }
        int row, col;
        global_row = (this_mpi - 1) * per_interface_dofs;
        global_col = global_row;
        /* 
        NO (1,1) submatrix!
        (2,2) submatrix
        enter the flux differences on the right interface 
        for basis eg lambda = (1,0,...,0) on the right interface
        */
        row = 2 * per_interface_dofs;
        col = row;
        for (int j = per_interface_dofs; j < col; ++j)
          for (int i = per_interface_dofs; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j); 
        /* 
        (1,2) submatrix
        first enter the flux differences from the left interface 
        for lambda = (1,0,...,0) on the right interface
        */
        row = per_interface_dofs;
        col = 2 * per_interface_dofs;
        for (int j = per_interface_dofs; j < col; ++j)
          for (int i = 0; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j); 
        /* 
        (2,1) submatrix
        now enter the flux differences from the right interface 
        for lambda = (1,0,...,0) on the left interface
        */
        row = 2 * per_interface_dofs;
        col = per_interface_dofs;
        for (int j = 0; j < col; ++j)
          for (int i = per_interface_dofs; i < row; ++i)
            interface_matrix(global_row + i,global_col + j) += local_matrix(i,j);
      }
      else // last subdomain or first subdomain
      {
        // DO NOTHING HERE, ALL DATA ALREADY ADDED BY PREVIOUS DOMAINS
      }
    }
    else // when even number of subdomains
    {
      AssertThrow(false, 
        ExcMessage("print_interface_matrix with n_processes even not yet implemented!"));
    }
    std::vector<double> matrix_data_send(interface_dofs_size * interface_dofs_size, 0.0);
    std::vector<double> matrix_data_recv(interface_dofs_size * interface_dofs_size, 0.0);
    for (int i = 0; i < interface_dofs_size; ++i)
      for (int j = 0; j < interface_dofs_size; ++j)
        matrix_data_send[i * interface_dofs_size + j] = interface_matrix(i,j);
    // Now we need to sum the matrix data across all processes
    MPI_Allreduce(&matrix_data_send[0], //sending this data
                  &matrix_data_recv[0], //receiving the result here
                  interface_dofs_size * interface_dofs_size, //number of elements in alpha and alpha_buffer
                  MPI_DOUBLE, //type of each element
                  MPI_SUM, //adding all elements received
                  mpi_communicator);
    // Now we copy the data back to the interface_matrix
    for (int i = 0; i < interface_dofs_size; ++i)
      for (int j = 0; j < interface_dofs_size; ++j)
        interface_matrix(i,j) = matrix_data_recv[i * interface_dofs_size + j];
  }


  //---------------old deprecated functions------------------
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
                          }
              }       
    }
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
                          }
              }
    }
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

} // namespace dd_stokes

#endif // STOKES_MFEDD_INTERFACE_H
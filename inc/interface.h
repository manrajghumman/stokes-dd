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

} // namespace dd_stokes

#endif // STOKES_MFEDD_INTERFACE_H
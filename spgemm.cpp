#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // TODO: Write your code here
    int row_rank, row_size;
    MPI_Comm_rank(row_comm,&row_rank);
    MPI_Comm_size(row_comm,&row_size);
    int column_rank, column_size;
    MPI_Comm_rank(col_comm,&column_rank);
    MPI_Comm_size(col_comm,&column_size);



    std::map<std::pair<int, int>, int> local_C;

    //grid dimension the same as the row_size
    for (int i=0;i<row_size;i++)
    {
        std::vector<std::pair<std::pair<int,int>, int>> A_i;
            if (row_rank==i)
            {
                A_i=A;
            }

            int A_count = (row_rank == i) ? A_i.size() : 0; 
            //bcast the size
            MPI_Bcast(&A_count, 1, MPI_INT, i, row_comm);
            
            if (row_rank != i) {
                A_i.resize(A_count);
            }
            MPI_Bcast(A_i.data(), A_count * sizeof(A[0]), MPI_BYTE, i, row_comm);


            std::vector<std::pair<std::pair<int,int>, int>> B_i;
            if (column_rank==i)
            {
                B_i=B;
            }

            int B_count = (column_rank == i) ? B_i.size() : 0; 
            //bcast the size
            MPI_Bcast(&B_count, 1, MPI_INT, i, col_comm);
            
            if (column_rank != i) {
                B_i.resize(A_count);
            }
            MPI_Bcast(B_i.data(), A_count * sizeof(A[0]), MPI_BYTE, i, col_comm);

            for (const auto &a_entry : A_i) {
                int a_row = a_entry.first.first;    // Global row index of A's entry
                int a_col = a_entry.first.second;     // Global column index of A's entry
                int a_val = a_entry.second;
                for (const auto &b_entry : B_i) {
                    int b_row = b_entry.first.first;  // Global row index of B's entry
                    int b_col = b_entry.first.second;   // Global column index of B's entry
                    int b_val = b_entry.second;
                    if (a_col == b_row) { // Valid multiplication condition
                        std::pair<int,int> pos(a_row, b_col);
                        if (local_C.find(pos) == local_C.end()) {
                            local_C[pos] = times(a_val, b_val);
                        } else {
                            local_C[pos] = plus(local_C[pos], times(a_val, b_val));
                        }
                    }
                }
            }

        
    }
    C.clear();
    for (const auto &entry : local_C) {
        C.push_back({entry.first, entry.second});
    }
}

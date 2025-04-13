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
    std::vector<std::pair<std::pair<int,int>, int>> A_i;
    std::vector<std::pair<std::pair<int,int>, int>> B_i;
    //grid dimension the same as the row_size
    for (int i=0;i<row_size;i++)
    {   
        
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


            
            if (column_rank==i)
            {
                B_i=B;
            }

            int B_count = (column_rank == i) ? B_i.size() : 0; 
            //bcast the size
            MPI_Bcast(&B_count, 1, MPI_INT, i, col_comm);
            
            if (column_rank != i) {
                B_i.resize(B_count);
            }
            MPI_Bcast(B_i.data(), B_count * sizeof(B[0]), MPI_BYTE, i, col_comm);


            // Sort A_i and B_i by their join keys.
            size_t i_ptr = 0, j_ptr = 0;
            while (i_ptr < A_i.size() && j_ptr < B_i.size()) {
                int a_col = A_i[i_ptr].first.second;
                int b_row = B_i[j_ptr].first.first;
                if (a_col < b_row) {
                    ++i_ptr;
                } else if (a_col > b_row) {
                    ++j_ptr;
                } else {
                    // Match found: a_col == b_row.
                    // You might need to loop over all matching values in one vector.
                    // Process matching elements:
                    for (size_t k = i_ptr; k < A_i.size() && A_i[k].first.second == a_col; ++k) {
                        for (size_t l = j_ptr; l < B_i.size() && B_i[l].first.first == b_row; ++l) {
                            int a_val = A_i[k].second;
                            int b_val = B_i[l].second;
                            int a_row = A_i[k].first.first;
                            int b_col = B_i[l].first.second;
                            std::pair<int, int> pos(a_row, b_col);
                            if (local_C.find(pos) == local_C.end()) {
                                local_C[pos] = times(a_val, b_val);
                            } else {
                                local_C[pos] = plus(local_C[pos], times(a_val, b_val));
                            }
                        }
                    }
                    // Advance pointers past this join key.
                    while (i_ptr < A_i.size() && A_i[i_ptr].first.second == a_col) i_ptr++;
                    while (j_ptr < B_i.size() && B_i[j_ptr].first.first == b_row) j_ptr++;
                }
            }

        
    }
    C.clear();
    for (const auto &entry : local_C) {
        C.push_back({entry.first, entry.second});
    }
}

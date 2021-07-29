#include "utils/ceresfunc.h"

void* ThreadsConstructA(void* threadsstruct)
{
  ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
  for (auto it : p->sub_factors)
  {
    for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
    {
      int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
      int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
      Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
      for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
      {
        int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
        int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
        Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
        if (i == j)
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
        else
        {
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
          p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
        }
      }
      p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
    }
  }
  return threadsstruct;
}


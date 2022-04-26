#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <rmos_smtx_def.h>
#include "cuda_defs.h"


__device__ double atomicDadd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
      __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ static double atomicDmax(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
      __double_as_longlong(max(val, __longlong_as_double(assumed))));

  } while (assumed != old);
  return __longlong_as_double(old);
}


__global__ void rmosCudaDMemset_kernel(
  double* x, const double value, const int size) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < size) {
    x[tid] = value;

    tid += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDMemset(
  const cudaStream_t sid,
  double* x,
  const double value,
  const int size) {

  size_t block_x = (size > 1024) ? 1024 : size;
  size_t gridCols = (size + block_x - 1) / block_x;
  rmosCudaDMemset_kernel << <1, block_x, 0, sid >> > (x, value, size);
}

__global__ void rmosCudaIMemset_kernel(
  int* x, const int value, const int size) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < size) {
    x[tid] = value;

    tid += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaIMemset(
  const cudaStream_t sid,
  int* x,
  const int value,
  const int size) {

  size_t block_x = (size > 1024) ? 1024 : size;
  size_t gridCols = (size + block_x - 1) / block_x;
  rmosCudaIMemset_kernel << <gridCols, block_x, 0, sid >> > (x, value, size);
}

__global__ void rmosCudaDvhadamard_kernel(
  const double* __restrict__ a,
  const double* __restrict__ b,
  double* c,
  const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double tmp_a, tmp_b;

  while (tid < size) {
    tmp_a = a[tid];
    tmp_b = b[tid];

    c[tid] = tmp_a * tmp_b;

    tid += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDvhadamard(
  const cudaStream_t sid,
  const double* __restrict__ a,
  const double* __restrict__ b,
  double* c,
  const int size) {

  size_t block_x = (size > 1024) ? 1024 : size;
  rmosCudaDvhadamard_kernel << <1, block_x, 0, sid >> > (a, b, c, size);
}

__global__ void rmosCudaDvsub_kernel(
  const double* __restrict__ a,
  const double* __restrict__ b,
  double* c,
  const int size) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double tmp_a, tmp_b;

  while (tid < size) {
    tmp_a = a[tid];
    tmp_b = b[tid];

    c[tid] = tmp_a - tmp_b;

    tid += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDvsub(
  const cudaStream_t sid,
  const double* __restrict__ a,
  const double* __restrict__ b,
  double* c,
  const int size) {

  size_t block_x = (size > 1024) ? 1024 : size;
  rmosCudaDvsub_kernel << <1, block_x, 0, sid >> > (a, b, c, size);
}

// This kernel is for performing vector permutation.
// x (input) -> vector
// y (output) -> vector
// perm (input) -> vector
// n (input) -> vector size
__global__ void rmosCudaDdnPerm_kernel(
  const double* __restrict__ x,
  double* y,
  const int* __restrict__ perm,
  const int n) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < n) {
    y[perm[tid]] = x[tid];

    tid += blockDim.x * gridDim.x;
    __syncthreads();
  }
}
extern "C" void rmosCudaDdnPerm(
  const cudaStream_t sid,
  const double* __restrict__ x,
  double* y,
  const int* __restrict__ perm,
  const int n) {

  size_t block_x = (n > 1024) ? 1024 : n;
  size_t gridCols = (n + block_x - 1) / block_x;
  rmosCudaDdnPerm_kernel << <gridCols, block_x, 0, sid >> > (x, y, perm, n);
}

// Performing Batched COO SpMV (ILP applied), (multiple A, single B)
// The gradient matrix, which are used in this kernel, is sparse similar that has 4 number of none zero values per each row.
// Therefore, output vector, which is initilized using gradient matrix, is also having sparsity.
// Sparse vector can be extended ELLpack matrix by batch scheme
__global__ void rmosCudaDcooSpMVSpYBatched_kernel(
  const int n_batch,                     // number of matrices
  const int nnz,                         // number of none zero value
  const double* __restrict__ nzval,      // none zero values of batch COO matrix
  const int* __restrict__ colind,        // column indices of batch COO matrix
  const int* __restrict__ rowind,        // row indices of batch COO matrix
  const double* __restrict__ x,          // dense vector (n_batch)
  double* y_nzval,                       // (output) none zero values of ELLpack matrix
  int* y_ind) {                          // (output) index matrix of ELLpack matrix

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  double nzval_tmp[4];  // register memory to store the none zero values of y
  int index_tmp[4];     // register memory to store the column index of none zero values

  double tmp;
  int cur_row;
  int index;

  while (tid < n_batch) {
    int cnt = 0;

    // unrolling the loop to improve instruction efficiency
#pragma unrolling
    for (int i = 0; i < 4; i++) {
      nzval_tmp[i] = 0;
      index_tmp[i] = -1;
    }

    for (int i = 0; i < nnz; i++) {

      tmp = nzval[(n_batch * i) + tid] * x[colind[(n_batch * i) + tid]];
      cur_row = rowind[(n_batch * i) + tid];

      // Find column location in buffer
      index = -1;
      for (int j = 0; j < 4; j++) {
        // If "cur_row" is already recorded in buffer, store its index.
        if (index_tmp[j] == cur_row) {
          index = j;
          break;
        }
      }
      __syncthreads();

      if (index != -1) {
        nzval_tmp[index] += tmp;
        index_tmp[index] = cur_row;
      }
      else { // If "cur_row" is not recorded in buffer, add it and cnt++
        nzval_tmp[cnt] += tmp;
        index_tmp[cnt] = cur_row;
        cnt++;
      }
      __syncthreads();
    }

    // Copy ELL matrix from register memory to global memory.
    // This is because for minimizing the memory latency.
    for (int i = 0; i < cnt; i++) {
      y_nzval[(n_batch * i) + row] = nzval_tmp[i];
    }
    for (int i = 0; i < cnt; i++) {
      y_ind[(n_batch * i) + row] = index_tmp[i];
    }


    row += blockDim.x * gridDim.x;
    __syncthreads();
  }
}
extern "C" void rmosCudaDcooSpMVSpYBatched(
  const cudaStream_t sid,
  const int n_batch,
  const int nnz,
  const double* __restrict__ nzval,
  const int* __restrict__ colind,
  const int* __restrict__ rowind,
  const double* __restrict__ x,
  double* y_nzval,
  int* y_ind) {

  size_t block_x = (n_batch > 128) ? 128 : n_batch;
  size_t gridCols = (n_batch + block_x - 1) / block_x;
  rmosCudaDcooSpMVSpYBatched_kernel << <gridCols, block_x, 0, sid >> > (
    n_batch, nnz,
    nzval, colind, rowind, x, y_nzval, y_ind);
}


// Performing Batched ELL SpMV (ILP applied)
// A is ELLpack matrix that has 4 none zero values per each row
__global__ void rmosCudaDellSpMV_kernel(
  const int m,                          // number of rows of matrix A
  const int n,                          // number of columns of matrix A
  const int nnz,                        // none zero value sper each row of matrix A
  const double* __restrict__ nzval,     // none zero values of matrix A
  const int* __restrict__ ind,          // indices of matrix A
  const double* __restrict__ x,         // dense vector
  double* y) {                          // (output) dense vector

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  double dot;
  int col;
  double val;

  while (row < m) {
    dot = 0;

    for (int i = 0; i < nnz; i++) {
      // getting the elements from the global memory
      col = ind[(m * i) + tid];   // column major
      val = nzval[(m * i) + tid]; // column major

      if (val != 0) {
        dot += val * x[col];
      }
    }

    y[tid] = dot;
    row += blockDim.x * gridDim.x;
    __syncthreads();
  }
}
extern "C" void rmosCudaDellSpMV(
  const cudaStream_t sid,
  const int m, // batch
  const int n,
  const int nnz,
  const double* __restrict__ nzval, // a
  const int* __restrict__ ind,  // a
  const double* __restrict__ x, // x
  double* y) {

  size_t block_x = (m > 128) ? 128 : m;
  size_t gridCols = (m + block_x - 1) / block_x;
  rmosCudaDellSpMV_kernel << <gridCols, block_x, 0, sid >> > (
    m, n, nnz, nzval, ind, x, y);
}

__global__ void rmosCudaDdnEVGupdBatched_kernel(
  double* d_ev_gradient,
  const double* __restrict__ d_vec1_result,
  const double* __restrict__ d_vec2_result,
  const double* __restrict__ d_lambda,
  const int n_batch) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < n_batch) {
    d_ev_gradient[tid] =
      d_vec1_result[tid] - (*d_lambda) * d_vec2_result[tid];

    tid += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDdnEVGupdBatched(
  const cudaStream_t sid,
  double* d_ev_gradient,
  const double* __restrict__ d_vec1_result,
  const double* __restrict__ d_vec2_result,
  const double* d_lambda,
  const int n_batch) {

  size_t block_x = (n_batch > 1024) ? 1024 : n_batch;

  rmosCudaDdnEVGupdBatched_kernel << <1, block_x, 0, sid >> > (
    d_ev_gradient, d_vec1_result, d_vec2_result, d_lambda, n_batch);

}

// Setting the max magnitude to zero
// "max_ef_value_ind" is already finded from CPU
// To minimize latency due to matrix permuting, permuting is applied at initilize step.
__global__ void rmosCudaDdnMaxInd2Zero_kernel(
  const int n_batch,                 // number of rows of d_mu
  const int n,                       // number of columns of d_mu
  const int max_ef_value_ind,        // index of max magnitude
  const int* __restrict__ perm,      // permutation matrix
  double* d_mu) {                    // (in & out) dense matrix (n_batch, n)

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < n_batch) {
    d_mu[(tid * n) + perm[max_ef_value_ind]] = 0;

    tid += blockDim.x * gridDim.x;
    __syncthreads();
  }

}
extern "C" void rmosCudaDdnMaxInd2Zero(
  const cudaStream_t sid,
  const int n_batch,
  const int n,
  const int max_ef_value_ind,
  const int* __restrict__ perm,
  double* d_mu) {

  size_t block_x = (n_batch > 256) ? 256 : n_batch;
  size_t gridCols = (n_batch + block_x - 1) / block_x;
  rmosCudaDdnMaxInd2Zero_kernel << <gridCols, block_x, 0, sid >> > (n_batch, n, max_ef_value_ind, n, perm, d_mu);
}


// Computing the Outer product (ILP applied)
// Using the property of the outer product, 
// this kernel minizes memory latency by storing and using input vector in shared and register memory.
__global__ void rmosCudaDnOuterProduct_kernel(
  const int n_batch,                           // number of rows of d_mu
  const int n,                                 // number of columns of d_mu
  const int ilp,                               // Iteration step for ILP
  const double* __restrict__ d_ev_gradient,    // dense vector (n_batch)
  const double* __restrict__ d_f,              // dense vector (n)
  double* d_mu) {                              // dense matrix (n_batch, n)

  double col;
  extern __shared__ double shrm_mem[];

  for (int bid_y = blockIdx.y * ilp; bid_y < n_batch; bid_y += ilp * gridDim.y) {
    // Load some elements from left vector and store them in shared memory
    if (threadIdx.x < ilp)
      shrm_mem[threadIdx.x] = d_ev_gradient[bid_y + threadIdx.x];
    __syncthreads();

    for (int tid_x = blockIdx.x * blockDim.x + threadIdx.x; tid_x < n; tid_x += blockDim.x * gridDim.x) {
      // Load a element from right vector and store them in register memory
      col = d_f[tid_x];

      // because "tmp" is shared memory, latency is minimized by broad casting.
      for (int i = 0; i < ilp; i++)
        d_mu[(bid_y + i) * n + tid_x] = shrm_mem[i] * col;

    }
  }
}
extern "C" void rmosCudaDnOuterProduct(
  const cudaStream_t sid,
  const int n_batch,
  const int n,
  const double* __restrict__ d_ev_gradient,
  const double* __restrict__ d_f,
  double* d_mu) {
  size_t blockCols = 256;
  int ilp = 4;
  size_t gridCols = (n + blockCols - 1) / blockCols;
  size_t gridRows = (n_batch + (ilp * 1) - 1) / (ilp * 1);
  dim3 blockSize(gridCols, gridRows);  // 10k 4096 --> 280ms

  rmosCudaDnOuterProduct_kernel << <blockSize, blockCols, sizeof(double)* ilp, sid >> > (
    d_mu, d_ev_gradient,
    n, d_f, n_batch, ilp);
}

// Computing matrix addition between dense matrix and ELLpack matrix.
// Y = Y(dense) - a(ELL) + beta * b(ELL)
// To minimize latency due to matrix permuting, permuting is applied at initilize step.
__global__ void rmosCudaDcooaSpXSpYBatched3_kernel(
  const int m,                           // number of rows of y
  const int n,                           // number of columns of y
  const int nnz,
  const double* __restrict__ a_nzval,    // ELL matrix (n_edge, n_vert)
  const int* __restrict__ a_ind,
  const int lda,                         // n_edge
  const double* beta,                    // scholar for Y
  const double* __restrict__ b_nzval,    // ELL matrix (n_edge, n_vert)
  const int* __restrict__ b_ind,
  const int k,                           // adjusted batch size
  double* __restrict__ y,                // (in & out)dense matrix (n_batch, n_vert)
  const int ldy,                         // batch size
  const int cur_batch_i,                 // current iteration
  const int* __restrict__ perm)
{

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  double tmp = 0;

  int col;
  double val;

  while (row < k) {

    for (int i = 0; i < nnz; i++) {

      col = a_ind[(lda * i) + (ldy * cur_batch_i) + row];
      val = a_nzval[(lda * i) + (ldy * cur_batch_i) + row];

      if (val != 0) {
        y[(row * m) + perm[col]] -= val;
      }

      col = b_ind[(lda * i) + (ldy * cur_batch_i) + row];
      val = b_nzval[(lda * i) + (ldy * cur_batch_i) + row];

      if (val != 0) {
        tmp = *beta * val;
        y[(row * m) + perm[col]] += tmp;
      }
    }
    row += blockDim.x * gridDim.x;
    __syncthreads();
  }
}
extern "C" void rmosCudaDcooaSpXSpYBatched3(
  const cudaStream_t sid,
  const int m,
  const int n,
  const int nnz,
  const double* __restrict__ a_nzval,
  const int* __restrict__ a_ind,
  const int lda,
  const double* beta,
  const double* __restrict__ b_nzval,
  const int* __restrict__ b_ind,
  const int k,
  double* __restrict__ y,
  const int ldy,  //n_batch
  const int cur_batch_i,
  const int* __restrict__ perm
) {
  size_t blockCols = 256;
  size_t gridCols = (m + blockCols - 1) / blockCols;

  rmosCudaDcooaSpXSpYBatched3_kernel << <gridCols, blockCols, 0, sid >> > (
    m, n, nnz, a_nzval, a_ind, lda, beta, b_nzval, b_ind, k, y, ldy, cur_batch_i, perm);
}

// Convert row major to column major
__global__ void rmosCudaDgemIDX2C_kernel(
  const int m,                   // number of rows of x
  const int n,                   // number of columns of x 
  const double* __restrict__ x,  // dense matrix (m, n), row major
  double* y) {                   // dense matrix (m, n), column major

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  while (tid_x < m) {
    tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    while (tid_y < n) {
      y[IDX2C(tid_x, tid_y, m)] = x[(tid_x * n) + tid_y];

      tid_y += blockDim.y * gridDim.y;
    }
    tid_x += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDgemIDX2C(
  const cudaStream_t sid,
  const int m,
  const int n,
  const double* __restrict__ x,
  double* y) {
  dim3 blockSize(8, 32, 1);
  size_t gridCols = (m + blockSize.x - 1) / blockSize.x;
  size_t gridRows = (n + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(gridCols, gridRows, 1);

  rmosCudaDgemIDX2C_kernel << <gridSize, blockSize, 0, sid >> > (m, n, x, y);

}

// Convert column major to row major
__global__ void rmosCudaDgemIDX2R_kernel(
  const int m,                   // number of rows of x
  const int n,                   // number of columns of x 
  const double* __restrict__ x,  // dense matrix (m, n), column major
  double* y) {                   // dense matrix (m, n), row major

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  while (tid_x < n) {
    tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    while (tid_y < m) {
      y[(tid_y * n) + tid_x] = x[tid_x * m + tid_y];

      tid_y += blockDim.y * gridDim.y;
    }
    tid_x += blockDim.x * gridDim.x;
  }
}
extern "C" void rmosCudaDgemIDX2R(
  const cudaStream_t sid,
  const int m,
  const int n,
  const double* __restrict__ x,
  double* y) {

  const dim3 blockSize(8, 32, 1);

  size_t gridCols = (n + blockSize.x - 1) / blockSize.x;
  size_t gridRows = (m + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(gridCols, gridRows, 1);

  rmosCudaDgemIDX2R_kernel << <gridSize, blockSize, 0, sid >> > (m, n, x, y);
}

// Permuting matrix

__global__ void rmodCudaDgemPerm_kernel(
  const int n_batch,              // number of rows of x
  const int n,                    // number of columns of x
  const double* __restrict__ x,   // dense matrix (m, n)
  double* y,                      // dense matrix (m, n)
  const int* __restrict__ perm) { // permutation vector

  int y_range;
  int ilp = 3;
  int perm_ind;
  double col;

  extern __shared__ double shrd_mem[];
  for (int tid_x = blockIdx.x * blockDim.x + threadIdx.x; tid_x < n; tid_x += blockDim.x * gridDim.x) {
    perm_ind = perm[tid_x];

    for (int bid_y = blockIdx.y * ilp; bid_y < n_batch; bid_y += ilp * gridDim.y) {
      y_range = ilp;
      if (n_batch - bid_y < ilp) y_range = n_batch - bid_y;

      for (int i = 0; i < y_range; i++) {
        shrd_mem[i * blockDim.x + threadIdx.x] = x[(bid_y + i) * n + perm_ind];
      }
      __syncthreads();

      for (int i = 0; i < y_range; i++) {
        y[(bid_y + i) * n + tid_x] = shrd_mem[i * blockDim.x + threadIdx.x];
      }
    }
  }
}
extern "C" void rmodCudaDgemPerm(
  const cudaStream_t sid,
  const int n_batch,
  const int n,
  const double* __restrict__ x,
  double* y,
  const int* __restrict__ perm) {
  size_t blockCols = 256;
  int ilp = 3;
  size_t gridCols = (n + blockCols - 1) / blockCols;
  size_t gridRows = (n_batch + ilp - 1) / ilp;
  dim3 blockSize(gridCols, gridRows);  // 10k 4096 --> 280ms

  rmodCudaDgemPerm_kernel << <blockSize, blockCols, sizeof(double)* blockCols* ilp, sid >> > (n_batch, n, x, y, perm);
}


/////////////////////////////////////////////////////////
//
//
// Data term optimization
//
//
/////////////////////////////////////////////////////////

// Calculating L2 distance of feature between source and target surfaces
__global__ void rmosCudaDdnDataGrad_kernel(
  const int n_vert,                              // number of vertices
  const int n_face,                              // number of faces
  const double* feat_weight,
  const double* __restrict__ src_feat,           // dense vector(n_vert), feature of source surface
  const double* __restrict__ trg_feat,           // dense vector(n_vert), feature of target surface
  const double* __restrict__ trg_feat_grad,	     // dense matrix(n_vert, 3), intrinsic feature gradient of target surface, column major
  const double* __restrict__ src_weight_fact,    // dense vector(n_vert), feature weight
  const double* __restrict__ s2t_data_coef_vec,  // dense matrix(n_vert, 3), coefficient matrix of pullback metric, column major
  const int* __restrict__ s2t_data_map,          // dense vector(n_vert), source to target pullback metric
  const int* __restrict__ trg_face_vert_map,     // dense matrix(n_face, 3), vectex map of each target face, column major
  double* data_coef_grad,                        // dense vector(n_vert), this is used for calcularing the data error
  double* src_data_grad) {                       // dense matrix(n_vert, 3), distance between source and target surface

  double data_grad_tmp[3];
  double coef_grad_tmp;
  double src_weight_tmp;

  int ind_tmp;
  int fv_ind_tmp;

  for (int tid_x = blockIdx.x * blockDim.x + threadIdx.x; tid_x < n_vert; tid_x += blockDim.x * gridDim.x) {

    // Loop unrolling
    coef_grad_tmp = src_feat[tid_x];
    ind_tmp = s2t_data_map[tid_x];    // map -> n_vert to n_face

    src_weight_tmp = src_weight_fact[tid_x];

    // Loop unrolling
    data_grad_tmp[0] = (*feat_weight) * src_weight_tmp;
    data_grad_tmp[1] = (*feat_weight) * src_weight_tmp;
    data_grad_tmp[2] = (*feat_weight) * src_weight_tmp;

    // calculating the feature gradient between source vertex and three vertices of mapped target face.
    for (int i = 0; i < 3; i++) {
      fv_ind_tmp = trg_face_vert_map[i * n_face + ind_tmp];
      coef_grad_tmp -=
        s2t_data_coef_vec[i * n_vert + tid_x]
        * trg_feat[fv_ind_tmp];
    }
    coef_grad_tmp *= 2;

    // Loop unrolling
    data_grad_tmp[0] *= coef_grad_tmp * trg_feat_grad[0 * n_face + ind_tmp];
    data_grad_tmp[1] *= coef_grad_tmp * trg_feat_grad[1 * n_face + ind_tmp];
    data_grad_tmp[2] *= coef_grad_tmp * trg_feat_grad[2 * n_face + ind_tmp];

    // Loop unrolling
    src_data_grad[0 * n_vert + tid_x] = data_grad_tmp[0];
    src_data_grad[1 * n_vert + tid_x] = data_grad_tmp[1];
    src_data_grad[2 * n_vert + tid_x] = data_grad_tmp[2];

    data_coef_grad[tid_x] = (*feat_weight)
      * src_weight_tmp
      * powf((coef_grad_tmp / 2), 2);

    __syncthreads();
  }
}
extern "C" void rmosCudaDdnDataGrad(
  cudaStream_t sid,
  const int n_vert,
  const int n_face,
  const double* feat_weight,
  const double* __restrict__ src_feat,
  const double* __restrict__ trg_feat,
  const double* __restrict__ trg_feat_grad,	// Column major
  const double* __restrict__ src_weight_fact,
  const double* __restrict__ s2t_data_coef_vec,  // Column major 필요
  const int* __restrict__ s2t_data_map,
  const int* __restrict__ trg_face_vert_map,  // Column major 필요
  double* data_coef_grad,  // (n_vert)
  double* src_data_grad) {

  size_t blockCols = 128;
  size_t gridCols = (n_vert + blockCols - 1) / blockCols;

  rmosCudaDdnDataGrad_kernel << <gridCols, blockCols, 0, sid >> > (
    n_vert, n_face, feat_weight, src_feat, trg_feat, trg_feat_grad,
    src_weight_fact, s2t_data_coef_vec, s2t_data_map, trg_face_vert_map,
    data_coef_grad, src_data_grad);
}


// Calculating distance with NCC of feature between source and target surfaces
__global__ void rmosCudaDdnNCCDataGrad_kernel(
  const int n_vert,                              // number of vertices
  const int n_face,                              // number of faces
  const double* feat_weight,
  const double* __restrict__ src_feat,           // dense vector(n_vert), feature of source surface
  const double* __restrict__ trg_feat,           // dense vector(n_vert), feature of target surface
  const double* __restrict__ trg_feat_grad,	     // dense matrix(n_vert, 3), intrinsic feature gradient of target surface, column major
  const double* __restrict__ src_weight_fact,    // dense vector(n_vert), feature weight
  const double* __restrict__ s2t_data_coef_vec,  // dense matrix(n_vert, 3), coefficient matrix of pullback metric, column major
  const int* __restrict__ s2t_data_map,          // dense vector(n_vert), source to target pullback metric
  const int* __restrict__ trg_face_vert_map,     // dense matrix(n_face, 3), vectex map of each target face, column major
  // CSR matrix (n_vert, n_vert), 1-ring neighbors of source vertex
  const int* __restrict__ src_vn_nzval,          // dense vector(nnz), ID of 1-ring neighbor of source vertex
  const int* __restrict__ src_vn_colind,         // dense vector(nnz), order of 1-ring neighbor of source vertex
  const int* __restrict__ src_vn_rowptr,         // dense vector(n_vert + 1), number of 1-ring neighbors of each vertex
  // CSR matrix (n_vert, n_vert), 1-ring neighbors of target vertex
  const int* __restrict__ trg_vn_nzval,
  const int* __restrict__ trg_vn_colind,
  const int* __restrict__ trg_vn_rowptr,
  double* data_coef_grad,                        // dense vector(n_vert), this is used for calcularing the data error
  double* src_data_grad,                         // dense matrix(n_vert, 3), distance between source and target surface
  double* src_coef_ncc) {

  double data_grad_tmp = 0;
  double coef_grad_tmp = 0;

  double src_weight_tmp;
  int ind_tmp;
  int fv_ind_tmp;

  int local_nnz;
  int trg_local_nnz;

  double sum = 0;
  double src_mean = 0;
  double src_std = 0;
  double trg_mean = 0;
  double trg_std = 0;

  // register variables to store gradient of NCC of three vertices on mapped face.
  double ncc_term[3];

  for (int tid_x = blockIdx.x * blockDim.x + threadIdx.x; tid_x < n_vert; tid_x += blockDim.x * gridDim.x) {

    // initilize variables
    double ncc = 0;
    sum = 0;
    coef_grad_tmp = 0;
    local_nnz = src_vn_rowptr[tid_x + 1] - src_vn_rowptr[tid_x];
    ind_tmp = s2t_data_map[tid_x];
    src_weight_tmp = src_weight_fact[tid_x];

    // calculating the feature mean and std of 1-ring neighbors of source vertex
    for (int i = 0; i < local_nnz; i++) {
      sum += src_feat[src_vn_nzval[src_vn_rowptr[tid_x] + i]];
    }
    src_mean = sum / local_nnz;

    sum = 0;
    for (int i = 0; i < local_nnz; i++) {
      sum += powf(src_feat[src_vn_nzval[src_vn_rowptr[tid_x] + i]] - src_mean, 2);
    }
    src_std = sum;
    __syncthreads();

    // calculating the feature mean and std of 1-ring neighbors of three vertices of mapped target face
    for (int i = 0; i < 3; i++) {
      fv_ind_tmp = trg_face_vert_map[i * n_face + ind_tmp];
      trg_local_nnz = trg_vn_rowptr[fv_ind_tmp + 1] - trg_vn_rowptr[fv_ind_tmp];

      sum = 0;
      for (int j = 0; j < trg_local_nnz; j++) {
        sum += trg_feat[trg_vn_nzval[trg_vn_rowptr[fv_ind_tmp] + j]];
      }
      trg_mean = sum / trg_local_nnz;

      sum = 0;
      for (int j = 0; j < trg_local_nnz; j++) {
        sum += powf(trg_feat[trg_vn_nzval[trg_vn_rowptr[fv_ind_tmp] + j]] - trg_mean, 2);
      }
      trg_std = sum;

      // adjusting the number of neighbors
      if (local_nnz > trg_local_nnz) local_nnz = trg_local_nnz;
      __syncthreads();

      sum = 0;
      // calculating the normalized difference between source and target
      for (int j = 0; j < local_nnz; j++) {
        sum +=
          (src_feat[src_vn_nzval[src_vn_rowptr[tid_x] + j]] - src_mean) *
          (trg_feat[trg_vn_nzval[trg_vn_rowptr[fv_ind_tmp] + j]] - trg_mean);
      }
      // calculating the ncc gradient
      ncc = sum / sqrt(src_std * trg_std);
      ncc_term[i] = (2 - 2 * ncc);

      // adjusting the gradient of ncc range from (0 ~ 2) to (0.5 ~ 1.5)
      ncc_term[i] = (ncc_term[i] - 0) / (2 - 0);
      ncc_term[i] = (ncc_term[i] - 0.5) * 1.0;
      ncc_term[i] += 1;

      src_coef_ncc[i * n_vert + tid_x] = ncc_term[i];
    }
    __syncthreads();

    ind_tmp = 0;

    coef_grad_tmp = 2 * src_feat[tid_x];
    ind_tmp = s2t_data_map[tid_x]; // range -> from n_vert to n_face

    src_weight_tmp = src_weight_fact[tid_x];

    // calculating the feature gradient between source vertex and three vertices of mapped target face.
    for (int i = 0; i < 3; i++) {
      fv_ind_tmp = trg_face_vert_map[i * n_face + ind_tmp];
      coef_grad_tmp -= 2
        * s2t_data_coef_vec[i * n_vert + tid_x]
        * trg_feat[fv_ind_tmp] * ncc_term[i];
    }
    __syncthreads();

    data_grad_tmp = 0;
    for (int i = 0; i < 3; i++) {
      data_grad_tmp = (*feat_weight)
        * src_weight_tmp;

      data_grad_tmp *= coef_grad_tmp
        * trg_feat_grad[i * n_face + ind_tmp];

      src_data_grad[i * n_vert + tid_x] = data_grad_tmp * ncc_term[i];
    }
    __syncthreads();

    data_coef_grad[tid_x] = (*feat_weight)
      * src_weight_tmp
      * powf((coef_grad_tmp / 2), 2);

    __syncthreads();
  }
}
extern "C" void rmosCudaDdnNCCDataGrad_stream(
  const cudaStream_t sid,
  const int n_vert,
  const int n_face,
  const double* feat_weight,
  const double* __restrict__ src_feat,
  const double* __restrict__ trg_feat,
  const double* __restrict__ trg_feat_grad,
  const double* __restrict__ src_weight_fact,
  const double* __restrict__ s2t_data_coef_vec,
  const int* __restrict__ s2t_data_map,
  const int* __restrict__ trg_face_vert_map,
  // CSR matrix (n_vert, n_vert), 1-ring neighbors of source vertex
  const int* __restrict__ src_vn_nzval,
  const int* __restrict__ src_vn_colind,
  const int* __restrict__ src_vn_rowptr,
  // CSR matrix (n_vert, n_vert), 1-ring neighbors of target vertex
  const int* __restrict__ trg_vn_nzval,
  const int* __restrict__ trg_vn_colind,
  const int* __restrict__ trg_vn_rowptr,
  double* data_coef_grad,
  double* src_data_grad,
  double* src_coef_ncc) {

  size_t blockCols = 128;

  if (n_vert > 128) blockCols = 128;
  else blockCols = n_vert;


  blockCols = 128;
  size_t gridCols = (n_vert + blockCols - 1) / blockCols;

  rmosCudaDdnNCCDataGrad_kernel << <gridCols, blockCols, 0, sid >> > (
    n_vert, n_face, feat_weight, src_feat, trg_feat, trg_feat_grad,
    src_weight_fact, s2t_data_coef_vec, s2t_data_map, trg_face_vert_map,
    src_vn_nzval, src_vn_colind, src_vn_rowptr, trg_vn_nzval, trg_vn_colind, trg_vn_rowptr,
    data_coef_grad, src_data_grad, src_coef_ncc);
}
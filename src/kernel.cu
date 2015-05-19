#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Part.h"

__global__ void cuda_move(float *pos, float *v, int N, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x<N){
      pos[x] += v[x] * dt;
      if (pos[x]>1) pos[x]--;
      if (pos[x]<0) pos[x]++;
    }
}


void Part::move(float dt){

#ifndef CUDA
  #pragma omp parallel for
	for(int i=0; i< 3*m_N; i++){
    m_pos[i] += m_vel[i]*dt;
    if (m_pos[i]>1) m_pos[i]--;
    if (m_pos[i]<0) m_pos[i]++;
  }
#else

  const unsigned int N = 3*m_N;
  const unsigned int block_size = 512 ;
  const unsigned int n_blocks =N/block_size + (N%block_size == 0 ? 0:1);

/*
  cuda_move<<<n_blocks,block_size>>>(m_pos_d, m_vel_d, N, dt);
  unsigned int size = N * sizeof(float);
  cudaMemcpy(m_pos, m_pos_d, size, cudaMemcpyDeviceToHost);
*/


  cudaGraphicsMapResources(1, &m_cuda_resource, 0);

    cudaGraphicsResourceGetMappedPointer ((void **)&m_pos_d, 0, m_cuda_resource);
    cuda_move<<<n_blocks,block_size>>>(m_pos_d, m_vel_d, N, dt);

  cudaGraphicsUnmapResources(1, &m_cuda_resource, 0);


#endif
}

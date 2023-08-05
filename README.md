# ZDO_Cuda
CSC612M Final Project - Parallelization of the ZDO algorithm

**a.) Discussion of parallel algorithms implemented in your program**
The program implements a parallel algorithm for the ZDO algorithm using CUDA. The specific function that is parallelized is the is_zero_indegree function, which is the function for determining whether or not a given node has zero indegree in a graph. The parallel algorithms implemented are the: Grid-strid loop, Memory Prefetching, Memory Advise, CUDA Memory Management, Shared Memory, Parallel Reduction, and Cooperative Group.

  a.1) Grid-stride loop

  The key concept in this GPU parallelization is the use of Grid-stride loop. In the code, the is_zero_indegree_parallel kernel is launched to execute the is_zero_indegree kernel for each node in the graph. The is_zero_indegree_parallel kernel is performed using grid-stride loop, where, in each kernel launch, the total number of threads is divided into multiple thread blocks, each thread block processes a portion of the nodes, and each thread in a block processes a specific node, allowing efficient parallel computation and better utilization of GPU resources. In the program, the total number of threads is specified in the NUMTHREADS constant, and the number of blocks is determined by the number of vertices in the graph. 

  a.2) Memory Prefetching

  Memory prefetching is 

**b.) Execution time comparison between sequential and parallel**


**c.) Detailed analysis and discussion of results**


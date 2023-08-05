# ZDO_Cuda
CSC612M Final Project - Parallelization of the ZDO algorithm

**a.) Discussion of parallel algorithms implemented in your program**

The program implements a parallel algorithm for the ZDO algorithm using CUDA. The specific function that is parallelized is the is_zero_indegree function, which is the function for determining whether or not a given node has zero indegree in a graph. The parallel algorithms implemented are the: Grid-strid loop, Memory Prefetching, Memory Advise, CUDA Memory Management, Shared Memory, Parallel Reduction, and Cooperative Group.

  a.1) Grid-stride loop

  The key concept in this GPU parallelization is the use of Grid-stride loop. In the code, the is_zero_indegree_parallel kernel is launched to execute the is_zero_indegree kernel for each node in the graph. The is_zero_indegree_parallel kernel is performed using grid-stride loop, where, in each kernel launch, the total number of threads is divided into multiple thread blocks, each thread block processes a portion of the nodes, and each thread in a block processes a specific node, allowing efficient parallel computation and better utilization of GPU resources. In the program, the total number of threads is specified in the NUMTHREADS constant, and the number of blocks is determined by the number of vertices in the graph. 

  a.2) Memory Prefetching

  To improve memory access patterns in GPU and achieve the goal of reducing the impact of memory latency, memory prefetching is applied to move the data from the main memory to the GPU memory before it is actually needed. This allows more efficient memory access which can  improve the overall program's performance. In the program, 'cudaMemPrefetchAsync' is called to prefetch 'd_graph_node' to the CPU's memory. The 'd_graph_node' is the one holding the graph node data on the GPU. Through 'cudaMemPrefetchAsync', the memory access latency needed when the GPU kernel accesses the graph node data can be minimized.

  a.3) Memory Advise

  

**b.) Execution time comparison between sequential and parallel**


**c.) Detailed analysis and discussion of results**


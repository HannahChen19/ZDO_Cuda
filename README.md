# ZDO_Cuda
CSC612M Final Project - Parallelization of the ZDO algorithm

**a.) Discussion of parallel algorithms implemented in your program**

The application uses CUDA to construct a parallel algorithm for the ZDO algorithm. The is_zero_indegree function, which determines whether or not a particular node in a graph has zero indegree, is the specific function that is parallelized. Grid-stride loop, Memory Prefetching, Memory Advise, CUDA Memory Management, Parallel Reduction, Shared Memory, and Cooperative Group are the parallel algorithms employed.

a.1) Grid-stride loop

The utilization of a Grid-stride loop is the essential concept in this GPU parallelization. The is_zero_indegree_parallel kernel is launched in the code to run the is_zero_indegree kernel for each node in the graph. The is_zero_indegree_parallel kernel is executed using a grid-stride loop, in which the total number of threads is divided into multiple thread blocks in each kernel launch, each thread block is processing a portion of the nodes, and each thread in a block is processing a specific node, allowing efficient parallel computation and better utilization of GPU resources. The total number of threads in the program is provided by the NUMTHREADS constant, and the number of blocks is determined by the number of vertices in the graph. 

a.2) Memory Prefetching

Memory prefetching is used to move data from main memory to GPU memory before it is needed in order to enhance memory access patterns in GPU and reduce the impact of memory latency. This allows for more efficient memory access, which can increase program performance overall. 'cudaMemPrefetchAsync' is called in the program to prefetch 'd_graph_node' to the CPU's memory. The 'd_graph_node' holds the graph node data on the GPU. The memory access latency required when the GPU kernel accesses the graph node data can be minimized with 'cudaMemPrefetchAsync'.

a.3) Memory Advise

Memory advice is implemented in the program using the 'cudaMemAdvise' function. Following the use of 'cudaMallocManaged' to allocate memory for the 'd_graph_node' array, 'cudaMemAdvise' is used to provide memory advice by stating the desired memory location for 'd_graph_node,' which is on the CPU memory. Memory transfer can be streamlined and data access can be improved by implementing memory advice. Because the GPU now knows where to retrieve the data, the number of page faults can be lowered. As a result, the program's overall performance improves. To further improve performance, the same memory advice technique is applied to the 'd_results' array.

a.4) CUDA Memory Management

CUDA Unified Memory is used in the program to allocate and manage the memories used by both the CPU and the GPU. The 'cudaMallocManaged' function is used to allocate memory for both the 'd_graph_node' and 'd_results' arrays, allowing the CPU and GPU to access the data without explicit copying. Memory management is simplified and data transfer complexities are reduced by using 'cudaMallocManaged' because memory is now managed automatically by the CUDA runtime, and data is copied to the GPU when accessed by GPU kernels and copied back to the CPU when accessed by CPU code.

a.5) Parallel Reduction

Parallel reduction is used in the 'is_zero_indegree' kernel to verify whether there is a path of nodes with zero in-degree leading to the current node. The parallel reduction is used in the program's thread blocks to combine the results of individual threads and to determine if a vertex has a zero in-degree. The reduction is accomplished in the kernel by halving the number of active threads in each step and updating the result accordingly, then saving the final result in 'results[0]'. 

a.6) Shared Memory

A shared memory array '__shared__ bool results[NUM_THREADS];' is allocated in the program's 'is_zero_indegree' kernel for storing the intermediate reduction results for each thread within the thread block. This is advantageous because, with shared memory, threads inside the same block may instantly interact and exchange their findings without having to access global memory, which speeds up the process.

a.7) Cooperative Group

The cooperative group implementation may be found in the following lines of code: 'cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
' as well as 'cta.sync();'. In this program, a cooperative group is used to ensure proper synchronization between threads within the same block, allowing them to collaborate and communicate as needed to perform the calculations correctly, enhancing parallelism and reducing synchronization overhead within thread blocks.

Memory deallocation is used in addition to the procedures outlined above to avoid memory leaks. These were done with the terms 'cudaFree' and 'freeGraph'. Memory deallocation allows the program to verify that the allocated memory is appropriately released and that memory utilization is optimized.  

**b.) Execution time comparison between sequential and parallel**

https://docs.google.com/spreadsheets/d/1rFoWU5vS_UR2b4-SCICiEz5bUZCpbAbnpUWtYpK1Kjk/edit?usp=sharing

<Insert image>

The figure above depicts the execution times of the sequential C kernel and the parallel CUDA kernel. The ZDO algorithm was exposed to a comprehensive execution time comparison between its sequential and parallel versions in this study. The input data are processed step by step in the sequential version, without taking advantage of parallel processing capabilities. The sequential version has an average execution time of **_________** microseconds at various vertex sizes.

Parallelization techniques were used to parallelize the is_zero_indegree function for the CUDA kernel. This enables the completion of tasks concurrently. Despite the fact that parallelization can increase communication overhead due to synchronization needs, the results were favorable. The parallelized version of the ZDO algorithm attained an average execution time of **_________** microseconds, demonstrating a significant speedup over its sequential equivalent. 

The speedup achieved through parallelization in this project is roughly **_______**. This demonstrates that the parallelized version was able to significantly enhance the algorithm's computational efficiency as well as the algorithm's execution time.

**c.) Detailed analysis and discussion of results**

https://docs.google.com/spreadsheets/d/1rFoWU5vS_UR2b4-SCICiEz5bUZCpbAbnpUWtYpK1Kjk/edit?usp=sharing

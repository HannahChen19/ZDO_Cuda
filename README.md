# ZDO_Cuda
CSC612M Final Project - Parallelization of the ZDO algorithm

# A.) Abstract 
This project aims to parallelize a new shortest-path problem algorithm called the
zero-degrees-only (ZDO) algorithm proposed by Amr Elmasry and Ahmed Shokry in
2018. The main idea of the algorithm is to consider both the relaxable graph and the
parent graph to select the best vertices to scan. It will be evaluated using mazes of
different sizes. The metrics used for evaluation will be the execution time, data transfer
overhead, and memory usage. Performance gains from each configuration will also be
recorded. The findings from this project can help future researchers or developers with
the implementation of one of the latest proposed shortest-path problem algorithms which
will be helpful in further research and real life application of the algorithm like in AI.

# B.) Description of the project
## a.) Inputs in the project
The inputs to the project are a graph, vertices and weight of the edges.
## b.) Proposed process
### i. What processes will be parallelized?
The process that will be parallelized is the process of identifying whether or not a
vertex has an indegree of zero.
### ii. What existing implementations have been done with the proposed process?
No existing implementations have been done with the
proposed process. The study in which the group will parallelize included an
implementation or pseudocode of the algorithm.
### iii. Process not parallelized
* The main loop that checks the length of the Queue. We decided not to implement
this in parallel because it is a single source shortest path algorithm and only one
vertex is initially touched. Also, each iteration depends on the result of the
previous iteration. It can also lead to synchronization errors in which the Queue is
being modified by the dequeue(Q) operation and the scan procedure’s
enqueue(Q, v) at the same time.
* The scan procedure will also not be parallelized. This is because enqueueing
requires a more careful handling of the thread synchronization as it could easily
lead to race conditions.




# C.) Parallelization using CUDA

![image](https://github.com/HannahChen19/ZDO_Cuda/assets/140621087/d8769bb3-ce5c-4315-b743-9b2c84b9b64e)

The above image is the pseudocode of the ZDO algorithm. The parallelization will focus on the is-zero-indegree function, which is the
function that identifies if a vertex has an indegree of zero. In order to parallelize
the process, data will first be transferred from the CPU to the GPU. Afterwards, a
CUDA Kernel function with multiple threads will be created, where each thread
will be processing a different vertex. Each threads will cycle through and
compare the current vertex and incoming neighboring vertices’ distance and
edge weights to determine if the indegree is zero or not. Once the execution of
the CUDA Kernel is complete, results will be copied back from the GPU to the
CPU, and cudaFree will be called to free up the allocated GPU memory.
Grid-stride loop, Memory Prefetching, Memory Advise, CUDA Memory Management, Parallel Reduction, Shared Memory, and Cooperative Group are the parallel algorithms employed.

## c.1) Grid-stride loop

The utilization of a Grid-stride loop is the essential concept in this GPU parallelization. The is_zero_indegree_parallel kernel is launched in the code to run the is_zero_indegree kernel for each node in the graph. The is_zero_indegree_parallel kernel is executed using a grid-stride loop, in which the total number of threads is divided into multiple thread blocks in each kernel launch, each thread block is processing a portion of the nodes, and each thread in a block is processing a specific node, allowing efficient parallel computation and better utilization of GPU resources. The total number of threads in the program is provided by the NUMTHREADS constant, and the number of blocks is determined by the number of vertices in the graph. 

## c.2) Memory Prefetching

Memory prefetching is used to move data from main memory to GPU memory before it is needed in order to enhance memory access patterns in GPU and reduce the impact of memory latency. This allows for more efficient memory access, which can increase program performance overall. 'cudaMemPrefetchAsync' is called in the program to prefetch 'd_graph_node' to the CPU's memory. The 'd_graph_node' holds the graph node data on the GPU. The memory access latency required when the GPU kernel accesses the graph node data can be minimized with 'cudaMemPrefetchAsync'.

## c.3) Memory Advise

Memory advice is implemented in the program using the 'cudaMemAdvise' function. Following the use of 'cudaMallocManaged' to allocate memory for the 'd_graph_node' array, 'cudaMemAdvise' is used to provide memory advice by stating the desired memory location for 'd_graph_node,' which is on the CPU memory. Memory transfer can be streamlined and data access can be improved by implementing memory advice. Because the GPU now knows where to retrieve the data, the number of page faults can be lowered. As a result, the program's overall performance improves. To further improve performance, the same memory advice technique is applied to the 'd_results' array.

## c.4) CUDA Memory Management

CUDA Unified Memory is used in the program to allocate and manage the memories used by both the CPU and the GPU. The 'cudaMallocManaged' function is used to allocate memory for both the 'd_graph_node' and 'd_results' arrays, allowing the CPU and GPU to access the data without explicit copying. Memory management is simplified and data transfer complexities are reduced by using 'cudaMallocManaged' because memory is now managed automatically by the CUDA runtime, and data is copied to the GPU when accessed by GPU kernels and copied back to the CPU when accessed by CPU code.

## c.5) Parallel Reduction

Parallel reduction is used in the 'is_zero_indegree' kernel to verify whether there is a path of nodes with zero in-degree leading to the current node. The parallel reduction is used in the program's thread blocks to combine the results of individual threads and to determine if a vertex has a zero in-degree. The reduction is accomplished in the kernel by halving the number of active threads in each step and updating the result accordingly, then saving the final result in 'results[0]'. 

## c.6) Shared Memory

A shared memory array '__shared__ bool results[NUM_THREADS];' is allocated in the program's 'is_zero_indegree' kernel for storing the intermediate reduction results for each thread within the thread block. This is advantageous because, with shared memory, threads inside the same block may instantly interact and exchange their findings without having to access global memory, which speeds up the process.

## c.7) Cooperative Group

The cooperative group implementation may be found in the following lines of code: 'cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
' as well as 'cta.sync();'. In this program, a cooperative group is used to ensure proper synchronization between threads within the same block, allowing them to collaborate and communicate as needed to perform the calculations correctly, enhancing parallelism and reducing synchronization overhead within thread blocks.

Memory deallocation is used in addition to the procedures outlined above to avoid memory leaks. These were done with the terms 'cudaFree' and 'freeGraph'. Memory deallocation allows the program to verify that the allocated memory is appropriately released and that memory utilization is optimized.  

# D.) Results and Discussion


![image](https://github.com/HannahChen19/ZDO_Cuda/assets/140621087/adcdd967-5e0a-4bf7-9ec1-8eb7884c68a0)



The figure above depicts the execution times of the sequential C kernel and parallel CUDA kernel with 1024 threads and different block sizes. The ZDO algorithm was exposed to a comprehensive execution time comparison between its sequential and parallel versions in this study. The input data are processed step by step in the sequential version, without taking advantage of parallel processing capabilities. 


Parallelization techniques were used to parallelize the is_zero_indegree function for the CUDA kernel. This enables the completion of tasks concurrently. Despite the fact that parallelization can increase communication overhead due to synchronization needs, the results were favorable. The parallelized version of the ZDO algorithm attained an execution time of 166.178 to 214.472 microseconds for 2000 nodes, 218.882 to 248.25 microseconds for 4000 nodes, 248.756 to 319.722 microseconds for 6000 nodes, 274.482 to 351.844 microseconds for 8000 nodes, and 343.558 to 367.904 microseconds for 10000 nodes, demonstrating a significant speedup over its sequential equivalent.

![image](https://github.com/HannahChen19/ZDO_Cuda/assets/140621087/b792fe42-afe4-4d2e-bfeb-2ac5b71be1f7)


The formula used for the speedup factor is C execution time / CUDA execution time. The speedup achieved through parallelization in this project is by a factor of 334.038 to 431.1148 for 2000 nodes, 1594.023 to 1807.897 for 4000 nodes, 3594.764 to 4620.291 for 6000 nodes, 6895.165 to 8838.548 for 8000 nodes, 10303.49 to 11033.64 for 10000 nodes. This demonstrates that the parallelized version was able to significantly enhance the algorithm's computational efficiency as well as the algorithm's execution time.



![image](https://github.com/HannahChen19/ZDO_Cuda/assets/140621087/3a52bd61-87fc-45d6-9222-88f275cddd49)
Analyzing the Rate of increase in running time is hard because the node increase is by a constant size. To make it easier for analysis, it is better to normalize the data.



![image](https://github.com/HannahChen19/ZDO_Cuda/assets/140621087/012950da-ab43-4ed6-b845-df9495233d42)


The results show that the edge increase factor is approximately 2, which is the same as the vertex increase factor. For the C version, there was an average increase of 5.83, much greater than the vertex increase. This suggests a non-linear relationship, making it difficult to scale and estimate the running time for vertex of different sizes. In the CUDA configurations, the 512-blocks configuration had the largest average increase of 1.4, while the 256-blocks had the smallest. We also observed that the 1024-block configuration had a significant increase of 22.46% in its rate of increase in running time, from vertex size 2000-4000 to 4000-8000. This could mean that using the 1024-block for larger vertex and edge sizes would lead to much slower performance and should be limited to small vertex size use. Unfortunately, due to the small test size, it is difficult to determine if this trend will continue.


# E. Conclusion
Comparisons of execution times reveals that for the ZDO algorithm, the parallel version had faster execution times than the sequential version. This is due to the fact that data in the sequential version is processed step by step, following a linear execution path that runs one operation after another, with no concurrent processing capabilities. As a result, processing times will be longer, especially when the vertex size is huge. In the parallel version, however, the application has massive throughput and parallel processing capabilities, allowing tasks to be completed concurrently. This effectively utilizes existing computing resources and can result in significant execution time reductions, particularly when dealing with computationally complex tasks or huge datasets, such as the ZDO algorithm.

It is important to note, however, that because CUDA executions may incur overheads such as thread synchronization overheads, data transfer overheads when transferring data between the CPU and GPU memory, and kernel launch overheads for configuring the GPU for execution and managing thread blocks, execution time of CUDA kernel in small vertex sizes can be longer than the C kernel because the overhead still outweighs the benefits. CUDA overheads may be more noticeable when data transfers occur frequently or when the vertex size.


## Problems Encountered
* Error encountered when running the CUDA code multiple times
* Error encountered at large vertex size e.g. 200,0000 vertex
* Cuda program sometimes break in Google Colab when using free edition

## Future recommendation
* Execute both Cuda and C program on local machine
* Test more vertex sizes for more accurate data

## References
Elmasry, A., & Shokry, A. (2018). A new algorithm for the shortest-path problem.
Networks. doi:10.1002/net.21870




#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

#define INF INT_MAX
// Block size used for CUDA kernel launches.
#define BLOCK_SIZE 256
// INF_PACK packs two INF values in a 64-bit integer: high 32 bits and low 32 bits.
#define INF_PACK (((long long)INF << 32) | ((unsigned int)INF))
// Modulus value used for final MST weight calculation.
#define MOD 1000000007

// Data Structures

// Structure to store an edge in the graph.
// It contains source vertex, destination vertex, the base weight, and a type code.
// The type code will later determine how the weight is updated.
struct Edge {
    int src;     // Source vertex of the edge.
    int dest;    // Destination vertex of the edge.
    int weight;  // Base weight of the edge; will be modified based on type.
    int type;    // Type of the edge: 0-normal, 1-green, 2-traffic, 3-dept (default=0).
};

// Device Functions

// Device function for the union-find "find" operation with path halving.
// It finds the root parent of vertex 'v' and flattens the tree structure along the way.
__device__ int findParent(int *d_parent, int v) {
    while(d_parent[v] != v) {
        d_parent[v] = d_parent[d_parent[v]];  // Path halving for efficiency i.e compression
        v = d_parent[v];
    }
    return v;
}

// Device Kernels

// Kernel to update each edge's weight based on its type.
// Each thread processes one edge and multiplies its weight by a factor determined by its type.
__global__ void updateEdgeWeights(Edge *d_edges, int E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E)
    {

        int factor;
        int t = d_edges[tid].type;
        // Determine factor based on edge type.
        switch(t) {
            case 0: factor = 1; break;  // normal edge: factor 1.
            case 1: factor = 2; break;  // green edge: factor 2.
            case 2: factor = 5; break;  // traffic edge: factor 5.
            case 3: factor = 3; break;  // dept edge: factor 3.
            default: factor = 1; break;
        }
        // Update the edge weight by multiplying with the factor.
        d_edges[tid].weight *= factor;
    }
}

// Kernel to initialize the per-component candidate array used to store the cheapest edge.
// Each vertex gets an initial value of INF_PACK.
__global__ void init_kernel(long long *d_cheapest_pack, int V) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V)
        d_cheapest_pack[tid] = INF_PACK;
}

// Given below was not stride based approach
// Kernel to compute the cheapest edge candidate for each component.
// Each thread processes one edge, computes a candidate value packed as (weight << 32) | (edge index),
// and uses atomicMin to update the cheapest candidate edge for both components connected by the edge.
// __global__ void find_cheapest_kernel(Edge *d_edges, int *d_parent, long long *d_cheapest_pack, int E) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= E) return;
    
//     // Load the edge.
//     Edge e = d_edges[tid];
//     int v1 = e.src;
//     int v2 = e.dest;
//     int w = e.weight;
    
//     // Find the parent components for both vertices.
//     int pv1 = findParent(d_parent, v1);
//     int pv2 = findParent(d_parent, v2);
    
//     // If the vertices are in different components, update the cheapest candidate for both components.
//     if (pv1 != pv2) {
//         long long candidate = (((long long)w) << 32) | ((unsigned int)tid);
//         atomicMin(&d_cheapest_pack[pv1], candidate);
//         atomicMin(&d_cheapest_pack[pv2], candidate);
//     }
// }

// Given below is stride based approach that searches across blocks
__global__ void find_cheapest_kernel(Edge *d_edges, int *d_parent, long long *d_cheapest_pack, int E) {
    // Compute the global thread index based on block and thread indices.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of threads in the grid.
    // This is used as the "stride" for each thread to process multiple edges.
    int stride = blockDim.x * gridDim.x;
    
    // Loop over edges using the stride-based approach:
    // Each thread starts at its computed index and processes every 'stride'th edge.
    for (int i = idx; i < E; i += stride) {
        // Load the edge from global memory.
        Edge e = d_edges[i];
        int v1 = e.src;
        int v2 = e.dest;
        int w = e.weight;
        
        // Find the parent components for both vertices.
        int pv1 = findParent(d_parent, v1);
        int pv2 = findParent(d_parent, v2);
        
        // If the vertices belong to different components, update the cheapest candidate.
        // This candidate packs the edge weight (in the high-order 32 bits) and the edge index (in the low-order 32 bits).
        if (pv1 != pv2) {
            long long candidate = (((long long)w) << 32) | ((unsigned int)i);
            // Atomic minimum update for both parent components ensures that the cheapest candidate is stored.
            atomicMin(&d_cheapest_pack[pv1], candidate);
            atomicMin(&d_cheapest_pack[pv2], candidate);
        }
    }
}





// Kernel to perform the union operation in parallel based on the cheapest edges found.
// For each component (root vertex), this kernel selects the cheapest edge and attempts to merge two components.
__global__ void union_kernel(Edge *d_edges, int *d_parent,
                                          long long *d_cheapest_pack, int V,
                                          int *d_mst, int *d_nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= V) return;
    if(tid < V)
    {
        // Process only if the current vertex is a root (representative of its component).
        if (d_parent[tid] != tid) return;

        // Retrieve the packed candidate for this component.
        long long pack = d_cheapest_pack[tid];
        int edge_idx = (int)(pack & 0xffffffff);  // Extract edge index from low 32 bits.
        int weight = (int)(pack >> 32);           // Extract weight from high 32 bits.

        // If the weight is INF, then no valid candidate exists.
        if (weight == INF) return;

        // Retrieve the edge corresponding to the candidate.
        Edge e = d_edges[edge_idx];
        int u = e.src;
        int v = e.dest;

        // Find the current roots for both vertices.
        int root_u = findParent(d_parent, u);
        int root_v = findParent(d_parent, v);

        // If both vertices are already in the same component, skip the union.
        if (root_u == root_v) return;


        // // Get current roots for both endpoints.
        // int root1 = findParent(d_parent, u);
        // int root2 = findParent(d_parent, v);
        // if (root1 == root2) continue;  // already in same component
        
        // // Deterministic union-by-rank.
        // if (d_rank[root1] < d_rank[root2]) {
        //     d_parent[root1] = root2;
        // } else if (d_rank[root1] > d_rank[root2]) {
        //     d_parent[root2] = root1;
        // } else {
        //     if (root1 < root2) {
        //         d_parent[root2] = root1;
        //         d_rank[root1]++; 
        //     } else {
        //         d_parent[root1] = root2;
        //         d_rank[root2]++;
        //     }
        // }


        // Merge the two components deterministically:
        // The smaller root becomes the parent.
        int high = max(root_u, root_v);
        int low = min(root_u, root_v);

        // Attempt to merge by making the higher numbered root point to the lower.
        if (atomicCAS(&d_parent[high], high, low) == high) {
            // If merge is successful, add the weight of the edge to the MST and decrease component count.
            atomicAdd(d_mst, weight);
            atomicSub(d_nc, 1);
        }
    }
}

// Kernel to perform modulus operation on the MST weight.
__global__ void mod_kernel(int *d_mst)
{
    *d_mst = *d_mst % MOD;
}

// Kernel to initialize the union-find structure.
// Each vertex is initially its own parent.
__global__ void union_find_kernel(int *d_parent, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < V) {
        d_parent[idx] = idx;  // Set parent of each vertex to itself.
    }
}

// Main Function

int main() {
    int V, E;
    // Read the number of vertices and edges.
    std::cin >> V >> E;
    
    // Allocate host memory for the edges.
    Edge* edges = new Edge[E];
    for (int i = 0; i < E; i++) {
        int u, v, wt;
        std::string s;
        // Read each edge's source, destination, weight, and type as a string.
        std::cin >> u >> v >> wt >> s;
        edges[i].src = u;
        edges[i].dest = v;
        edges[i].weight = wt; // Base weight.
        // Convert the string type to an integer code.
        if (s == "normal") edges[i].type = 0;
        else if (s == "green") edges[i].type = 1;
        else if (s == "traffic") edges[i].type = 2;
        else if (s == "dept") edges[i].type = 3;
        else edges[i].type = 0;
    }
        
    // Allocate device memory for edges.
    Edge *d_edges;
    cudaMalloc(&d_edges, E * sizeof(Edge));
    // Copy edges from host to device.
    cudaMemcpy(d_edges, edges, E * sizeof(Edge), cudaMemcpyHostToDevice);
    
    // Allocate device memory for the union-find parent array.
    int *d_parent;
    cudaMalloc(&d_parent, V * sizeof(int));
    int *parent = new int[V];

    // Calculate grid dimensions based on number of vertices and edges.
    int vertexBlocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int edgeBlocks = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate device memory for the cheapest edge candidate array.
    long long *d_cheapest_pack;
    cudaMalloc(&d_cheapest_pack, V * sizeof(long long));
    
    // Variables to store the MST weight and number of connected components.
    int mst = 0;
    int numComponents = V;
    
    // Allocate device memory for MST weight and number of components.
    int *d_mst, *d_nc;
    cudaMalloc(&d_mst, sizeof(int));
    cudaMalloc(&d_nc, sizeof(int));
    cudaMemcpy(d_mst, &mst, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nc, &numComponents, sizeof(int), cudaMemcpyHostToDevice);

    // Start timer for performance measurement.
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize union-find structure on device.
    union_find_kernel<<<vertexBlocks, BLOCK_SIZE>>>(d_parent, V);
    
    // Update edge weights based on their type.
    updateEdgeWeights<<<edgeBlocks, BLOCK_SIZE>>>(d_edges, E);
    cudaDeviceSynchronize();
    // int iterations=0
    // int maxIterations=V-1;

    // Main loop: while there are more than one component, perform MST merging.
    // while (numComponents > 1 && iteration < maxIterations) {
    while (numComponents > 1) {
        // iterations++;
        // Reset the candidate array for each component.
        init_kernel
    <<<vertexBlocks, BLOCK_SIZE>>>(d_cheapest_pack, V);
        cudaDeviceSynchronize();
        
        // Find the cheapest edge candidate for each component.
        find_cheapest_kernel
    <<<edgeBlocks, BLOCK_SIZE>>>(d_edges, d_parent, d_cheapest_pack, E);
        cudaDeviceSynchronize();
        
        // Perform union operation deterministically on each component using the cheapest edge.
        union_kernel<<<vertexBlocks, BLOCK_SIZE>>>(d_edges, d_parent, d_cheapest_pack, V, d_mst, d_nc);
        cudaDeviceSynchronize();
        
        // Check for CUDA errors.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // Update the number of components on the host after union operations.
        cudaMemcpy(&numComponents, d_nc, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // if (iteration >= maxIterations)
    // std::cout << "Warning: Reached maximum iterations (" << maxIterations << ")" << std::endl;

    // Apply modulus to the MST weight.
    mod_kernel<<<1,1>>>(d_mst);
    cudaMemcpy(&mst, d_mst, sizeof(int), cudaMemcpyDeviceToHost);
    // Output the final MST weight.
    std::cout << mst << std::endl;
    
    // End timer and optionally print execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;
    
    // Free allocated host and device memory.
    delete[] edges;
    delete[] parent;
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_cheapest_pack);
    cudaFree(d_mst);
    cudaFree(d_nc);
    
    return 0;
}

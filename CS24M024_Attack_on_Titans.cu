#include <iostream>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#define MAX_CITIES 20000
#define INF 1000000000LL

struct CityData {
    int start_city;
    int population;
    int elderly;
};

struct Edge {
    int v, w, capacity;
};

__global__ void simulateEvacuation(
    int num_cities,
    int* row_ptr,
    int* col_ind,
    int* edge_len,
    int* edge_cap,
    int* shelter_city,
    int* shelter_capacity,
    CityData* cities,
    int max_dist_elderly,
    int num_pop,
    int* output_paths,
    int* output_path_len,
    int* output_drops,
    int* output_drop_len,
    int* road_time,
    int* road_lock,
    // long long* global_start_time,
    long long max_cycles
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_pop) return;

    

    int start = cities[tid].start_city;
    int prime = cities[tid].population;
    int elderly = cities[tid].elderly;
    int evacuees = prime + elderly;

    bool visited[MAX_CITIES];
    int dist[MAX_CITIES];
    int parent[MAX_CITIES];
    int drop_idx = 0;
    int path_ptr = 0;
    int current_time = 0;
    int distance_walked = 0;
    long long start_clock = clock64();

    if (evacuees == 0) {
        output_paths[tid * num_cities * 5 + path_ptr++] = start;
        output_path_len[tid] = path_ptr;
        output_drop_len[tid] = drop_idx;
        return;
    }

    while (evacuees > 0) {
          long long now = clock64();
          long long elapsed_cycles = now - start_clock;
          if (elapsed_cycles >= max_cycles){
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 0] = start;
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 1] = prime;
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 2] = elderly;
            drop_idx++;
            output_path_len[tid] = path_ptr;
            output_drop_len[tid] = drop_idx;
            return;
          }

        for (int i = 0; i < num_cities; i++) {
            dist[i] = INF;
            parent[i] = -1;
            visited[i] = false;
        }
        dist[start] = 0;

        for (int i = 0; i < num_cities; i++) {
            int u = -1;
            for (int j = 0; j < num_cities; j++) {
                if (!visited[j] && (u == -1 || dist[j] < dist[u])) u = j;
            }
            if (u == -1 || dist[u] == INF) break;
            visited[u] = true;

            for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
                int v = col_ind[j];
                int est = max(road_time[j], current_time + dist[u]);
                int cost = est + edge_len[j];

                if (dist[v] > cost) {
                    dist[v] = cost;
                    parent[v] = u;
                }
            }
        }

        int nearest = -1, mindist = INF;
        for (int i = 0; i < num_cities; i++) {
            if (shelter_city[i] && shelter_capacity[i] > 0 && dist[i] < mindist) {
                mindist = dist[i];
                nearest = i;
            }
        }
        if (nearest == -1) break;

        int cur = nearest;
        int path_temp[1024], temp_idx = 0;
        while (cur != -1) {
            path_temp[temp_idx++] = cur;
            cur = parent[cur];
        }

        if (path_ptr == 0 || start != output_paths[tid * num_cities * 5 + path_ptr - 1]) {
            for (int i = temp_idx - 1; i >= 0; i--) output_paths[tid * num_cities * 5 + path_ptr++] = path_temp[i];
        } else {
            for (int i = temp_idx - 2; i >= 0; i--) output_paths[tid * num_cities * 5 + path_ptr++] = path_temp[i];
        }

        int time_ptr = current_time;
        int last_valid_city = path_temp[temp_idx - 1];
        int drop_elderly_at = -1;

        for (int i = temp_idx - 1; i > 0; i--) {
            int u = path_temp[i];
            int v = path_temp[i - 1];
            int edge_idx = -1;
            for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
                if (col_ind[j] == v) {
                    edge_idx = j;
                    break;
                }
            }
            if (edge_idx == -1) continue;

            int batches = (evacuees + edge_cap[edge_idx] - 1) / edge_cap[edge_idx];
            int batch_time = batches * edge_len[edge_idx];

            bool acquired = false;
            while (!acquired) {
                if (atomicCAS(&road_lock[edge_idx], 0, 1) == 0) {
                    int avail_time = road_time[edge_idx];
                    if (avail_time <= time_ptr) {
                        road_time[edge_idx] = time_ptr + batch_time;
                        acquired = true;
                    } else {
                        time_ptr = avail_time;
                    }
                    atomicExch(&road_lock[edge_idx], 0);
                }
            }

            time_ptr += batch_time;
            distance_walked += edge_len[edge_idx];

            if (elderly > 0 && distance_walked > max_dist_elderly && drop_elderly_at == -1) {
                drop_elderly_at = last_valid_city;
                while (path_ptr > 0 && output_paths[tid * num_cities * 5 + path_ptr - 1] != drop_elderly_at) {
                    path_ptr--;
                }
            }
            last_valid_city = v;
        }

        current_time = time_ptr;

        if (drop_elderly_at != -1 && elderly > 0) {
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 0] = drop_elderly_at;
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 1] = 0;
            output_drops[tid * 3 * 1000 + drop_idx * 3 + 2] = elderly;
            evacuees -= elderly;
            elderly = 0;
            drop_idx++;
            start = drop_elderly_at;
            continue;
        }
        // if (clock64() - *global_start_time >= max_duration) return;
        while (true) {
            int cap = atomicAdd(&shelter_capacity[nearest], 0);
            // if (clock64() - *global_start_time >= max_duration) return;
            if (cap == 0) break;
            int to_assign = min(cap, prime + elderly);
            if (to_assign == 0) break;

            if (atomicCAS(&shelter_capacity[nearest], cap, cap - to_assign) == cap) {
                int assign_elderly = min(elderly, to_assign);
                int assign_prime = to_assign - assign_elderly;

                output_drops[tid * 3 * 1000 + drop_idx * 3 + 0] = nearest;
                output_drops[tid * 3 * 1000 + drop_idx * 3 + 1] = assign_prime;
                output_drops[tid * 3 * 1000 + drop_idx * 3 + 2] = assign_elderly;
                prime -= assign_prime;
                elderly -= assign_elderly;
                evacuees = prime + elderly;
                drop_idx++;
                break;
            }
        }

        start = nearest;
    }

    output_path_len[tid] = path_ptr;
    output_drop_len[tid] = drop_idx;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./evacuation_sim input.txt output.txt\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    int num_cities, num_roads;
    infile >> num_cities >> num_roads;

    std::vector<std::vector<Edge>> adj(num_cities);
    for (int i = 0; i < num_roads; i++) {
        int u, v, len, cap;
        infile >> u >> v >> len >> cap;
        adj[u].push_back({v, len, cap});
        adj[v].push_back({u, len, cap});
    }

    std::vector<int> row_ptr(num_cities + 1);
    std::vector<int> col_ind, edge_len, edge_cap;
    int edge_idx = 0;
    for (int i = 0; i < num_cities; i++) {
        row_ptr[i] = edge_idx;
        for (auto& e : adj[i]) {
            col_ind.push_back(e.v);
            edge_len.push_back(e.w);
            edge_cap.push_back(e.capacity);
            edge_idx++;
        }
    }
    row_ptr[num_cities] = edge_idx;

    int num_shelters;
    infile >> num_shelters;
    std::vector<int> shelter_city(num_cities, 0);
    std::vector<int> shelter_capacity(num_cities, 0);
    for (int i = 0; i < num_shelters; i++) {
        int city, cap;
        infile >> city >> cap;
        shelter_city[city] = 1;
        shelter_capacity[city] = cap;
    }

    int num_pop;
    infile >> num_pop;
    std::vector<CityData> pop(num_pop);
    for (int i = 0; i < num_pop; i++) {
        infile >> pop[i].start_city >> pop[i].population >> pop[i].elderly;
    }

    int max_dist_elderly;
    infile >> max_dist_elderly;
    infile.close();

    // Device memory
    int *d_row_ptr, *d_col_ind, *d_edge_len, *d_edge_cap;
    int *d_shelter, *d_shelter_cap;
    CityData* d_cities;
    int *d_output_paths, *d_output_path_len;
    int *d_output_drops, *d_output_drop_len;
    int *d_road_time, *d_road_lock;
    // long long *d_global_start_time;

    cudaMalloc(&d_row_ptr, sizeof(int) * row_ptr.size());
    cudaMalloc(&d_col_ind, sizeof(int) * col_ind.size());
    cudaMalloc(&d_edge_len, sizeof(int) * edge_len.size());
    cudaMalloc(&d_edge_cap, sizeof(int) * edge_cap.size());
    cudaMalloc(&d_shelter, sizeof(int) * num_cities);
    cudaMalloc(&d_shelter_cap, sizeof(int) * num_cities);
    cudaMalloc(&d_cities, sizeof(CityData) * num_pop);
    cudaMalloc(&d_output_paths, sizeof(int) * num_pop * num_cities * 5);
    cudaMalloc(&d_output_path_len, sizeof(int) * num_pop);
    cudaMalloc(&d_output_drops, sizeof(int) * num_pop * 3 * 1000);
    cudaMalloc(&d_output_drop_len, sizeof(int) * num_pop);
    cudaMalloc(&d_road_time, sizeof(int) * edge_cap.size());
    cudaMalloc(&d_road_lock, sizeof(int) * edge_cap.size());
    // cudaMalloc(&d_global_start_time, sizeof(long long));

    cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(int) * row_ptr.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind.data(), sizeof(int) * col_ind.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_len, edge_len.data(), sizeof(int) * edge_len.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_cap, edge_cap.data(), sizeof(int) * edge_cap.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelter, shelter_city.data(), sizeof(int) * num_cities, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelter_cap, shelter_capacity.data(), sizeof(int) * num_cities, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cities, pop.data(), sizeof(CityData) * num_pop, cudaMemcpyHostToDevice);
    cudaMemset(d_road_time, 0, sizeof(int) * edge_cap.size());
    cudaMemset(d_road_lock, 0, sizeof(int) * edge_cap.size());
    // long long zero = 0;
    // cudaMemcpy(d_global_start_time, &zero, sizeof(long long), cudaMemcpyHostToDevice);

    // long long max_duration = 9LL * 60 * 1000000000 + 30LL * 1000000000 / 60; // 9m55s
    // long long max_duration = 2LL * 60 * 1000000000LL; // 2 minutes in nanoseconds
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    long long max_cycles = static_cast<long long>(prop.clockRate) * 1000LL * 400LL;

    int threadsPerBlock = 256;
    int blocks = (num_pop + threadsPerBlock - 1) / threadsPerBlock;
    // auto start_cpu = std::chrono::high_resolution_clock::now();
    // long long start_cpu = clock64();

    simulateEvacuation<<<blocks, threadsPerBlock>>>(
        num_cities, d_row_ptr, d_col_ind, d_edge_len, d_edge_cap,
        d_shelter, d_shelter_cap, d_cities, max_dist_elderly, num_pop,
        d_output_paths, d_output_path_len, d_output_drops, d_output_drop_len,
        d_road_time, d_road_lock, max_cycles
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end_time - start_cpu;
    // double elapsed_sec = elapsed.count();
    // std::cout << "elapsed_time_sec = " << elapsed_sec << "\n";
    std::vector<int> h_paths(num_pop * num_cities * 5);
    std::vector<int> h_path_len(num_pop);
    std::vector<int> h_drops(num_pop * 3 * 1000);
    std::vector<int> h_drop_len(num_pop);

    cudaMemcpy(h_paths.data(), d_output_paths, sizeof(int) * h_paths.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_path_len.data(), d_output_path_len, sizeof(int) * num_pop, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_drops.data(), d_output_drops, sizeof(int) * h_drops.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_drop_len.data(), d_output_drop_len, sizeof(int) * num_pop, cudaMemcpyDeviceToHost);

    std::ofstream outfile(argv[2]);

    // outfile << "path_sizes = [";
    // for (int i = 0; i < num_pop; i++) {
    //     if (i != 0) outfile << ", ";
    //     outfile << h_path_len[i];
    // }
    // outfile << "]\n";

    // outfile << "paths = [";
    // for (int i = 0; i < num_pop; i++) {
    //     outfile << "[";
    //     for (int j = 0; j < h_path_len[i]; j++) {
    //         if (j != 0) outfile << ", ";
    //         outfile << h_paths[i * num_cities * 5 + j];
    //     }
    //     outfile << "]";
    //     if (i != num_pop - 1) outfile << ", ";
    // }
    // outfile << "]\n";

    // outfile << "num_drops = [";
    // for (int i = 0; i < num_pop; i++) {
    //     if (i != 0) outfile << ", ";
    //     outfile << h_drop_len[i];
    // }
    // outfile << "]\n";

    // outfile << "drops = [";
    // for (int i = 0; i < num_pop; i++) {
    //     outfile << "[";
    //     for (int j = 0; j < h_drop_len[i]; j++) {
    //         if (j != 0) outfile << ", ";
    //         outfile << "(" << h_drops[i * 3 * 1000 + j * 3 + 0] << ", "
    //                 << h_drops[i * 3 * 1000 + j * 3 + 1] << ", "
    //                 << h_drops[i * 3 * 1000 + j * 3 + 2] << ")";
    //     }
    //     outfile << "]";
    //     if (i != num_pop - 1) outfile << ", ";
    // }
    // outfile << "]\n";

    // Write output
    for (int i = 0; i < num_pop; i++) {
        for (int j = 0; j < h_path_len[i]; j++)
            outfile << h_paths[i * num_cities * 5 + j] << " ";
        outfile << "\n";
    }
    for (int i = 0; i < num_pop; i++) {
        for (int j = 0; j < h_drop_len[i]; j++) {
            outfile << h_drops[i * 3 * 1000 + j * 3 + 0] << " " << h_drops[i * 3 * 1000 + j * 3 + 1] << " " << h_drops[i * 3 * 1000 + j * 3 + 2] << " ";
        }
        outfile << "\n";
    }

    // long long end_cpu = clock64();
    // double elapsed_sec = (double)(end_cpu - start_cpu) / 1e9;
    // outfile << "elapsed_time_sec = " << elapsed_sec << "\n";

    outfile.close();
    return 0;
}

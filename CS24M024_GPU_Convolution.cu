#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
// #include <cuda/cuda_runtime.h>   // I have commented out this line as it was not working on my local machine

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix,long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ long int shared_filter[];
    // if(threadIdx.x==0)
    // {
    //     for(int i=0;i<r*s*c*k;i++)
    //     {
    //         shared_filter[i] = filter[i];
    //     }   
    // }
    //above approach was making only one thread to copy all contents, hence to use to 
    //memeory coalescing, I have used below approach

    //given below copying filter to shared_filter takes performs memory coalescing.  
    for(int i=threadIdx.x;i<r*s*c*k;i+=blockDim.x)
    {
        shared_filter[i] = filter[i];
    }
    __syncthreads();

    if (id < h * w * k)
    {
        int kk = id / (h * w);  
        int new_id = id % (h * w);
        int row = new_id / w;
        int col = new_id % w;
        long int fin = 0;

        for (int cc = 0; cc < c; cc++)  //iterating over channels
        {
            for (int i = 0; i < r; i++)   //iterating over rows
            {
                for (int j = 0; j < s; j++)     //iterating over columns
                {
                    int r_i = row - (r / 2) + i;
                    int c_i = col - (s / 2) + j;
                    // check condition to see out of bounds error
                    if (r_i >= 0 && r_i < h && c_i >= 0 && c_i < w)
                    {
                    
                        int matrix_index = (cc * h + r_i) * w + c_i;     
                        int filter_index = ((kk * c + cc) * r + i) * s + j;
                        fin += matrix[matrix_index] * shared_filter[filter_index];
                    }
                }
            }
        }
        result[id] = fin;
    }
}



int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }
    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch
    /****************************************************Start Here***********************************************************/
    /*
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    long int *d_input, *d_filters, *d_output;
    long int input_size = c * h * w * sizeof(long int);
    long int filter_size = cf * k * r * s * sizeof(long int);
    long int output_size = k * h * w * sizeof(long int);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filters, filter_size);
    cudaMalloc(&d_output, output_size);
    
    cudaMemcpy(d_input, h_mat, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, h_filter, filter_size, cudaMemcpyHostToDevice);

    // I have calculated shared memory size here and stored in variable type in size_t
    size_t shared_memory_size = r * s * c * k * sizeof(long int);

    //Block size of 256 is chosen after trying out multiple runs with 1024,512 and 256 threads per block
    // I am using total threads equals to output size i.e k*h*w
    int n_blocks=ceil((k*h*w)/256.0);
    dim3 gridsize(n_blocks);
    dim3 blocksize(256);
    // Used third parameter to specify the size of shared memory dynamically
    dkernel<<<gridsize,blocksize,shared_memory_size>>>(d_input,d_filters,d_output,h,w,c,r,s,k);
    cudaMemcpy(h_ans, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_filters);
    cudaFree(d_output);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}



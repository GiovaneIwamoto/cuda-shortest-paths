#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define INFNTY INT_MAX

int *graph_data;
int n_vertices;

/* Undirected graph non-negative edge weights */
void generate_random_adj_matrix(int n_vertices)
{
    int N = n_vertices;
    int i, j;

    /* Allocate memory for adjacency matrix 2D array */
    graph_data = (int *)malloc(N * N * sizeof(int));
    for (i = 0; i < N; i++)
    {
        for (j = i; j < N; j++)
        {
            if (i == j)
            {
                graph_data[i * N + j] = 0; /* Diagonal */
            }
            else
            {
                /* Zero to nine random */
                int r = rand() % 10;
                int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
                graph_data[i * N + j] = val;     /* Symmetrically */
                graph_data[j * N + i] = val;     /* Symmetrically */
            }
        }
    }
}

/* Kernel CUDA Floyd algorithm */
__global__ void floyd_warshall_parallel(int *dp, int N, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) /* Within valid range of matrix */
    {
        if (dp[i * N + k] != INFNTY && dp[k * N + j] != INFNTY) /* Path through vertex k exists */
        {
            int new_dist = dp[i * N + k] + dp[k * N + j];
            if (new_dist < dp[i * N + j])
                dp[i * N + j] = new_dist;
        }
    }
}

void floyd_warshall_cuda(int *dp, int N)
{
    int *dev_dp;
    size_t size = N * N * sizeof(int);

    /* Allocate memory on GPU*/
    cudaMalloc((void **)&dev_dp, size);

    /* Copy data from CPU to GPU */
    cudaMemcpy(dev_dp, dp, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    /* Execute kernel for each vertex, k pivot */
    for (int k = 0; k < N; ++k)
    {
        floyd_warshall_parallel<<<gridSize, blockSize>>>(dev_dp, N, k);
        cudaDeviceSynchronize(); /* Sync all kernels finished */
    }
    /* Copy data from GPU to CPU */
    cudaMemcpy(dp, dev_dp, size, cudaMemcpyDeviceToHost);

    /* Free GPU allocated memory */
    cudaFree(dev_dp);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./floyd_parallel <number_of_vertices>\n");
        return 1;
    }

    n_vertices = atoi(argv[1]);
    generate_random_adj_matrix(n_vertices);

    int *dp_matrix = (int *)malloc(n_vertices * n_vertices * sizeof(int));
    memcpy(dp_matrix, graph_data, n_vertices * n_vertices * sizeof(int));

    clock_t start = clock(); /* Start measuring execution time */

    floyd_warshall_cuda(dp_matrix, n_vertices);

    /* Stop measuring execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);

    /* Free allocated memory */
    free(dp_matrix);

    return 0;
}

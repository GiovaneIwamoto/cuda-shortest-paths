#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define INFNTY INT_MAX

int **adjacency_matrix, **dp_matrix;
int n_vertices;

/* Undirected graph non-negative edge weights */
void generate_random_adj_matrix(int n_vertices)
{
    int N = n_vertices;
    int i, j;

    /* Allocate memory for adjacency matrix 2D array */
    adjacency_matrix = (int **)malloc(N * sizeof(int *));

    for (i = 0; i < N; i++)
    {
        adjacency_matrix[i] = (int *)malloc(N * sizeof(int));
    }
    srand(0);

    for (i = 0; i < N; i++)
    {
        for (j = i; j < N; j++)
        {
            if (i == j)
            {
                adjacency_matrix[i][j] = 0;
            }
            else
            {
                /* Zero to nine random */
                int r = rand() % 10;
                int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
                adjacency_matrix[i][j] = val;    /* Symmetrically */
                adjacency_matrix[j][i] = val;    /* Symmetrically */
            }
        }
    }
}

/* Kernel CUDA Floyd algorithm */
__global__ void floyd_warshall_parallel(int *dev_dp, int n_vertices, int k)
{
    int i, j;
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < n_vertices * n_vertices) /* tid is within matrix dim */
    {
        i = tid / n_vertices;
        j = tid - i * n_vertices;

        if (dev_dp[tid] > (dev_dp[i * n_vertices + k]) + (dev_dp[k * n_vertices + j]))
        {
            dev_dp[tid] = (dev_dp[i * n_vertices + k]) + (dev_dp[k * n_vertices + j]);
        }
    }
}

int main(int argc, char **argv)
{
    int i;
    if (argc != 2)
    {
        printf("USAGE: ./floyd_parallel <number_of_vertices>\n");
        return 1;
    }

    n_vertices = atoi(argv[1]);

    dp_matrix = (int **)malloc(n_vertices * sizeof(int *));
    for (i = 0; i < n_vertices; i++)
    {
        dp_matrix[i] = (int *)malloc(n_vertices * sizeof(int));
    }

    generate_random_adj_matrix(n_vertices);

    dim3 blockSize(16, 16);
    dim3 gridSize((n_vertices + blockSize.x - 1) / blockSize.x, (n_vertices + blockSize.y - 1) / blockSize.y);

    int *dev_dp;
    cudaMalloc((void **)&dev_dp, n_vertices * n_vertices * sizeof(int));

    for (int i = 0; i < n_vertices; i++)
    {
        cudaMemcpy(dev_dp + i * n_vertices, adjacency_matrix[i], n_vertices * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    clock_t start = clock(); /* Start measuring execution time */

    /* Execute kernel for each vertex, k pivot */
    for (int k = 0; k < n_vertices; k++)
    {
        floyd_warshall_parallel<<<gridSize, blockSize>>>(dev_dp, n_vertices, k);
        cudaDeviceSynchronize(); /* Sync all kernels finished */
    }

    /* Stop measuring execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);

    /* Return dp matrix to CPU */
    for (int i = 0; i < n_vertices; i++)
    {
        cudaMemcpy(dp_matrix[i], dev_dp + i * n_vertices, n_vertices * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    /* Free allocated memory */
    free(dp_matrix);

    return 0;
}

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

void floyd_warshall_serial(int **graph, int **dp, int N)
{
    int i, j, k;
    /* Initialize copy graph to dp matrix */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            dp[i][j] = graph[i][j];

    clock_t start = clock(); /* Start measuring execution time */

    /* Floyd Warshall algorithm */
    for (k = 0; k < N; k++)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (dp[i][k] + dp[k][j] < dp[i][j])
                    dp[i][j] = dp[i][k] + dp[k][j];
            }
        }
    }

    /* Stop measuring execution time */
    clock_t end = clock();
    float cpu_time = (float)(end - start) / CLOCKS_PER_SEC;

    /* Save serial results to log file */
    FILE *file = fopen("floyd_results.log", "a");
    fprintf(file, "Floyd CPU | Execution Time: %f seconds | Size: %d\n", cpu_time, n_vertices);

    fclose(file);
}

/* Kernel CUDA Floyd algorithm */
__global__ void floyd_warshall_kernel(int *dev_dp, int n_vertices, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_vertices && j < n_vertices)
    {
        int ij_index = i * n_vertices + j;
        int ik_index = i * n_vertices + k;
        int kj_index = k * n_vertices + j;

        if (dev_dp[ik_index] != INFNTY && dev_dp[kj_index] != INFNTY)
        {
            int sum = dev_dp[ik_index] + dev_dp[kj_index];
            if (sum < dev_dp[ij_index])
            {
                dev_dp[ij_index] = sum;
            }
        }
    }
}

// Function to initialize GPU and run Floyd-Warshall algorithm
void floyd_warshall_parallel(int **adj_matrix, int **dp_matrix, int n_vertices)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((n_vertices + blockSize.x - 1) / blockSize.x, (n_vertices + blockSize.y - 1) / blockSize.y);

    int *dev_dp;
    cudaMalloc((void **)&dev_dp, n_vertices * n_vertices * sizeof(int));

    for (int i = 0; i < n_vertices; i++)
    {
        cudaMemcpy(dev_dp + i * n_vertices, adj_matrix[i], n_vertices * sizeof(int), cudaMemcpyHostToDevice);
    }

    clock_t start = clock(); /* Start measuring execution time */

    /* Execute kernel for each vertex, k pivot */
    for (int k = 0; k < n_vertices; k++)
    {
        floyd_warshall_kernel<<<gridSize, blockSize>>>(dev_dp, n_vertices, k);
        cudaDeviceSynchronize(); /* Sync all kernels finished */
    }

    /* Stop measuring execution time */
    clock_t end = clock();
    float gpu_time = (float)(end - start) / CLOCKS_PER_SEC;

    /* Save parallel results to log file */
    FILE *file = fopen("floyd_results.log", "a");
    fprintf(file, "Floyd GPU | Execution Time: %f seconds | Size: %d\n", gpu_time, n_vertices);
    fclose(file);

    /* Return dp matrix to CPU */
    for (int i = 0; i < n_vertices; i++)
    {
        cudaMemcpy(dp_matrix[i], dev_dp + i * n_vertices, n_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    }

    /* Free allocated memory */
    cudaFree(dev_dp);
}

int main(int argc, char **argv)
{
    int i;
    if (argc != 2)
    {
        printf("USAGE: ./floyd_comparison <number_of_vertices>\n");
        return 1;
    }

    n_vertices = atoi(argv[1]);

    dp_matrix = (int **)malloc(n_vertices * sizeof(int *));
    for (i = 0; i < n_vertices; i++)
    {
        dp_matrix[i] = (int *)malloc(n_vertices * sizeof(int));
    }

    generate_random_adj_matrix(n_vertices);

    /* Call parallel implementations */
    floyd_warshall_parallel(adjacency_matrix, dp_matrix, n_vertices);

    /* Free allocated memory */
    for (int i = 0; i < n_vertices; i++)
    {
        free(dp_matrix[i]);
    }

    free(dp_matrix);

    dp_matrix = (int **)malloc(n_vertices * sizeof(int *));
    for (i = 0; i < n_vertices; i++)
    {
        dp_matrix[i] = (int *)malloc(n_vertices * sizeof(int));
    }

    /* Call serial implementation */
    floyd_warshall_serial(adjacency_matrix, dp_matrix, n_vertices);

    /* Free allocated memory */
    for (int i = 0; i < n_vertices; i++)
    {
        free(dp_matrix[i]);
        free(adjacency_matrix[i]);
    }

    free(dp_matrix);
    free(adjacency_matrix);

    return 0;
}

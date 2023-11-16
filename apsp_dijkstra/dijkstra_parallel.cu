#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TRUE 1
#define FALSE 0
#define INFNTY INT_MAX

typedef int boolean;

/* Generates a random undirected graph represented by an adjacency matrix */
void generate_random_graph(int V, int *adjacency_matrix)
{
    srand(time(NULL));

    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (i != j)
            {
                adjacency_matrix[i * V + j] = rand() % 10;                 /* Assign a random value corresponding to the edge */
                adjacency_matrix[j * V + i] = adjacency_matrix[i * V + j]; /* Graph is undirected, the adjacency matrix is symmetric */
            }
            else
            {
                adjacency_matrix[i * V + j] = 0;
            }
        }
    }
}

__global__ void dijkstra_kernel(int V, int *graph, int *len, int *temp_distance, boolean *visited)
{
    int source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source < V)
    {
        for (int i = 0; i < V; ++i)
        {
            visited[i] = FALSE;
            temp_distance[i] = INFNTY;
            len[source * V + i] = INFNTY;
        }

        len[source * V + source] = 0;

        for (int count = 0; count < V - 1; ++count)
        {
            int current_vertex = -1;
            int min_distance = INFNTY;

            for (int v = 0; v < V; ++v)
            {
                if (!visited[v] && len[source * V + v] <= min_distance)
                {
                    min_distance = len[source * V + v];
                    current_vertex = v;
                }
            }

            visited[current_vertex] = TRUE;

            for (int v = 0; v < V; ++v)
            {
                int weight = graph[current_vertex * V + v];
                if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                    len[source * V + current_vertex] + weight < len[source * V + v])
                {
                    len[source * V + v] = len[source * V + current_vertex] + weight;
                    temp_distance[v] = len[source * V + v];
                }
            }
        }
    }
}

void dijkstra_parallel(int V, int *adjacency_matrix, int *len, int *temp_distance)
{
    boolean *d_visited;
    int *d_len, *d_temp_distance, *d_adjacency_matrix;

    /* Allocate memory on GPU */
    cudaMalloc((void **)&d_visited, V * sizeof(boolean));
    cudaMalloc((void **)&d_len, V * V * sizeof(int));
    cudaMalloc((void **)&d_temp_distance, V * sizeof(int));
    cudaMalloc((void **)&d_adjacency_matrix, V * V * sizeof(int));

    /* Copy data to GPU */
    cudaMemcpy(d_adjacency_matrix, adjacency_matrix, V * V * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x);

    clock_t start = clock(); /* Start timer */

    /* Launch CUDA kernel */
    dijkstra_kernel<<<gridSize, blockSize>>>(V, d_adjacency_matrix, d_len, d_temp_distance, d_visited);

    cudaDeviceSynchronize(); /* Sync GPU and CPU to ensure kernel finished */

    /* Copy results back to CPU */
    cudaMemcpy(len, d_len, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_distance, d_temp_distance, V * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);

    /* Free allocated memory on GPU */
    cudaFree(d_visited);
    cudaFree(d_len);
    cudaFree(d_temp_distance);
    cudaFree(d_adjacency_matrix);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra_parallel <number_of_vertices>\n");
        return 1;
    }

    int *len, *temp_distance;
    int V = atoi(argv[1]); /* Number of vertices */

    len = (int *)malloc(V * V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));

    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);
    dijkstra_parallel(V, adjacency_matrix, len, temp_distance);

    free(len);
    free(temp_distance);
    free(adjacency_matrix);

    return 0;
}

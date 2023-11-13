#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TRUE 1
#define FALSE 0
#define INFINITY INT_MAX

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

/* Print adjacency matrix */
void print_adjacency_matrix(int V, int *adjacency_matrix)
{
    printf("\nADJACENCY MATRIX:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            printf("%d ", adjacency_matrix[i * V + j]);
        }
        printf("\n");
    }
}

/* Kernel for Dijkstra */
__global__ void dijkstra_kernel(int V, int *adjacency_matrix, int *len, int *temp_distance, boolean *visited)
{
    int source = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < V)
    {
        visited[tid] = FALSE;
        temp_distance[tid] = INFINITY;
        len[source * V + tid] = INFINITY;
    }

    __syncthreads();

    if (tid == 0)
    {
        len[source * V + source] = 0;
    }

    __syncthreads();

    /* Algorithm */
    for (int count = 0; count < V - 1; count++)
    {
        int current_vertex = -1;
        int min_distance = INFINITY;

        for (int v = 0; v < V; v++)
        {
            if (!visited[v] && len[source * V + v] <= min_distance)
            {
                min_distance = len[source * V + v];
                current_vertex = v;
            }
        }

        visited[current_vertex] = TRUE;

        __syncthreads();

        for (int v = 0; v < V; v++)
        {
            int weight = adjacency_matrix[current_vertex * V + v];
            if (!visited[v] && weight && len[source * V + current_vertex] != INFINITY &&
                len[source * V + current_vertex] + weight < len[source * V + v])
            {
                len[source * V + v] = len[source * V + current_vertex] + weight;
                temp_distance[v] = len[source * V + v];
            }
        }
        __syncthreads();
    }
}

/* Finds vertex with the minimum distance among the vertices that have not been visited yet */
int find_min_distance(int V, int *distance, boolean *visited) // FIXME:
{
    int min_distance = INFINITY; /* Init value */
    int min_index = -1;

    for (int v = 0; v < V; v++) /* Iterates over all vertices */
    {
        if (!visited[v] && distance[v] <= min_distance)
        {
            min_distance = distance[v];
            min_index = v;
        }
    }
    return min_index;
}

void dijkstra_parallel(int V, int *adjacency_matrix, int *len, int *temp_distance)
{
    boolean *d_visited;
    int *d_len, *d_temp_distance;

    cudaMalloc((void **)&d_visited, V * sizeof(boolean));
    cudaMalloc((void **)&d_len, V * V * sizeof(int));
    cudaMalloc((void **)&d_temp_distance, V * sizeof(int));

    dim3 blockSize(V);
    dim3 gridSize(V);
    int sharedMemorySize = V * (sizeof(boolean) + 2 * sizeof(int));

    clock_t start = clock();

    dijkstra_kernel<<<gridSize, blockSize, sharedMemorySize>>>(V, adjacency_matrix, d_len, d_temp_distance, d_visited);

    cudaDeviceSynchronize();

    cudaMemcpy(len, d_len, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_distance, d_temp_distance, V * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);

    cudaFree(d_visited);
    cudaFree(d_len);
    cudaFree(d_temp_distance);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra <number_of_vertices>\n"); // FIXME:
        return 1;
    }

    int *len, *temp_distance;
    int V = atoi(argv[1]); /* Number of vertices */

    len = (int *)malloc(V * V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));

    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);
    dijkstra_parallel(V, adjacency_matrix, len, temp_distance);

    /* print_adjacency_matrix(V, adjacency_matrix); */

    free(len);
    free(temp_distance);
    free(adjacency_matrix);

    return 0;
}

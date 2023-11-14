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

/* Finds vertex with the minimum distance among the vertices that have not been visited yet */
int find_min_distance(int V, int *distance, boolean *visited)
{
    int min_distance = INFNTY; /* Init value */
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

void dijkstra_serial(int V, int *adjacency_matrix, int *len, int *temp_distance)
{
    boolean *visited = (boolean *)malloc(V * sizeof(boolean));

    clock_t start = clock(); /* Records the start time for measuring the execution time */

    /* Computing the All Pairs Shortest Paths (APSP) in the graph */
    for (int source = 0; source < V; source++)
    {
        for (int i = 0; i < V; i++) /* Initialize vars arrays to current source */
        {
            visited[i] = FALSE;
            temp_distance[i] = INFNTY;
            len[source * V + i] = INFNTY;
        }

        len[source * V + source] = 0; /* Set the distance of the source vertex as 0 */

        for (int count = 0; count < V - 1; count++)
        {
            /* Finds the vertex with the minimum distance from the current source vertex */
            int current_vertex = find_min_distance(V, len + source * V, visited);

            visited[current_vertex] = TRUE;

            for (int v = 0; v < V; v++)
            {
                int weight = adjacency_matrix[current_vertex * V + v];
                if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                    len[source * V + current_vertex] + weight < len[source * V + v])
                {
                    /* Updating the distance is beneficial */
                    len[source * V + v] = len[source * V + current_vertex] + weight;
                    temp_distance[v] = len[source * V + v];
                }
            }
        }
    }

    /* Records the end time for measuring the execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON CPU = %f SECS\n", seconds);

    free(visited);
}

/* Kernel for Dijkstra */
__global__ void dijkstra_kernel(int V, int *adjacency_matrix, int *len, int *temp_distance, boolean *visited)
{
    int source = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < V) /* Init arrays for current thread */
    {
        visited[tid] = FALSE;
        temp_distance[tid] = INFNTY;
        len[source * V + tid] = INFNTY;
    }

    __syncthreads();

    if (tid == 0) /* Set distance of the source vertex to zero */
    {
        len[source * V + source] = 0;
    }

    __syncthreads();

    /* Start Dijkstra algorithm */
    for (int count = 0; count < V - 1; count++)
    {
        int current_vertex = -1;
        int min_distance = INFNTY;

        /* Find vertex with min distance among unvisited */
        for (int v = 0; v < V; v++)
        {
            if (!visited[v] && len[source * V + v] <= min_distance)
            {
                min_distance = len[source * V + v];
                current_vertex = v;
            }
        }
        visited[current_vertex] = TRUE; /* Current marked as visited */

        __syncthreads();

        for (int v = 0; v < V; v++) /* Update dist for neighboring vertices */
        {
            int weight = adjacency_matrix[current_vertex * V + v];
            if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                len[source * V + current_vertex] + weight < len[source * V + v])
            {
                /* Update dist if shorter path is found */
                len[source * V + v] = len[source * V + current_vertex] + weight;
                temp_distance[v] = len[source * V + v];
            }
        }
        __syncthreads();
    }
}

void dijkstra_parallel(int V, int *adjacency_matrix, int *len, int *temp_distance)
{
    boolean *d_visited;
    int *d_len, *d_temp_distance;

    /* Allocate memory on GPU */
    cudaMalloc((void **)&d_visited, V * sizeof(boolean));
    cudaMalloc((void **)&d_len, V * V * sizeof(int));
    cudaMalloc((void **)&d_temp_distance, V * sizeof(int));

    dim3 blockSize(V); /* 1D block with V threads */
    dim3 gridSize(V);  /* 1D grid with V blocks */
    int sharedMemorySize = V * (sizeof(boolean) + 2 * sizeof(int));

    clock_t start = clock(); /* Start timer */

    /* Launch CUDA kernel */
    dijkstra_kernel<<<gridSize, blockSize, sharedMemorySize>>>(V, adjacency_matrix, d_len, d_temp_distance, d_visited);

    cudaDeviceSynchronize(); /* Sync GPU and CPU to ensure kernel finished*/

    cudaMemcpy(len, d_len, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_distance, d_temp_distance, V * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);

    /* Free allocated memory on GPU */
    cudaFree(d_visited);
    cudaFree(d_len);
    cudaFree(d_temp_distance);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra_comparison <number_of_vertices>\n");
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

    len = (int *)malloc(V * V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));
    dijkstra_serial(V, adjacency_matrix, len, temp_distance);

    /* print_adjacency_matrix(V, adjacency_matrix); */

    free(len);
    free(temp_distance);
    free(adjacency_matrix);

    return 0;
}

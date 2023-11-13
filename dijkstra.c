#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

#define TRUE 1
#define FALSE 0
#define INFINITY INT_MAX

typedef int boolean;

void generate_random_graph(int V, int *adjacency_matrix)
{
    srand(time(NULL));

    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            if (i != j)
            {
                adjacency_matrix[i * V + j] = rand() % 10;
                adjacency_matrix[j * V + i] = adjacency_matrix[i * V + j];
            }
            else
            {
                adjacency_matrix[i * V + j] = 0;
            }
        }
    }
}

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

int find_min_distance(int V, int *distance, boolean *visited)
{
    int min_distance = INFINITY;
    int min_index = -1;

    for (int v = 0; v < V; v++)
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

    clock_t start = clock();

    for (int source = 0; source < V; source++)
    {
        for (int i = 0; i < V; i++)
        {
            visited[i] = FALSE;
            temp_distance[i] = INFINITY;
            len[source * V + i] = INFINITY;
        }

        len[source * V + source] = 0; /* Set the distance of the source vertex as 0 */

        for (int count = 0; count < V - 1; count++)
        {
            int current_vertex = find_min_distance(V, len + source * V, visited);

            visited[current_vertex] = TRUE;

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
        }
    }

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON CPU = %f SECS\n", seconds);

    free(visited);
}

int main(int argc, char **argv)
{
    int *len, *temp_distance;

    /* Usage case */
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra <number_of_vertices>\n");
        return 1;
    }

    int V = atoi(argv[1]); /* Number of vertices */

    len = (int *)malloc(V * V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));

    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);

    // print_adjacency_matrix(V, adjacency_matrix);

    dijkstra_serial(V, adjacency_matrix, len, temp_distance);

    free(len);
    free(temp_distance);
    free(adjacency_matrix);

    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define TRUE 1
#define FALSE 0
#define INFINITY 999999

typedef int boolean;

typedef struct
{
    int u;
    int v;
} Edge;

typedef struct
{
    int identifier;
    boolean visited;
} Vertex;

Vertex *vertices;
Edge *edges;
int *weights;

int find_edge(Vertex u, Vertex v, Edge *edges, int *weights, int E)
{
    for (int i = 0; i < E; i++)
    {
        if (edges[i].u == u.identifier && edges[i].v == v.identifier)
        {
            return weights[i];
        }
    }
    return INFINITY;
}

/* Random graph with number of vertices V and edges E */
void generate_graph(int V, int E)
{
    srand(time(NULL)); // Random numbers generation
    /* Create vertices */
    for (int i = 0; i < V; i++)
    {
        Vertex vertex = {.identifier = (int)i, .visited = FALSE};
        vertices[i] = vertex;
    }
    /* Create edges */
    for (int i = 0; i < E; i++)
    {
        /* Edge created and initialized with two random vertices */
        Edge edg = {.u = (int)rand() % V, .v = rand() % V};
        edges[i] = edg;
        weights[i] = rand() % 10; /* Random weight*/
    }
}

void dijkstra_serial(int V, int E, int *len, int *temp_distance)
{
    Vertex *root = (Vertex *)malloc(sizeof(Vertex) * V);

    for (int count = 0; count < V; count++)
    {
        root[count].identifier = count;
        root[count].visited = FALSE;
    }

    clock_t start = clock();

    /* Each vertex in the graph */
    for (int count = 0; count < V; count++)
    {
        /* Current vertex visited, init len array */
        root[count].visited = TRUE;
        len[root[count].identifier] = 0;
        temp_distance[root[count].identifier] = 0;

        /* Compute for vertices not equal to root */
        for (int i = 0; i < V; i++)
        {
            if (vertices[i].identifier != root[count].identifier)
            {
                len[(int)vertices[i].identifier] = find_edge(root[count], vertices[i], edges, weights, E);
                temp_distance[vertices[i].identifier] = len[(int)vertices[i].identifier];
            }
            else
            {
                vertices[i].visited = TRUE;
            }
        }

        /* Update distance based on neighboring vertices and their weights */
        for (int i = 0; i < V; i++)
        {
            if (vertices[i].visited == FALSE)
            {
                vertices[i].visited = TRUE;
                for (int v = 0; v < V; v++)
                {
                    int weight = find_edge(vertices[i], vertices[v], edges, weights, E);
                    if (weight < INFINITY)
                    {
                        if (temp_distance[v] > len[i] + weight)
                        {
                            temp_distance[v] = len[i] + weight;
                        }
                    }
                }
            }

            for (int j = 0; j < V; j++)
            {
                if (len[j] > temp_distance[j])
                {
                    vertices[j].visited = FALSE;
                    len[j] = temp_distance[j];
                }
                temp_distance[j] = len[j];
            }
        }
    }

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time on CPU = %f sec\n", seconds);

    free(root);
}

int main(int argc, char **argv)
{
    int *len, *temp_distance;

    int V = atoi(argv[1]); /* Number of vertices */
    int E = atoi(argv[2]); /* Number of edges */

    vertices = (Vertex *)malloc(sizeof(Vertex) * V);
    edges = (Edge *)malloc(sizeof(Edge) * E);
    weights = (int *)malloc(E * sizeof(int));

    generate_graph(V, E); /* Create random graph */

    len = (int *)malloc(V * sizeof(int));
    temp_distance = (int *)malloc(V * sizeof(int));

    dijkstra_serial(V, E, len, temp_distance);

    free(vertices);
    free(edges);
    free(weights);
    free(len);
    free(temp_distance);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define INFNTY INT_MAX

int **adjacency_matrix, **dp_matrix;

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
                adjacency_matrix[i][j] = 0; /* Diagonal */
            }
            else
            {
                /* Zero to nine random */
                int r = rand() % 10;
                int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
                adjacency_matrix[i][j] = val;    /* Symmetrically */
                adjacency_matrix[j][i] = val;
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
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./floyd_serial <number_of_vertices>\n");
        return 1;
    }

    int n_vertices;
    n_vertices = atoi(argv[1]);

    // Allocate memory for dp matrix
    dp_matrix = (int **)malloc(n_vertices * sizeof(int *));
    for (int i = 0; i < n_vertices; i++)
    {
        dp_matrix[i] = (int *)malloc(n_vertices * sizeof(int));
    }

    generate_random_adj_matrix(n_vertices);

    clock_t start = clock(); /* Start measuring execution time */

    floyd_warshall_serial(adjacency_matrix, dp_matrix, n_vertices);

    /* Stop measuring execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON CPU = %f SECS\n", seconds);

    /* Free allocated memory */
    for (int i = 0; i < n_vertices; i++)
    {
        free(adjacency_matrix[i]);
        free(dp_matrix[i]);
    }
    free(adjacency_matrix);
    free(dp_matrix);

    return 0;
}

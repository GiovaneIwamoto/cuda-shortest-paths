# CUDA SHORTEST PATHS

### **OVERVIEW**

All-pairs shortest path (or APSP) problem requires finding the shortest path between all pairs of nodes in a graph.

https://en.wikipedia.org/wiki/Parallel_all-pairs_shortest_path_algorithm#:~:text=A%20central%20problem%20in%20algorithmic,%2Dpaths%20(APSP)%20problem.

### **COMPILE**

**DIJKSTRA**

gcc -o dijkstra_serial .\dijkstra_serial.c -std=c99 -lm -Wall -Wextra -g3
.\dijkstra_serial.exe 1000

nvcc -o dijkstra_parallel .\dijkstra_parallel.cu
.\dijkstra_parallel.exe 1000

nvcc -o dijkstra_comparison .\dijkstra_comparison.cu
.\dijkstra_comparison.exe 1000

**FLOYD WARSHALL**

gcc -o floyd_serial .\floyd_serial.c -std=c99 -lm -Wall -Wextra -g3
.\floyd_serial.exe 1000

nvcc -o floyd_parallel .\floyd_parallel.cu
.\floyd_parallel.exe 1000

nvcc -o floyd_comparison .\floyd_comparison.cu
.\floyd_comparison.exe 1000

NVIDIA GeForce MX250

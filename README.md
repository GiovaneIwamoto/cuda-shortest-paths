# CUDA SHORTEST PATHS

### **OVERVIEW**

All-pairs shortest path (or APSP) problem requires finding the shortest path between all pairs of nodes in a graph.

"Sequential Dijkstra's Algorithm"
"Parallel Partitioned Source Dijkstra Algorithm"
"Parallel Parallel Source Dijkstra Algorithm"

https://en.wikipedia.org/wiki/Parallel_all-pairs_shortest_path_algorithm#:~:text=A%20central%20problem%20in%20algorithmic,%2Dpaths%20(APSP)%20problem.

gcc -o dijkstra_serial .\dijkstra_serial.c -std=c99 -lm -Wall -Wextra -g3
.\dijkstra_serial.exe 10

nvcc -o dijkstra_parallel .\dijkstra_parallel.cu
.\dijkstra_parallel.exe 10

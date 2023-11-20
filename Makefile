CC = gcc
NVCC = nvcc
CFLAGS = -std=c99 -lm -Wall -Wextra -g3

all: dijkstra_serial dijkstra_parallel dijkstra_comparison floyd_serial floyd_parallel floyd_comparison

dijkstra_serial: dijkstra_serial.c
	$(CC) -o dijkstra_serial dijkstra_serial.c $(CFLAGS)

dijkstra_parallel: dijkstra_parallel.cu
	$(NVCC) -o dijkstra_parallel dijkstra_parallel.cu

dijkstra_comparison: dijkstra_comparison.cu
	$(NVCC) -o dijkstra_comparison dijkstra_comparison.cu

floyd_serial: floyd_serial.c
	$(CC) -o floyd_serial floyd_serial.c $(CFLAGS)

floyd_parallel: floyd_parallel.cu
	$(NVCC) -o floyd_parallel floyd_parallel.cu

floyd_comparison: floyd_comparison.cu
	$(NVCC) -o floyd_comparison floyd_comparison.cu

clean:
	rm -f dijkstra_serial dijkstra_parallel dijkstra_comparison floyd_serial floyd_parallel floyd_comparison

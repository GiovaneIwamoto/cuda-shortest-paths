CC = gcc
CFLAGS = -std=c99 -lm

dijkstra_apsp_serial: dijkstra.c
    $(CC) $^ -o $@ $(CFLAGS)

clean:
    rm -f apsp2


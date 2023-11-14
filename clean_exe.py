import os

# Files to delete
files_to_delete = ["apsp_dijkstra/dijkstra_parallel.exe", "apsp_dijkstra/dijkstra_parallel.exp",
                   "apsp_dijkstra/dijkstra_parallel.lib", "apsp_dijkstra/dijkstra_serial.exe",
                   "apsp_dijkstra/dijkstra_comparison.exe", "apsp_dijkstra/dijkstra_comparison.exp",
                   "apsp_dijkstra/dijkstra_comparison.lib", "apsp_floyd/floyd_serial.exe",
                   "apsp_floyd/floyd_parallel.exe", "apsp_floyd/floyd_parallel.exp",
                   "apsp_floyd/floyd_parallel.lib", "apsp_floyd/floyd_comparison.exe",
                   "apsp_floyd/floyd_comparison.exp", "apsp_floyd/floyd_comparison.lib"]

# Directory path for files
directory_path = os.path.dirname(os.path.abspath(__file__))

for file_name in files_to_delete:
    file_path = os.path.join(directory_path, file_name)
    try:
        os.remove(file_path)
        print(f"File {file_name} removed")

    except FileNotFoundError:
        print(f"File {file_name} not found")

    except Exception as e:
        print(f"Error removing {file_name}: {e}")

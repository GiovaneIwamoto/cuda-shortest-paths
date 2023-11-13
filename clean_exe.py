import os

# Files to delete
files_to_delete = ["dijkstra_parallel.exe", "dijkstra_parallel.exp",
                   "dijkstra_parallel.lib", "dijkstra_serial.exe",
                   "dijkstra_comparison.exe", "dijkstra_comparison.exp",
                   "dijkstra_comparison.lib"]

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

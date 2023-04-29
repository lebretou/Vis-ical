import os
import random

def get_svg_paths(root_path):
    svg_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        filenames.sort()
        for filename in filenames:
            if filename.endswith('svg') and not(filename.startswith('example')) and not (filename.startswith('instruction')):
                svg_paths.append('./python'+os.path.join(dirpath, filename)[1:])
    return svg_paths

def shuffle_range(arr, start, end):
    sub_arr = arr[start:end]
    random.shuffle(sub_arr)
    arr[start:end] = sub_arr

root_path = "./"

svg_paths = get_svg_paths(root_path)

indices = [[0,1,2,3,4,5],
           [1,2,3,4,5,0],
           [2,3,4,5,0,1],
           [3,4,5,0,1,2],
           [4,5,0,1,2,3],
           [5,0,1,2,3,4]]
random.shuffle(indices)

paths = []
# 0, 7, 14, ..
for i in range(6):
    for j in range(6):
        k = j * 6
        index = indices[j][i]
        paths.append(svg_paths[index + k])

for i in range(6):
    k = i * 6
    shuffle_range(paths, k, k + 6)    


with open('svg_paths.txt', "w") as file:
    for path in paths:
        file.write(path + "\n")



file.close()
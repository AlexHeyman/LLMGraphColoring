'''
When run, this file iterates through the problem sets named in metadata.py and,
for each problem data file it finds in the data directory, it calculates the
problem's greedy score (if the problem is colorable) or checks for the presence
of a complete subgraph of size k + 1 (if the problem is uncolorable) and adds
the result to the data file.

This file does not run with any arguments.
'''

from os import path, listdir

from metadata import *
from utils import compute_vertex_conns, compute_greedy_score,\
     find_complete_subgraph

for problem_set in problem_sets:
  ps_short_name = problem_set['short_name']
  num_vertices = problem_set['num_vertices']
  num_colors = problem_set['num_colors']
  
  ps_data_dir = path.join(data_dir, ps_short_name)
  
  if not path.exists(ps_data_dir):
    continue
  
  print('Categorizing %s' % ps_short_name)
  
  filenames = [filename for filename in listdir(ps_data_dir)]
  
  for index, filename in enumerate(filenames):
    if index % 100 == 0:
      print('%d of %d' % (index, len(filenames)))
    
    data_file_path = path.join(ps_data_dir, filename)
    
    # Read
    
    data_file = open(data_file_path, 'r')
    
    lines = data_file.read().splitlines()
    
    data_file.close()
    
    edges = [tuple(int(num) for num in edge.split(','))
             for edge in lines[0].split('|')]
    is_possible = (lines[1] == 'True')
    
    # Compute greedy score / presence of complete subgraph
    
    vertex_conns = compute_vertex_conns(num_vertices, edges)
    
    if is_possible:
      gs = compute_greedy_score(num_vertices, vertex_conns, num_colors)
    else:
      sg = find_complete_subgraph(num_vertices, set(edges), vertex_conns,
                                  num_colors + 1)
    
    # Write
    
    data_file = open(data_file_path, 'w')
    
    edges_str = '|'.join(('%d,%d' % (edge[0], edge[1])) for edge in edges)
    print(edges_str, file=data_file)
    
    print(is_possible, file=data_file)
    
    if is_possible:
      print('%.4f' % gs, file=data_file)
    else:
      print((sg is not None), file=data_file)
    
    data_file.close()

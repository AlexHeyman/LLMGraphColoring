'''
When run, this file generates sets of graph coloring problems based on the
configuration in metadata.py, and stores basic data about the problems in the
data directory and prompts based on the problems in the prompt directory (again,
see metadata.py).

This file does not run with any arguments.
'''

from os import path, mkdir
import math
import itertools
import random

from metadata import *
from utils import sample_combinations, compute_vertex_conns, color_greedy

math_template = '''Consider an undirected graph with %d vertices (numbered 0 through %d) and the following set of edges:

{%s}

Suppose that we want to color every vertex either %s so that no two adjacent vertices receive the same color. Is this possible? If it is impossible, write "Impossible" as the final line of your response. If it is possible, the final lines of your response should present a plan for it in a format like the following:

%s
(etc.)'''

math_demanding_template = '''Consider an undirected graph with %d vertices (numbered 0 through %d) and the following set of edges:

{%s}

Color every vertex either %s so that no two adjacent vertices receive the same color, or if this is impossible, say so. If it is impossible, write "Impossible" as the final line of your response. If it is possible, the final lines of your response should present your vertex coloring in a format like the following:

%s
(etc.)'''

cities_template = '''Suppose there is a country with %d cities (numbered 1 through %d) and a highway system where each highway connects exactly two cities. %s %s.

Suppose the government wants to erect a monument in each city. Each monument will be to either %s. The government wants it so, if any pair of cities are directly connected by a highway, the monuments in those cities cannot both be to the same person. Is this possible? If it is impossible, write "Impossible" as the final line of your response. If it is possible, the final lines of your response should present a plan for it in a format like the following:

%s
(etc.)'''

friends_template = '''Imagine %d people: %s. Suppose that the following friendships exist between them: %s.

Suppose that the %d people are all going to attend a party, and each of them is going to wear either %s. Suppose that none of them want to wear the same color shirt as anyone they are friends with. Is this possible? If it is impossible, write "Impossible" as the final line of your response. If it is possible, the final lines of your response should present a plan for it in a format like the following:

%s
(etc.)
'''

for problem_set in problem_sets:
  ps_name = problem_set['name']
  ps_short_name = problem_set['short_name']
  num_vertices = problem_set['num_vertices']
  num_colors = problem_set['num_colors']
  
  print('Generating problem set: %s' % ps_name)
  
  possible_edges = list(itertools.combinations(range(num_vertices), 2))
  
  if 'edge_counts' in problem_set:
    edge_counts = problem_set['edge_counts']
  else:
    edge_counts = list(range(1, len(possible_edges) + 1))
  
  if 'max_samples' in problem_set:
    max_samples = problem_set['max_samples']
  else:
    max_samples = -1
  
  if 'max_samples_per_ec' in problem_set:
    max_samples_per_ec = problem_set['max_samples_per_ec']
  else:
    max_samples_per_ec = -1
  
  if 'random_method' in problem_set:
    random_method = problem_set['random_method']
  else:
    random_method = 'efficient'
  
  # Make data and prompt directories for this problem set
  ps_data_dir = path.join(data_dir, ps_short_name)
  if not path.exists(ps_data_dir):
    mkdir(ps_data_dir)
  
  ps_prompt_dir = path.join(prompt_dir, ps_short_name)
  if not path.exists(ps_prompt_dir):
    mkdir(ps_prompt_dir)
  
  # Make subdirectories for frames
  for frame in frames:
    frame_dir = path.join(ps_prompt_dir, frame)
    if not path.exists(frame_dir):
      mkdir(frame_dir)
  
  if 'random_seed' in problem_set:
    random.seed(problem_set['random_seed'])
  
  if random_method == 'efficient':
    max_num_ec = [math.comb(len(possible_edges), num_edges)
                  for num_edges in edge_counts]
    num_ec = list(max_num_ec)
    
    if max_samples_per_ec >= 0:
      for ec_index in range(len(edge_counts)):
        if max_samples_per_ec < num_ec[ec_index]:
          num_ec[ec_index] = max_samples_per_ec
    
    total_combinations = sum(num_cs for num_cs in num_ec)
    
    if max_samples >= 0 and max_samples < total_combinations:
      # Shrink num_ec to have a total size of max_samples, maintaining
      # proportionality between edge counts
      sample_counts = []
      signed_errors = []
      total_samples = 0
      
      for ec_index in range(len(edge_counts)):
        proportion = max_samples * num_ec[ec_index] / total_combinations
        sample_count = round(proportion)
        sample_counts.append(sample_count)
        signed_errors.append(sample_count - proportion)
        total_samples += sample_count
      
      while total_samples < max_samples:
        smallest_error_index = 0
        
        for ec_index in range(1, len(edge_counts)):
          if signed_errors[ec_index] < signed_errors[smallest_error_index]:
            smallest_error_index = ec_index
        
        sample_counts[smallest_error_index] += 1
        signed_errors[smallest_error_index] += 1
        total_samples += 1
      
      while total_samples > max_samples:
        largest_error_index = 0
        
        for ec_index in range(1, len(edge_counts)):
          if signed_errors[ec_index] > signed_errors[largest_error_index]:
            largest_error_index = ec_index

        sample_counts[largest_error_index] -= 1
        signed_errors[largest_error_index] -= 1
        total_samples -= 1
      
      for ec_index in range(len(edge_counts)):
        num_ec[ec_index] = sample_counts[ec_index]
    
    edge_combinations = []
    for ec_index in range(len(edge_counts)):
      num_edges = edge_counts[ec_index]
      
      if num_ec[ec_index] >= max_num_ec[ec_index]:
        ec_combs = list(itertools.combinations(possible_edges, num_edges))
      else:
        ec_combs = sample_combinations(
          possible_edges, num_edges, num_ec[ec_index])
      
      edge_combinations.append(ec_combs)
    
  else: # random_method == 'legacy'
    edge_combinations = [list(itertools.combinations(possible_edges, num_edges))
                         for num_edges in edge_counts]
    
    if max_samples_per_ec >= 0:
      for ec_index in range(len(edge_counts)):
        if max_samples_per_ec < len(edge_combinations[ec_index]):
          # Turn edge_combinations[ec_index] into a subsample of itself of size
          # max_samples_per_ec
          sample_indices = random.sample(
            range(len(edge_combinations[ec_index])), max_samples_per_ec)
          edge_combinations[ec_index] = [edge_combinations[ec_index][si]
                                         for si in sample_indices]
    
    total_combinations = sum(len(cs) for cs in edge_combinations)
    
    if max_samples >= 0 and max_samples < total_combinations:
      # Turn edge_combinations into a subsample of itself of size max_samples,
      # maintaining proportionality between edge counts
      sample_counts = []
      signed_errors = []
      total_samples = 0
      
      for ec_index in range(len(edge_counts)):
        proportion = max_samples * len(edge_combinations[ec_index])\
                     / total_combinations
        sample_count = round(proportion)
        sample_counts.append(sample_count)
        signed_errors.append(sample_count - proportion)
        total_samples += sample_count
      
      while total_samples < max_samples:
        smallest_error_index = 0
        
        for ec_index in range(1, len(edge_counts)):
          if signed_errors[ec_index] < signed_errors[smallest_error_index]:
            smallest_error_index = ec_index
        
        sample_counts[smallest_error_index] += 1
        signed_errors[smallest_error_index] += 1
        total_samples += 1
      
      while total_samples > max_samples:
        largest_error_index = 0
        
        for ec_index in range(1, len(edge_counts)):
          if signed_errors[ec_index] > signed_errors[largest_error_index]:
            largest_error_index = ec_index

        sample_counts[largest_error_index] -= 1
        signed_errors[largest_error_index] -= 1
        total_samples -= 1
      
      for ec_index in range(len(edge_counts)):
        sample_indices = random.sample(range(len(edge_combinations[ec_index])),
                                       sample_counts[ec_index])
        edge_combinations[ec_index] = [edge_combinations[ec_index][si]
                                       for si in sample_indices]
  
  num_problems = 0
  num_possible = 0
  
  problem_number = 1
  for ec_index in range(len(edge_counts)):
    ec_num_problems = 0
    ec_num_possible = 0
    
    num_edges = edge_counts[ec_index]
    
    for edges in edge_combinations[ec_index]:
      ec_num_problems += 1
      
      # Make data file
      data_file_path = path.join(ps_data_dir, '%d.txt' % problem_number)
      data_file = open(data_file_path, 'w')
      
      edges_str = '|'.join(('%d,%d' % (edge[0], edge[1])) for edge in edges)
      print(edges_str, file=data_file)

      vertex_conns = compute_vertex_conns(num_vertices, edges)
      coloring = color_greedy(num_vertices, vertex_conns, num_colors)
      is_possible = (coloring is not None)
      
      if is_possible:
        ec_num_possible += 1
      
      print(is_possible, file=data_file)
      
      data_file.close()
      
      # Make prompt file for math frame
      math_file_path = path.join(ps_prompt_dir, 'math',
                                 '%d.txt' % problem_number)
      math_file = open(math_file_path, 'w')

      edges_str = ', '.join(('(%d,%d)' % (edge[0], edge[1])) for edge in edges)

      if num_colors == 2:
        colors_str = '%s or %s' % (math_colors[0].lower(),
                                   math_colors[1].lower())
      else:
        colors_str_segs = list(math_colors_lower[:num_colors])
        colors_str_segs[-1] = 'or ' + colors_str_segs[-1]
        colors_str = ', '.join(colors_str_segs)

      example_str = '\n'.join(
        ('%d %s' % (i, math_colors[i % num_colors]))\
        for i in range(max(3, num_colors)))
      
      prompt = math_template % (num_vertices, num_vertices - 1, edges_str,
                                colors_str, example_str)
      math_file.write(prompt)
      
      math_file.close()

      # Make prompt file for math_demanding frame
      math_d_file_path = path.join(ps_prompt_dir, 'math_demanding',
                                   '%d.txt' % problem_number)
      math_d_file = open(math_d_file_path, 'w')

      edges_str = ', '.join(('(%d,%d)' % (edge[0], edge[1])) for edge in edges)

      if num_colors == 2:
        colors_str = '%s or %s' % (math_colors[0].lower(),
                                   math_colors[1].lower())
      else:
        colors_str_segs = list(math_colors_lower[:num_colors])
        colors_str_segs[-1] = 'or ' + colors_str_segs[-1]
        colors_str = ', '.join(colors_str_segs)

      example_str = '\n'.join(
        ('%d %s' % (i, math_colors[i % num_colors]))\
        for i in range(max(3, num_colors)))
      
      prompt = math_demanding_template % (num_vertices, num_vertices - 1,
                                          edges_str, colors_str, example_str)
      math_d_file.write(prompt)
      
      math_d_file.close()

      # Make prompt file for cities frame
      cities_file_path = path.join(ps_prompt_dir, 'cities',
                                   '%d.txt' % problem_number)
      cities_file = open(cities_file_path, 'w')

      if num_edges >= 2:
        edges_intro = 'The highways are:'
      else:
        edges_intro = 'There is one highway,'
      
      edges_str_segs = list(('between city %d and city %d'\
                             % (edge[0] + 1, edge[1] + 1)) for edge in edges)
      if num_edges >= 2:
        edges_str_segs[-1] = 'and ' + edges_str_segs[-1]
      
      edges_str = ', '.join(edges_str_segs)

      if num_colors == 2:
        colors_str = '%s or %s' % (cities_colors_full[0], cities_colors_full[1])
      else:
        colors_str_segs = list(cities_colors_full[:num_colors])
        colors_str_segs[-1] = 'or ' + colors_str_segs[-1]
        colors_str = ', '.join(colors_str_segs)

      example_str = '\n'.join(
        ('%d %s' % (i + 1, cities_colors[i % num_colors]))\
        for i in range(max(3, num_colors)))
      
      prompt = cities_template % (num_vertices, num_vertices, edges_intro,
                                  edges_str, colors_str, example_str)
      cities_file.write(prompt)
      
      cities_file.close()

      # Make prompt file for friends frame
      friends_file_path = path.join(ps_prompt_dir, 'friends',
                                   '%d.txt' % problem_number)
      friends_file = open(friends_file_path, 'w')

      vertices_str_segs = list(friends_names[:num_vertices])
      vertices_str_segs[-1] = 'and ' + vertices_str_segs[-1]
      vertices_str = ', '.join(vertices_str_segs)
      
      edges_str_segs = list(('%s is friends with %s'\
                             % (friends_names[edge[0]],
                                friends_names[edge[1]])) for edge in edges)
      if num_edges >= 2:
        edges_str_segs[-1] = 'and ' + edges_str_segs[-1]
      
      edges_str = ', '.join(edges_str_segs)

      if num_colors == 2:
        colors_str = 'a %s shirt or a %s shirt'\
                     % (friends_colors_lower[0], friends_colors_lower[1])
      else:
        colors_str_segs = list(('a %s shirt' % c)
                               for c in friends_colors_lower[:num_colors])
        colors_str_segs[-1] = 'or ' + colors_str_segs[-1]
        colors_str = ', '.join(colors_str_segs)

      example_str = '\n'.join(
        ('%s: %s' % (friends_names[i], friends_colors[i % num_colors]))\
        for i in range(max(3, num_colors)))
      
      prompt = friends_template % (num_vertices, vertices_str, edges_str,
                                   num_vertices, colors_str, example_str)
      friends_file.write(prompt)
      
      friends_file.close()
      
      problem_number += 1

    print('%d edges: %d problems (%.1f%% possible)'\
          % (num_edges, ec_num_problems,
             (ec_num_possible / ec_num_problems) * 100))
    
    num_problems += ec_num_problems
    num_possible += ec_num_possible
  
  print('Total: %d problems (%.1f%% possible)'\
          % (num_problems, (num_possible / num_problems) * 100))

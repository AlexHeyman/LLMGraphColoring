'''
When run, this file reads through the responses in the response directory (see
metadata.py), parses them, evaluates the answers it finds, and records the
results in the evaluation directory. The response directory is assumed to have
a structure matching what generate.py generates in the prompt directory,
optionally with subdirectories for multiple repeats (see ../test.py). The file
overrides.txt directly inside the response directory, if it exists, is read in
at the beginning of execution and can specify overrides to the automatic
parser's output for specific response files. At the end of execution, any files
in which the automatic parser did not detect a coherent answer are added to
overrides.txt for optional manual review.

This file does not run with any arguments.
'''

from os import path, listdir
import re

from metadata import *
from utils import coloring_to_string, string_to_coloring, coloring_is_valid

def get_clean_tokens(string):
  # Replace stuff like "\text" with spaces
  string = re.sub(r'\\([a-zA-Z0-9])+', ' ', string)
  
  # Replace non-alphanumeric characters with spaces
  string = re.sub(r'[^a-zA-Z0-9]', ' ', string)

  # Convert all letters to lowercase
  string = string.lower()
  
  # Split around whitespace
  return string.split()

def get_math_coloring(file_path, num_vertices, connected_vertices, num_colors):
  vertices = {v: i for i, v in enumerate(str(i) for i in range(num_vertices))}
  
  file = open(file_path, 'r', encoding='utf-8')
  lines = file.read().splitlines()
  file.close()
  
  tokens = []
  for line in lines:
    line_tokens = get_clean_tokens(line)
    if len(line_tokens) >= 1:
      tokens.append(line_tokens)
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0] == 'vertex':
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      line_tokens = line_tokens[1:]
      
      if len(line_tokens) >= 2 and line_tokens[0] == 'is':
        line_tokens = line_tokens[1:]
      
      if len(line_tokens) >= 2 and line_tokens[0] == 'colored':
        line_tokens = line_tokens[1:]
      
      for j in range(num_colors):
        if math_colors_lower[j].startswith(line_tokens[0]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        coloring[vertex] = 0 # Arbitrary
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary

        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

def get_cities_coloring(file_path, num_vertices,
                        connected_vertices, num_colors):
  vertices = {v: i for i, v in enumerate(
    str(i) for i in range(1, num_vertices + 1))}
  vp_index = cities_colors_lower.index('vp')
  
  file = open(file_path, 'r', encoding='utf-8')
  lines = file.read().splitlines()
  file.close()
  
  tokens = []
  for line in lines:
    line_tokens = get_clean_tokens(line)
    if len(line_tokens) >= 1:
      tokens.append(line_tokens)
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0] == 'city':
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 \
    and line_tokens[0].isdigit() and line_tokens[1] == 'city':
      line_tokens = line_tokens[2:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      for j in range(num_colors):
        if cities_colors_lower[j].startswith(line_tokens[1]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        if vp_index < num_colors and len(line_tokens) >= 3 \
        and line_tokens[1] == 'vice' and line_tokens[2] == 'president':
          coloring[vertex] = vp_index
          non_arbitrary_vertices += 1
        else:
          coloring[vertex] = 0 # Arbitrary
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary

        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

def get_friends_coloring(file_path, num_vertices,
                         connected_vertices, num_colors):
  vertices = {v: i for i, v in enumerate(friends_names_lower[:num_vertices])}
  
  file = open(file_path, 'r', encoding='utf-8')
  lines = file.read().splitlines()
  file.close()
  
  tokens = []
  for line in lines:
    line_tokens = get_clean_tokens(line)
    if len(line_tokens) >= 1:
      tokens.append(line_tokens)
  
  coloring = {}
  non_arbitrary_vertices = 0
  
  i = len(tokens) - 1
  
  if 'impossible' in tokens[i]:
    return 'impossible'
  
  while i >= 0:
    if len(tokens[i]) == 1 and tokens[i][0] == 'impossible':
      return 'impossible'
    
    line_tokens = tokens[i]
    
    if line_tokens[0].isdigit():
      line_tokens = line_tokens[1:]
    
    if len(line_tokens) >= 2 and line_tokens[0] in vertices:
      vertex = vertices[line_tokens[0]]
      for j in range(num_colors):
        if friends_colors_lower[j].startswith(line_tokens[1]):
          coloring[vertex] = j
          non_arbitrary_vertices += 1
          break
      else:
        coloring[vertex] = 0 # Arbitrary
      
      if len(coloring) == num_vertices:
        if non_arbitrary_vertices > 0:
          return coloring
        else:
          coloring = {}
          non_arbitrary_vertices = 0
    elif len(coloring) > 0:
      coloring_vertices = set(coloring)
      if len(connected_vertices - coloring_vertices) == 0\
      and non_arbitrary_vertices > 0:
        # Coloring includes all connected vertices
        for vertex in range(num_vertices):
          if vertex not in coloring:
            coloring[vertex] = 0 # Arbitrary
        
        return coloring
      else:
        coloring = {}
        non_arbitrary_vertices = 0
    
    i -= 1
  
  # If a partial coloring was found right at the beginning of the file,
  # check for inclusion of all connected vertices
  coloring_vertices = set(coloring)
  if len(connected_vertices - coloring_vertices) == 0:
    for vertex in range(num_vertices):
      if vertex not in coloring:
        coloring[vertex] = 0
    
    return coloring
  
  return 'not_found'

frame_coloring_funcs = [get_math_coloring, get_math_coloring,
                        get_cities_coloring, get_friends_coloring]

overrides = {}

if path.exists(overrides_path):
  overrides_file = open(overrides_path, 'r')
  
  for line in overrides_file:
    line = line.strip()
    
    if len(line) == 0:
      continue
    
    model, ps_short_name, frame, repeat, filename, coloring = line.split()
    overrides[(model, ps_short_name, frame, repeat, filename)] = coloring
  
  overrides_file.close()

for model in models:
  for problem_set in problem_sets:
    ps_name = problem_set['name']
    ps_short_name = problem_set['short_name']
    num_vertices = problem_set['num_vertices']
    num_colors = problem_set['num_colors']
    
    ps_data_dir = path.join(data_dir, ps_short_name)
    mps_response_dir = path.join(response_dir, model, ps_short_name)
    
    if not path.exists(ps_data_dir) or not path.exists(mps_response_dir):
      continue
    
    eval_file_paths = []
    mps_frame_indices = []
    repeat_dirs = []
    
    for i in range(len(frames)):
      f_response_dir = path.join(mps_response_dir, frames[i])
      eval_file_paths.append(path.join(evaluation_dir,
        '%s_%s_%s.txt' % (model, ps_short_name, frames[i])))
      
      if not path.exists(f_response_dir) or path.exists(eval_file_paths[-1]):
        continue
      
      mps_frame_indices.append(i)
      f_repeat_dirs = []
      
      for filename in listdir(f_response_dir):
        file_path = path.join(f_response_dir, filename)
        if path.isdir(file_path):
          f_repeat_dirs.append(file_path)
      
      if len(f_repeat_dirs) == 0:
        f_repeat_dirs.append(f_response_dir)
      
      repeat_dirs.append(f_repeat_dirs)
    
    if len(mps_frame_indices) == 0:
      continue
    
    print('Evaluating %s %s' % (model, ps_short_name))
    
    filenames = []
    possible = []
    answers = [[[] for j in range(len(repeat_dirs[m]))]
               for m in range(len(mps_frame_indices))]
    
    for filename in listdir(ps_data_dir):
      filenames.append(filename)
      
      data_file_path = path.join(ps_data_dir, filename)
      data_file = open(data_file_path, 'r')
      
      lines = data_file.read().splitlines()
      
      data_file.close()
      
      edges = [tuple(int(vertex) for vertex in edge.split(','))
               for edge in lines[0].split('|')]
      connected_vertices = set(vertex for edge in edges for vertex in edge)
      
      problem_is_possible = (lines[1] == 'True')
      possible.append(problem_is_possible)
      
      for m in range(len(mps_frame_indices)):
        i = mps_frame_indices[m]
        for j in range(len(repeat_dirs[m])):
          file_path = path.join(repeat_dirs[m][j], filename)
          
          if not path.exists(file_path):
            answers[m][j].append(None)
            continue
          
          repeat = path.basename(repeat_dirs[m][j])
          overrides_key = (model, ps_short_name, frames[i], repeat, filename)
          coloring = None
          
          if overrides_key in overrides:
            coloring = overrides[overrides_key]
            if coloring == 'not_found':
              del overrides[overrides_key]
              coloring = None
          
          if coloring is None:
            coloring = frame_coloring_funcs[i](
              file_path, num_vertices, connected_vertices, num_colors)
          
          if isinstance(coloring, str) and coloring.startswith('not_found'):
            overrides[overrides_key] = coloring
            if coloring == 'not_found--refuse':
              evaluation = 'refuse'
            else:
              evaluation = 'incorrect'
          elif coloring == 'impossible':
            if problem_is_possible:
              evaluation = 'incorrect'
            else:
              evaluation = 'correct'
          else: # coloring is an actual attempt at a valid coloring
            if isinstance(coloring, str):
              coloring = string_to_coloring(coloring)
            
            if problem_is_possible\
            and coloring_is_valid(num_vertices, edges, coloring):
              evaluation = 'correct'
            else:
              evaluation = 'incorrect'
            
            coloring = coloring_to_string(coloring)
          
          answers[m][j].append((coloring, evaluation))
    
    '''
    for i in [frames.index('cities'), frames.index('friends')]:
      contains_graph = 0
      total_outputs = 0
      
      for j in range(len(repeat_dirs[i])):
        for k in range(len(filenames)):
          file = open(path.join(repeat_dirs[i][j], filenames[k]),
                      'r', encoding='utf-8')
          
          for line in file.readlines():
            if 'graph' in line:
              contains_graph += 1
              break
          
          total_outputs += 1
          
          file.close()
      
      print('%s %s %s: %d of %d outputs contain "graph"'\
            % (model, ps_short_name, frames[i], contains_graph, total_outputs))
    '''
    
    for m in range(len(mps_frame_indices)):
      i = mps_frame_indices[m]
      eval_file = open(eval_file_paths[i], 'w')
      for k in range(len(filenames)):
        for j in range(len(repeat_dirs[m])):
          answer = answers[m][j][k]
          
          if answer is None:
            continue
          
          print('%s %s %s %s %s'\
                % (filenames[k], path.basename(repeat_dirs[m][j]),
                   possible[k], answer[0], answer[1]),
                file=eval_file)
      
      eval_file.close()

overrides_file = open(overrides_path, 'w')

for key, coloring in sorted(overrides.items()):
  model, ps_short_name, frame, repeat, filename = key
  print('%s %s %s %s %s %s' % (model, ps_short_name, frame, repeat, filename,
                               coloring), file=overrides_file)

overrides_file.close()

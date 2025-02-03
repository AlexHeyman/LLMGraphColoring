import math
import itertools
import random

def sample_permutations_enum(elements, perm_size, sample_size):
  perms = list(itertools.permutations(elements, perm_size))
  return random.sample(perms, sample_size)

def sample_permutations_iter(elements, perm_size, sample_size):
  perms = set()
  
  while len(perms) < sample_size:
    perms.add(tuple(random.sample(elements, perm_size)))
  
  return list(perms)

def sample_permutations(elements, perm_size, sample_size, threshold=0.5):
  if sample_size > math.perm(len(elements), perm_size) * threshold:
    return sample_permutations_enum(elements, perm_size, sample_size)
  else:
    return sample_permutations_iter(elements, perm_size, sample_size)

def sample_combinations_enum(elements, comb_size, sample_size):
  combs = list(itertools.combinations(elements, comb_size))
  return random.sample(combs, sample_size)

def sample_combinations_iter(elements, comb_size, sample_size):
  index_population = range(len(elements))
  index_combs = set()
  
  while len(index_combs) < sample_size:
    index_combs.add(tuple(sorted(random.sample(index_population, comb_size))))
  
  return [tuple(elements[i] for i in index_comb) for index_comb in index_combs]

def sample_combinations(elements, comb_size, sample_size, threshold=0.5):
  if sample_size > math.comb(len(elements), comb_size) * threshold:
    return sample_combinations_enum(elements, comb_size, sample_size)
  else:
    return sample_combinations_iter(elements, comb_size, sample_size)

def compute_vertex_conns(num_vertices, edges):
  vertex_conns = [[] for _ in range(num_vertices)]
  
  for edge in edges:
    vertex_conns[edge[0]].append(edge[1])
    vertex_conns[edge[1]].append(edge[0])
    
  return vertex_conns

def coloring_to_string(coloring):
  return '|'.join(str(coloring[i]) for i in range(len(coloring)))

def string_to_coloring(coloring_str):
  return coloring_str.split('|')

def coloring_is_valid(num_vertices, edges, coloring):
  for edge in edges:
    if coloring[edge[0]] == coloring[edge[1]]:
      return False
  return True

def color_brute_force(num_vertices, edges, num_colors):
  # Iterate through all possible colorings of all vertices except the first one
  # (which we assign color 0 WLOG)
  for coloring in itertools.product(list(range(num_colors)),
                                    repeat=num_vertices-1):
    coloring = (0,) + coloring
    for edge in edges:
      if coloring[edge[0]] == coloring[edge[1]]:
        break
    else:
      return coloring
  
  return None

def color_greedy(num_vertices, vertex_conns, num_colors):
  # Vertex coloring order
  v = sorted(range(num_vertices), key=lambda i: -len(vertex_conns[i]))
  
  if len(vertex_conns[v[0]]) == 0:
    # If even the highest-degree vertex has 0 connections, the graph must have
    # 0 edges, so we can trivially give every vertex the same color (color 0)
    return [0 for _ in range(num_vertices)]
  elif num_colors == 1:
    # We now know the graph has >=1 edge, so this is trivially impossible
    return None
  
  # We now know there are >=2 colors and (since there's >=1 edge) >=2 vertices

  # Generate an inverse coloring order that maps vertex index -> order index
  vr = [None for _ in range(num_vertices)]
  for i in range(len(v)):
    vr[v[i]] = i
  
  # WLOG, we can fix the colors of the first vertex in the coloring order and
  # one other vertex connected to it
  second_fixed_vertex = vertex_conns[v[0]][0]
  
  # Swap the second fixed vertex to index 1 in the coloring order
  old_v1 = v[1]
  old_vrs = vr[second_fixed_vertex]
  v[1] = second_fixed_vertex
  v[old_vrs] = old_v1
  vr[old_v1] = old_vrs
  vr[second_fixed_vertex] = 1
  
  vertex_possibilities = [[0], [1]]
  vp_indices = [0, 0]
  
  i = 2
  while i < len(v):
    if len(vertex_possibilities) == i:
      possibility_values = [True for c in range(num_colors)]
      
      for j in range(len(vertex_conns[v[i]])):
        jo = vr[vertex_conns[v[i]][j]]
        if jo < i: # Only check already-colored vertices
          possibility_values[vertex_possibilities[jo][vp_indices[jo]]] = False
      
      vertex_possibilities.append([c for c in range(num_colors)\
                                   if possibility_values[c]])
      vp_indices.append(0)
    else: # len(vertex_possibilities) == i + 1
      vp_indices[i] += 1

    if vp_indices[i] == len(vertex_possibilities[i]):
      # Hit a dead end; try something different with the previous vertices
      i -= 1
      vertex_possibilities.pop()
      vp_indices.pop()
      if i < 2:
        return None
    else:
      # Coloring works so far; continue on to the uncolored vertices
      i += 1

  coloring = [-1 for _ in range(num_vertices)]
  
  for i in range(len(v)):
    coloring[v[i]] = vertex_possibilities[i][vp_indices[i]]
  
  return coloring

def compute_greedy_score_method1(num_vertices, vertex_conns, num_colors):
  if num_vertices <= 6:
    trials = itertools.permutations(range(num_vertices))
  else:
    trials = sample_permutations_iter(range(num_vertices), num_vertices, 1000)
  
  trial_count = 0
  success_count = 0
  
  for v in trials:
    trial_count += 1
    
    # Try to color the vertices in the order v using a naive greedy algorithm
    # that gives up if it reaches a dead end
    
    # Generate an inverse coloring order that maps vertex index -> order index
    vr = [None for _ in range(num_vertices)]
    for i in range(len(v)):
      vr[v[i]] = i
    
    coloring = [None for _ in range(num_vertices)]
    
    # Color the first vertex in the order with the lowest-index color, 0
    coloring[v[0]] = 0
    
    for i in range(1, len(v)):
      possibility_values = [True for c in range(num_colors)]
      
      for nv in vertex_conns[v[i]]:
        if vr[nv] < i: # Only check already-colored vertices
          possibility_values[coloring[nv]] = False
      
      for c in range(num_colors):
        if possibility_values[c]:
          # Found a valid color for this vertex
          coloring[v[i]] = c
          break
      else:
        break # No valid colors for this vertex; coloring attempt failed
    else:
      success_count += 1 # Coloring attempt succeeded
  
  return success_count / trial_count

def compute_greedy_score_method2(num_vertices, vertex_conns, num_colors):
  trial_count = 10000
  success_count = 0
  
  for _ in range(trial_count):
    coloring = [None for _ in range(num_vertices)]
    
    current_vertex = random.randrange(num_vertices)
    
    while True:
      possibility_values = [True for c in range(num_colors)]
      neighboring_uncolored = []
      
      for nv in vertex_conns[current_vertex]:
        if coloring[nv] is None:
          neighboring_uncolored.append(nv)
        else:
          possibility_values[coloring[nv]] = False
      
      for c in range(num_colors):
        if possibility_values[c]:
          # Found a valid color for this vertex
          coloring[current_vertex] = c
          break
      else:
        break # No valid colors for this vertex; coloring attempt failed
      
      if len(neighboring_uncolored) == 0:
        uncolored = [v for v in range(num_vertices) if coloring[v] is None]
        if len(uncolored) == 0:
          # We've successfully colored all the vertices
          success_count += 1
          break
        else:
          current_vertex = random.choice(uncolored)
      else:
        current_vertex = random.choice(neighboring_uncolored)
  
  return success_count / trial_count

def compute_greedy_score(num_vertices, vertex_conns, num_colors):
  return compute_greedy_score_method2(num_vertices, vertex_conns, num_colors)

def find_complete_subgraph(num_vertices, edges_set, vertex_conns,
                           subgraph_size):
  if subgraph_size > num_vertices:
    return None
  elif subgraph_size == 1:
    return [0]
  
  # Vertices with sufficient degree to be part of a complete subgraph of the
  # required size
  sd_vertices = [i for i in range(num_vertices)\
                 if len(vertex_conns[i]) >= subgraph_size - 1]
  
  # Vertex testing order (first candidate will be v[:subgraph_size], etc.)
  v = sorted(sd_vertices, key=lambda i: -len(vertex_conns[i]))
  
  for candidate in itertools.combinations(v, subgraph_size):
    is_complete = True
    
    for pair in itertools.combinations(candidate, 2):
      if pair[0] < pair[1]:
        if (pair[0], pair[1]) not in edges_set:
          is_complete = False
          break
      else:
        if (pair[1], pair[0]) not in edges_set:
          is_complete = False
          break
    
    if is_complete:
      return candidate
  
  return None

if __name__ == '__main__':
  num_vertices = 5
  num_colors = 3
  
  possible_edges = list(itertools.combinations(range(num_vertices), 2))
  
  for num_edges in range(len(possible_edges) + 1):
    for edges in itertools.combinations(possible_edges, num_edges):
      vertex_conns = compute_vertex_conns(num_vertices, edges)
      
      coloring_bf = color_brute_force(num_vertices, edges, num_colors)
      coloring_g = color_greedy(num_vertices, vertex_conns, num_colors)
      gs = compute_greedy_score(num_vertices, vertex_conns, num_colors)
      sg = find_complete_subgraph(num_vertices, set(edges), vertex_conns,
                                  num_colors + 1)
      
      if coloring_bf is None:
        assert(coloring_g is None)
        assert(gs == 0)
      else:
        assert(coloring_g is not None)
        assert(coloring_is_valid(num_vertices, edges, coloring_g))
        assert(sg is None)

      sg_str = 'none'
      if sg is not None:
        sg_str = ','.join(str(v) for v in sg)
      
      print('|'.join(','.join(str(v) for v in edge) for edge in edges),
            (coloring_bf is not None), ('%.4f' % gs), sg_str)

from os import path

data_dir = 'data'
prompt_dir = 'prompts'
response_dir = 'responses'
overrides_path = path.join(response_dir, 'overrides.txt')
evaluation_dir = 'evaluations'
summary_dir = 'summaries'

models = ['llama3.1_405B',
          'gpt4o',
          'o1-mini',
          'deepseek-r1',
          'claude3.5s',
          'gemini1.5p'
          ]

# random_method is either 'efficient' (default) or 'legacy'
problem_sets = [{'name': '4 vertices, 2 colors',
                 'short_name': '4v2c',
                 'num_vertices': 4,
                 'num_colors': 2
                 },
                {'name': '5 vertices, 2 colors',
                 'short_name': '5v2c',
                 'num_vertices': 5,
                 'num_colors': 2
                 },
                {'name': '6 vertices, 3 colors',
                 'short_name': '6v3c',
                 'num_vertices': 6,
                 'num_colors': 3,
                 'max_samples_per_ec': 50,
                 'random_method': 'legacy',
                 'random_seed': 100
                 },
                {'name': '7 vertices, 3 colors',
                 'short_name': '7v3c',
                 'num_vertices': 7,
                 'num_colors': 3,
                 'max_samples_per_ec': 50,
                 'random_method': 'legacy',
                 'random_seed': 101
                 },
                {'name': '8 vertices, 4 colors',
                 'short_name': '8v4c',
                 'num_vertices': 8,
                 'num_colors': 4,
                 'max_samples_per_ec': 50,
                 'random_method': 'legacy',
                 'random_seed': 102
                 }
                ]

frames = ['math', 'math_demanding', 'cities', 'friends']
frame_indices = {v: i for i, v in enumerate(frames)}

math_colors = ['Red', 'Green', 'Blue', 'Yellow']
math_colors_lower = [s.lower() for s in math_colors]

cities_colors = ['President', 'VP', 'Secretary', 'Speaker']
cities_colors_lower = [s.lower() for s in cities_colors]
cities_colors_full = ['the first President',
                      'the first Vice President',
                      'the first Secretary of State',
                      'the first Speaker of the House']

friends_colors = ['Red', 'Green', 'Blue', 'Yellow']
friends_colors_lower = [s.lower() for s in friends_colors]
friends_names = ['Alice', 'Bob', 'Carol', 'Dave', 'Ethan', 'Fran', 'George',
                 'Heather']
friends_names_lower = [s.lower() for s in friends_names]

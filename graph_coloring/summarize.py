'''
When run, this file reads through the files in the evaluation directory (see
metadata.py) and generates plots summarizing the results, putting them in the
summary directory.

This file does not run with any arguments.
'''

from os import path, listdir, mkdir
from math import ceil
from scipy.stats import binomtest
import matplotlib.pyplot as plt

from metadata import *

complete_models = ['llama3.1_405B', 'gpt4o', 'claude3.5s', 'gemini1.5p']
complete_model_indices = [models.index(name) for name in complete_models]
model_proper_names = ['Llama 3.1 405B', 'GPT-4o', 'o1-mini', 'DeepSeek-R1',
                      'Claude 3.5 Sonnet', 'Gemini 1.5 Pro']
frame_proper_names = ['Math', 'Math (demanding)', 'Cities', 'Friends']
frame_colors = ['b', 'r', 'm', 'g']
frame_line_styles = ['dashed', 'dashdot', 'dotted', 'solid']
num_types = 5
num_poss_types = 3
type_names = ['g ≥ 0.9', '0.5 ≤ g < 0.9', 'g < 0.5',
              'D-uncolorable', 'C-uncolorable']
type_colors = [('royalblue', 'deepskyblue'),
               ('turquoise', 'aquamarine'),
               ('limegreen', 'palegreen'),
               ('gold', (255/255, 236/255, 128/255)),
               ('orangered', 'lightsalmon')]
min_responses = 20
graph_width = 2 * (6.75 / 2)
graph_height = 2 * ((9 - 1) / 3)
border_size = 0.025

def get_stats(positive, total):
  acc = positive / total
  ci = binomtest(positive, total).proportion_ci(confidence_level=0.95)
  return acc, (acc - ci.low), (ci.high - acc)

plt.rcdefaults()
plt.rcParams['font.family'] = 'Helvetica'

ec_accuracy_dir = path.join(summary_dir, 'ec_accuracy')
if not path.exists(ec_accuracy_dir):
  mkdir(ec_accuracy_dir)

type_error_dir = path.join(summary_dir, 'type_error')
if not path.exists(type_error_dir):
  mkdir(type_error_dir)

all_filename_indices = [None for i in problem_sets]
all_edge_counts = [None for i in problem_sets]
all_problem_types = [None for i in problem_sets]
all_evals = [[None for j in complete_model_indices] for i in problem_sets]
all_type_responses = [[None for j in complete_model_indices]
                      for i in problem_sets]
all_type_incorrect = [[None for j in complete_model_indices]
                      for i in problem_sets]

for ps_index in range(len(problem_sets)):
  problem_set = problem_sets[ps_index]
  ps_short_name = problem_set['short_name']
  ps_data_dir = path.join(data_dir, ps_short_name)
  
  print('Getting data for %s' % ps_short_name)
  
  filename_indices = {}
  edge_counts = []
  ec_set = set()
  problem_types = []
  
  all_filename_indices[ps_index] = filename_indices
  all_edge_counts[ps_index] = edge_counts
  all_problem_types[ps_index] = problem_types
  
  problem_index = 0
  for filename in listdir(ps_data_dir):
    filename_indices[filename] = problem_index
    
    data_file_path = path.join(ps_data_dir, filename)
    data_file = open(data_file_path, 'r')
    
    lines = data_file.read().splitlines()
    
    data_file.close()
    
    edges = [tuple(int(num) for num in edge.split(','))
             for edge in lines[0].split('|')]
    num_edges = len(edges)
    edge_counts.append(num_edges)
    ec_set.add(num_edges)
    
    problem_type = None
    
    is_possible = (lines[1] == 'True')
    
    if is_possible:
      greedy_score = float(lines[2])
      if greedy_score >= 0.9:
        problem_type = 0
      elif greedy_score >= 0.5:
        problem_type = 1
      elif greedy_score >= 0:
        problem_type = 2
    else:
      complete_subgraph_exists = (lines[2] == 'True')
      if complete_subgraph_exists:
        problem_type = 4
      else:
        problem_type = 3
    
    problem_types.append(problem_type)
    
    problem_index += 1
  
  ec_all = {ec: 0 for ec in ec_set}
  ec_g00 = {ec: 0 for ec in ec_set}
  ec_g05 = {ec: 0 for ec in ec_set}
  ec_g09 = {ec: 0 for ec in ec_set}
  
  for problem_index in range(len(edge_counts)):
    num_edges = edge_counts[problem_index]
    problem_type = problem_types[problem_index]
    ec_all[num_edges] += 1
    if problem_type <= 2:
      ec_g00[num_edges] += 1
      if problem_type <= 1:
        ec_g05[num_edges] += 1
        if problem_type == 0:
          ec_g09[num_edges] += 1
  
  for cm in range(len(complete_model_indices)):
    model_index = complete_model_indices[cm]
    model = models[model_index]
    model_pn = model_proper_names[model_index]
    
    print('Summarizing %s %s' % (model, ps_short_name))
    
    evals = [[[] for fi in filename_indices] for i in range(len(frames))]
    ec_responses = [{ec: 0 for ec in ec_set} for i in range(len(frames))]
    ec_correct = [{ec: 0 for ec in ec_set} for i in range(len(frames))]
    type_responses = [[0 for j in range(num_types)] for i in range(len(frames))]
    type_incorrect = [[0 for j in range(num_types)] for i in range(len(frames))]
    type_fp = [[0 for j in range(num_poss_types)] for i in range(len(frames))]
    type_not_found = [[0 for j in range(num_types)] for i in range(len(frames))]
    
    all_evals[ps_index][cm] = evals
    all_type_responses[ps_index][cm] = type_responses
    all_type_incorrect[ps_index][cm] = type_incorrect
    
    for i in range(len(frames)):
      eval_file_path = path.join(evaluation_dir,
        '%s_%s_%s.txt' % (model, ps_short_name, frames[i]))
      eval_file = open(eval_file_path, 'r')
      
      for line in eval_file:
        line = line.strip()
        
        if len(line) == 0:
          continue
        
        filename, repeat, is_possible, coloring, evaluation = line.split()
        
        problem_index = filename_indices[filename]
        num_edges = edge_counts[problem_index]
        problem_type = problem_types[problem_index]
        
        evals[i][problem_index].append((coloring, evaluation))
        ec_responses[i][num_edges] += 1
        type_responses[i][problem_type] += 1
        
        if evaluation == 'correct':
          ec_correct[i][num_edges] += 1
        elif evaluation == 'incorrect':
          type_incorrect[i][problem_type] += 1
          if coloring.startswith('not_found'):
            type_not_found[i][problem_type] += 1
          elif is_possible == 'True' and coloring != 'impossible':
            type_fp[i][problem_type] += 1
      
      eval_file.close()
    
    # Make graph of accuracy vs. edge count for each frame
    
    ec_list = sorted(ec for ec in ec_set
                     if ec_responses[0][ec] >= min_responses)
    ec_ticks = list(ec_list)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(graph_width, graph_height)
    
    legend_order = []
    
    for i in range(len(frames)):
      accuracies = []
      lower_errs = []
      upper_errs = []
      
      for j in ec_list:
        acc, lower, upper = get_stats(ec_correct[i][j], ec_responses[i][j])
        accuracies.append(acc)
        lower_errs.append(lower)
        upper_errs.append(upper)

      label = frame_proper_names[i]
      legend_order.append(label)
      ax.errorbar(ec_list, accuracies, [lower_errs, upper_errs],
                  capsize=3, marker='.', color=frame_colors[i],
                  label=label, zorder=i)
    
    ec_list_zeros = [0 for _ in ec_list]
    
    if max((ec_g00[i] - ec_g05[i]) for i in ec_list) > 0:
      label = type_names[2]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g00[i] / ec_all[i] for i in ec_list],
                      color=str(7/8), label=label, zorder=-3)
    
    if max((ec_g05[i] - ec_g09[i]) for i in ec_list) > 0:
      label = type_names[1]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g05[i] / ec_all[i] for i in ec_list],
                      color=str(6/8), label=label, zorder=-2)
    
    if max(ec_g09[i] for i in ec_list) > 0:
      label = type_names[0]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g09[i] / ec_all[i] for i in ec_list],
                      color=str(5/8), label=label, zorder=-1)
    
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles),
                                  key=lambda t: legend_order.index(t[0])))
    ax.legend(handles, labels, loc='lower left')
    
    width = ec_list[-1] - ec_list[0]
    left = ec_list[0] - border_size * width
    right = ec_list[-1] + border_size * width
    ax.set_xlim(left, right)
    ax.set_ylim(0 - border_size, 1 + border_size)
    ax.set_title('%s (%s)' % (model_pn, ps_short_name))
    ax.set_xlabel('Number of edges')
    ax.set_xticks(ec_ticks)
    ax.set_ylabel('Accuracy')
    
    ax2 = ax.twinx()
    ax2.set_ylim(0 - border_size, 1 + border_size)
    ax2.set_ylabel('Fraction of problems')

    fig.tight_layout()
    save_path = path.join(ec_accuracy_dir, '%s_%s' % (model, ps_short_name))
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.pdf')
    plt.close(fig)
    
    # Make graph of error vs. frame and problem type
    
    types_to_include = [j for j in range(num_types)
                        if type_responses[0][j] >= min_responses]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(graph_width, graph_height)
    
    bar_x = [[] for k in range(len(types_to_include))]
    bar_h1 = [[] for k in range(len(types_to_include))]
    bar_h2 = [[] for k in range(len(types_to_include))]
    bar_h3 = [[] for k in range(len(types_to_include))]
    lower_errs = [[] for k in range(len(types_to_include))]
    upper_errs = [[] for k in range(len(types_to_include))]
    xticks = []
    
    x = 0
    
    for i in range(len(frames)):
      for k in range(len(types_to_include)):
        j = types_to_include[k]
        bar_x[k].append(x)
        err, lower, upper = get_stats(type_incorrect[i][j],
                                      type_responses[i][j])
        if j <= 2:
          fp_frac = type_fp[i][j] / type_responses[i][j]
        else:
          fp_frac = 0
        not_found_frac = type_not_found[i][j] / type_responses[i][j]
        bar_h1[k].append(not_found_frac)
        bar_h2[k].append(err - not_found_frac - fp_frac)
        bar_h3[k].append(fp_frac)
        lower_errs[k].append(lower)
        upper_errs[k].append(upper)
        x += 1
      
      xticks.append(sum(bar_x[k][i] for k in range(len(types_to_include)))\
                    / len(types_to_include))
      x += 1

    bar_h1h2 = [[bar_h1[k][i] + bar_h2[k][i] for i in range(len(frames))]
                for k in range(len(types_to_include))]
    bar_h1h2h3 = [[bar_h1h2[k][i] + bar_h3[k][i] for i in range(len(frames))]
                  for k in range(len(types_to_include))]
    
    for k in range(len(types_to_include)):
      j = types_to_include[k]
      ax.bar(bar_x[k], bar_h1[k], width=0.8, align='center', color='k',
             edgecolor='k', linewidth=1)
      ax.bar(bar_x[k], bar_h2[k], bottom=bar_h1[k], width=0.8, align='center',
             color=type_colors[j][0], edgecolor='k', linewidth=1,
             label=type_names[j])
      if j <= 2:
        ax.bar(bar_x[k], bar_h3[k], bottom=bar_h1h2[k], width=0.8,
               align='center', color=type_colors[j][1], edgecolor='k',
               linewidth=1)
      ax.errorbar(bar_x[k], bar_h1h2h3[k], [lower_errs[k], upper_errs[k]],
                  capsize=3, color='k', linestyle='none')
    
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.set_title('%s (%s)' % (model_pn, ps_short_name))
    ax.set_xlabel('Frame')
    ax.set_xticks(ticks=xticks, labels=frame_proper_names)
    ax.set_ylabel('Error rate')
    
    fig.tight_layout()
    save_path = path.join(type_error_dir, '%s_%s' % (model, ps_short_name))
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.pdf')
    plt.close(fig)

# Make graphs of error vs. problem set, problem type, and frame (one per model)

columns = 2
rows = ceil(len(complete_model_indices) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(graph_width * columns, graph_height * rows + 0.6)

col = 0
row = 0
legend = {}

for cm in range(len(complete_model_indices)):
  model_index = complete_model_indices[cm]
  model_pn = model_proper_names[model_index]
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  bar_x = [[] for j in range(num_types)]
  frame_h = [[] for j in range(num_types)]
  bar_h1 = [[] for j in range(num_types)]
  bar_h2 = [[] for j in range(num_types)]
  xticks = []
  xticklabels = []
  
  x = 0
  
  for ps_index in range(len(problem_sets)):
    type_responses = all_type_responses[ps_index][cm]
    type_incorrect = all_type_incorrect[ps_index][cm]
    
    types_to_include = [j for j in range(num_types)
                        if type_responses[0][j] >= min_responses]
    
    for k in range(len(types_to_include)):
      j = types_to_include[k]
      bar_x[j].append(x)
      frame_h[j].append([(type_incorrect[i][j] / type_responses[i][j])\
                         for i in range(len(frames))])
      frame_min = min(frame_h[j][-1])
      frame_max = max(frame_h[j][-1])
      bar_h1[j].append(frame_min)
      bar_h2[j].append(frame_max - frame_min)
      x += 1
    
    xticks.append(sum(bar_x[types_to_include[k]][-1] for k in\
                      range(len(types_to_include))) / len(types_to_include))
    xticklabels.append(problem_sets[ps_index]['short_name'])
    x += 1
  
  for j in range(num_types):
    if len(bar_x[j]) == 0:
      continue
    
    ax.bar(bar_x[j], bar_h1[j], width=0.8, align='center',
           color=type_colors[j][0], label=type_names[j])
    ax.bar(bar_x[j], bar_h2[j], bottom=bar_h1[j], width=0.8, align='center',
           color=type_colors[j][1])
    
    for m in range(len(bar_x[j])):
      x1 = bar_x[j][m] - 0.8/2
      x2 = bar_x[j][m] + 0.8/2
      for i in range(len(frames)):
        y = frame_h[j][m][i]
        ax.plot([x1, x2], [y, y], color='k', linestyle=frame_line_styles[i],
                label=frame_proper_names[i])
  
  ax.set_ylim(0, 1)
  ax.set_title(model_pn)
  ax.set_xlabel('Problem set')
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  ax.set_ylabel('Error rate')
  
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  col += 1
  if col >= columns:
    row += 1
    col = 0

fig.legend(handles=legend.values(), labels=legend.keys(),
           loc='upper center', ncols=ceil((num_types + len(frames)) / 2))
fig.tight_layout()
fig.subplots_adjust(top=((graph_height * rows - 0.25)\
                         / (graph_height * rows + 0.6)))
save_path = path.join(type_error_dir, 'all_complete_models')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

# Make graphs featuring the models with responses only for selected problems

sel_models = ['o1-mini', 'deepseek-r1']
sel_model_indices = [models.index(name) for name in sel_models]
sel_ps_indices = [0, 1, 2, 3, 4] # 4v2c, 5v2c, 6v3c, 7v3c, 8v4c
sel_ps_names = ['4v2c', '5v2c (3-6e)', '6v3c (7-12e)', '7v3c (7-15e)',
                '8v4c (14-23e)']
sel_ps_names_inparen = ['4v2c', '5v2c, 3-6e', '6v3c, 7-12e', '7v3c, 7-15e',
                        '8v4c, 14-23e']
sel_ps_names_file = ['4v2c', '5v2c_3-6e', '6v3c_7-12e', '7v3c_7-15e',
                     '8v4c_14-23e']
sel_frames = ['math', 'friends']
sel_frame_indices = [frames.index(name) for name in sel_frames]
graph_model_indices = complete_model_indices + sel_model_indices

sel_type_responses = [[None for i in sel_ps_indices] for m in sel_model_indices]
sel_type_incorrect = [[None for i in sel_ps_indices] for m in sel_model_indices]

for sps_index in range(len(sel_ps_indices)):
  ps_index = sel_ps_indices[sps_index]
  problem_set = problem_sets[ps_index]
  ps_short_name = problem_set['short_name']
  filename_indices = all_filename_indices[ps_index]
  edge_counts = all_edge_counts[ps_index]
  problem_types = all_problem_types[ps_index]
  
  print('Summarizing selected-problem models on %s' % sel_ps_names[sps_index])
  
  problem_indices = set()

  # Arbitrary
  model = sel_models[0]
  frame = sel_frames[0]
  
  eval_file_path = path.join(evaluation_dir,
                             '%s_%s_%s.txt' % (model, ps_short_name, frame))
  eval_file = open(eval_file_path, 'r')
  
  for line in eval_file:
    line = line.strip()
    
    if len(line) == 0:
      continue
    
    filename, repeat, is_possible, coloring, evaluation = line.split()
    
    problem_index = filename_indices[filename]
    problem_indices.add(problem_index)
  
  eval_file.close()
  
  ec_set = set(edge_counts[pi] for pi in problem_indices)
  
  ec_all = {ec: 0 for ec in ec_set}
  ec_g00 = {ec: 0 for ec in ec_set}
  ec_g05 = {ec: 0 for ec in ec_set}
  ec_g09 = {ec: 0 for ec in ec_set}
  
  for problem_index in problem_indices:
    num_edges = edge_counts[problem_index]
    problem_type = problem_types[problem_index]
    ec_all[num_edges] += 1
    if problem_type <= 2:
      ec_g00[num_edges] += 1
      if problem_type <= 1:
        ec_g05[num_edges] += 1
        if problem_type == 0:
          ec_g09[num_edges] += 1
  
  models_type_responses = [[[0 for j in range(num_types)]
                            for i in sel_frames]
                           for m in graph_model_indices]
  models_type_incorrect = [[[0 for j in range(num_types)]
                            for i in sel_frames]
                           for m in graph_model_indices]
  
  for i in range(len(sel_frames)):
    si = sel_frame_indices[i]
    
    for problem_index in problem_indices:
      num_edges = edge_counts[problem_index]
      problem_type = problem_types[problem_index]
      
      for m in range(len(complete_model_indices)):
        for coloring, evaluation in all_evals[ps_index][m][si][problem_index]:
          models_type_responses[m][i][problem_type] += 1
          if evaluation == 'incorrect':
            models_type_incorrect[m][i][problem_type] += 1
  
  for sm in range(len(sel_model_indices)):
    model = sel_models[sm]
    m = len(complete_model_indices) + sm
    model_pn = model_proper_names[graph_model_indices[m]]
    
    ec_responses = [{ec: 0 for ec in ec_set} for i in sel_frames]
    ec_correct = [{ec: 0 for ec in ec_set} for i in sel_frames]
    type_fp = [[0 for j in range(num_poss_types)] for i in sel_frames]
    type_not_found = [[0 for j in range(num_types)] for i in sel_frames]
    
    sel_type_responses[sm][sps_index] = models_type_responses[m]
    sel_type_incorrect[sm][sps_index] = models_type_incorrect[m]
    type_responses = models_type_responses[m]
    type_incorrect = models_type_incorrect[m]
    
    for i in range(len(sel_frames)):
      eval_file_path = path.join(evaluation_dir,
        '%s_%s_%s.txt' % (model, ps_short_name, sel_frames[i]))
      eval_file = open(eval_file_path, 'r')
      
      for line in eval_file:
        line = line.strip()
        
        if len(line) == 0:
          continue
        
        filename, repeat, is_possible, coloring, evaluation = line.split()
        
        problem_index = filename_indices[filename]
        num_edges = edge_counts[problem_index]
        problem_type = problem_types[problem_index]
        
        ec_responses[i][num_edges] += 1
        type_responses[i][problem_type] += 1
        
        if evaluation == 'correct':
          ec_correct[i][num_edges] += 1
        elif evaluation == 'incorrect':
          type_incorrect[i][problem_type] += 1
          if coloring.startswith('not_found'):
            type_not_found[i][problem_type] += 1
          elif is_possible == 'True' and coloring != 'impossible':
            type_fp[i][problem_type] += 1
      
      eval_file.close()
    
    # Make graph of accuracy vs. edge count for each frame
    
    ec_list = sorted(ec for ec in ec_set
                     if ec_responses[0][ec] >= min_responses)
    ec_ticks = list(ec_list)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(graph_width, graph_height)
    
    legend_order = []
    
    for i in range(len(sel_frames)):
      si = sel_frame_indices[i]
      
      accuracies = []
      lower_errs = []
      upper_errs = []
      
      for j in ec_list:
        acc, lower, upper = get_stats(ec_correct[i][j], ec_responses[i][j])
        accuracies.append(acc)
        lower_errs.append(lower)
        upper_errs.append(upper)
      
      label = frame_proper_names[si]
      legend_order.append(label)
      ax.errorbar(ec_list, accuracies, [lower_errs, upper_errs],
                  capsize=3, marker='.', color=frame_colors[si],
                  label=label, zorder=i)
    
    ec_list_zeros = [0 for _ in ec_list]
    
    if max((ec_g00[i] - ec_g05[i]) for i in ec_list) > 0:
      label = type_names[2]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g00[i] / ec_all[i] for i in ec_list],
                      color=str(7/8), label=label, zorder=-3)
    
    if max((ec_g05[i] - ec_g09[i]) for i in ec_list) > 0:
      label = type_names[1]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g05[i] / ec_all[i] for i in ec_list],
                      color=str(6/8), label=label, zorder=-2)
    
    if max(ec_g09[i] for i in ec_list) > 0:
      label = type_names[0]
      legend_order.append(label)
      ax.fill_between(ec_list, ec_list_zeros,
                      [ec_g09[i] / ec_all[i] for i in ec_list],
                      color=str(5/8), label=label, zorder=-1)
    
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles),
                                  key=lambda t: legend_order.index(t[0])))
    ax.legend(handles, labels, loc='lower left')
    
    width = ec_list[-1] - ec_list[0]
    left = ec_list[0] - border_size * width
    right = ec_list[-1] + border_size * width
    ax.set_xlim(left, right)
    ax.set_ylim(0 - border_size, 1 + border_size)
    ax.set_title('%s (%s)' % (model_pn, sel_ps_names_inparen[sps_index]))
    ax.set_xlabel('Number of edges')
    ax.set_xticks(ec_ticks)
    ax.set_ylabel('Accuracy')
    
    ax2 = ax.twinx()
    ax2.set_ylim(0 - border_size, 1 + border_size)
    ax2.set_ylabel('Fraction of problems')
    
    fig.tight_layout()
    save_path = path.join(ec_accuracy_dir,
                          '%s_%s' % (model, sel_ps_names_file[sps_index]))
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.pdf')
    plt.close(fig)
    
    # Make graph of error vs. frame and problem type
    
    types_to_include = [j for j in range(num_types)
                        if type_responses[0][j] >= min_responses]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(graph_width, graph_height)
    
    bar_x = [[] for k in range(len(types_to_include))]
    bar_h1 = [[] for k in range(len(types_to_include))]
    bar_h2 = [[] for k in range(len(types_to_include))]
    bar_h3 = [[] for k in range(len(types_to_include))]
    lower_errs = [[] for k in range(len(types_to_include))]
    upper_errs = [[] for k in range(len(types_to_include))]
    xticks = []
    xticklabels = []
    
    x = 0
    
    for i in range(len(sel_frames)):
      for k in range(len(types_to_include)):
        j = types_to_include[k]
        bar_x[k].append(x)
        err, lower, upper = get_stats(type_incorrect[i][j],
                                      type_responses[i][j])
        if j <= 2:
          fp_frac = type_fp[i][j] / type_responses[i][j]
        else:
          fp_frac = 0
        not_found_frac = type_not_found[i][j] / type_responses[i][j]
        bar_h1[k].append(not_found_frac)
        bar_h2[k].append(err - not_found_frac - fp_frac)
        bar_h3[k].append(fp_frac)
        lower_errs[k].append(lower)
        upper_errs[k].append(upper)
        x += 1
      
      xticks.append(sum(bar_x[k][i] for k in range(len(types_to_include)))\
                    / len(types_to_include))
      xticklabels.append(frame_proper_names[sel_frame_indices[i]])
      x += 1
    
    bar_h1h2 = [[bar_h1[k][i] + bar_h2[k][i]
                 for i in range(len(sel_frames))]
                for k in range(len(types_to_include))]
    bar_h1h2h3 = [[bar_h1h2[k][i] + bar_h3[k][i]
                   for i in range(len(sel_frames))]
                  for k in range(len(types_to_include))]
    
    for k in range(len(types_to_include)):
      j = types_to_include[k]
      ax.bar(bar_x[k], bar_h1[k], width=0.8, align='center', color='k',
             edgecolor='k', linewidth=1)
      ax.bar(bar_x[k], bar_h2[k], bottom=bar_h1[k], width=0.8, align='center',
             color=type_colors[j][0], edgecolor='k', linewidth=1,
             label=type_names[j])
      if j <= 2:
        ax.bar(bar_x[k], bar_h3[k], bottom=bar_h1h2[k], width=0.8,
               align='center', color=type_colors[j][1], edgecolor='k',
               linewidth=1)
      ax.errorbar(bar_x[k], bar_h1h2h3[k], [lower_errs[k], upper_errs[k]],
                  capsize=3, color='k', linestyle='none')
    
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.3)
    ax.set_title('%s (%s)' % (model_pn, sel_ps_names_inparen[sps_index]))
    ax.set_xlabel('Frame')
    ax.set_xticks(ticks=xticks, labels=xticklabels)
    ax.set_ylabel('Error rate')
    
    fig.tight_layout()
    save_path = path.join(type_error_dir,
                          '%s_%s' % (model, sel_ps_names_file[sps_index]))
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.pdf')
    plt.close(fig)
  
  # Make graph of error vs. model, problem type, and frame

  types_to_include = [j for j in range(num_types)
    if min(models_type_responses[m][0][j]
      for m in range(len(complete_model_indices))) >= min_responses]
  
  fig, ax = plt.subplots()
  fig.set_size_inches(graph_width, graph_height)
  
  bar_x = [[] for j in range(num_types)]
  frame_h = [[] for j in range(num_types)]
  bar_h1 = [[] for j in range(num_types)]
  bar_h2 = [[] for j in range(num_types)]
  xticks = []
  xticklabels = []
  
  x = 0
  
  for m in range(len(graph_model_indices)):
    type_responses = models_type_responses[m]
    type_incorrect = models_type_incorrect[m]
    
    for k in range(len(types_to_include)):
      j = types_to_include[k]
      bar_x[j].append(x)
      frame_h[j].append([(type_incorrect[i][j] / type_responses[i][j])\
                         for i in range(len(sel_frames))])
      frame_min = min(frame_h[j][-1])
      frame_max = max(frame_h[j][-1])
      bar_h1[j].append(frame_min)
      bar_h2[j].append(frame_max - frame_min)
      x += 1
    
    model_pn = model_proper_names[graph_model_indices[m]]
    if model_pn == 'Claude 3.5 Sonnet':
      model_pn = 'Claude 3.5 S.'
    elif model_pn == 'Gemini 1.5 Pro':
      model_pn = 'Gemini 1.5 P.'
    
    xticks.append(sum(bar_x[types_to_include[k]][-1] for k in\
                      range(len(types_to_include))) / len(types_to_include))
    xticklabels.append(model_pn)
    x += 1
  
  for j in range(num_types):
    if len(bar_x[j]) == 0:
      continue
    
    ax.bar(bar_x[j], bar_h1[j], width=0.8, align='center',
           color=type_colors[j][0], label=type_names[j])
    ax.bar(bar_x[j], bar_h2[j], bottom=bar_h1[j], width=0.8, align='center',
           color=type_colors[j][1])
    
    for m in range(len(bar_x[j])):
      x1 = bar_x[j][m] - 0.8/2
      x2 = bar_x[j][m] + 0.8/2
      for i in range(len(sel_frames)):
        si = sel_frame_indices[i]
        y = frame_h[j][m][i]
        ax.plot([x1, x2], [y, y], color='k', linestyle=frame_line_styles[si],
                label=frame_proper_names[si])
  
  legend = {}
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  ax.legend(handles=legend.values(), labels=legend.keys(), loc='upper right')
  ax.set_ylim(0, 1)
  ax.set_title(sel_ps_names[sps_index])
  ax.set_xlabel('Model')
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  ax.set_ylabel('Error rate')
  
  fig.tight_layout()
  save_path = path.join(type_error_dir,
                        'all_models_%s' % sel_ps_names_file[sps_index])
  plt.savefig(save_path + '.png')
  plt.savefig(save_path + '.pdf')
  plt.close(fig)

# Make graphs of error vs. problem set, problem type, and frame (one per model)

columns = 2
rows = ceil(len(sel_model_indices) / columns)
fig, axs = plt.subplots(rows, columns)
fig.set_size_inches(graph_width * columns, graph_height * rows + 0.6)

col = 0
row = 0
legend = {}

for sm in range(len(sel_model_indices)):
  model_index = sel_model_indices[sm]
  model_pn = model_proper_names[model_index]
  if rows == 1:
    ax = axs[col]
  else:
    ax = axs[row][col]
  
  bar_x = [[] for j in range(num_types)]
  frame_h = [[] for j in range(num_types)]
  bar_h1 = [[] for j in range(num_types)]
  bar_h2 = [[] for j in range(num_types)]
  xticks = []
  xticklabels = []
  
  x = 0
  
  for sps_index in range(len(sel_ps_indices)):
    type_responses = sel_type_responses[sm][sps_index]
    type_incorrect = sel_type_incorrect[sm][sps_index]
    
    types_to_include = [j for j in range(num_types)
                        if type_responses[0][j] >= min_responses]
    
    for k in range(len(types_to_include)):
      j = types_to_include[k]
      bar_x[j].append(x)
      frame_h[j].append([(type_incorrect[i][j] / type_responses[i][j])\
                         for i in range(len(sel_frames))])
      frame_min = min(frame_h[j][-1])
      frame_max = max(frame_h[j][-1])
      bar_h1[j].append(frame_min)
      bar_h2[j].append(frame_max - frame_min)
      x += 1
    
    xticks.append(sum(bar_x[types_to_include[k]][-1] for k in\
                      range(len(types_to_include))) / len(types_to_include))
    xticklabels.append(sel_ps_names[sps_index])
    x += 1
  
  for j in range(num_types):
    if len(bar_x[j]) == 0:
      continue
    
    ax.bar(bar_x[j], bar_h1[j], width=0.8, align='center',
           color=type_colors[j][0], label=type_names[j])
    ax.bar(bar_x[j], bar_h2[j], bottom=bar_h1[j], width=0.8, align='center',
           color=type_colors[j][1])
    
    for m in range(len(bar_x[j])):
      x1 = bar_x[j][m] - 0.8/2
      x2 = bar_x[j][m] + 0.8/2
      for i in range(len(sel_frames)):
        si = sel_frame_indices[i]
        y = frame_h[j][m][i]
        ax.plot([x1, x2], [y, y], color='k', linestyle=frame_line_styles[si],
                label=frame_proper_names[si])
  
  handles, labels = ax.get_legend_handles_labels()
  for i in range(len(labels)):
    if labels[i] not in legend:
      legend[labels[i]] = handles[i]
  
  ax.set_ylim(0, 0.2)
  ax.set_title(model_pn)
  ax.set_xlabel('Problem set')
  ax.set_xticks(ticks=xticks, labels=xticklabels)
  ax.set_ylabel('Error rate')

  col += 1
  if col >= columns:
    row += 1
    col = 0

fig.legend(handles=legend.values(), labels=legend.keys(),
           loc='upper center', ncols=ceil((num_types + len(sel_frames)) / 2))
fig.tight_layout()
fig.subplots_adjust(top=((graph_height * rows - 0.25)\
                         / (graph_height * rows + 0.6)))
save_path = path.join(type_error_dir, 'all_sel_models')
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.close(fig)

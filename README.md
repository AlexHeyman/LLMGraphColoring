# LLMGraphColoring

This repository hosts a tweaked version of the code used to run the experiments in the paper "Evaluating the Systematic Reasoning Abilities of Large Language Models through Graph Coloring", as well as records of the problems, prompts, and responses involved in those experiments.

## Requirements

This codebase was developed for Python 3.11.7, with the packages Anthropic 0.32.0, Fireworks AI 0.15.3, Google Generative AI 0.7.2, and OpenAI 1.59.3 used for interfacing with LLMs (see `models.py`), and SciPy 1.14.1 and Matplotlib 3.9.2 used for generating plots (see `summarize.py`). Earlier or later versions of the requirements may or may not work.

## Usage

`test.py`, `graph_coloring/generate.py`, `graph_coloring/categorize.py`, `graph_coloring/evaluate.py`, and `graph_coloring/summarize.py` are all executable code files with different functions; see their respective header docstrings for details.

Note that the compressed data archives in `graph_coloring/data`, `graph_coloring/prompts`, and `graph_coloring/responses` must be uncompressed before the executable code files can operate on the data properly.

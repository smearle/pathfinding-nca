import json
import os
from pdb import set_trace as TT
import re
from typing import List

import yaml
from config import ClArgsConfig

import pandas as pd


EVAL_DIR = 'runs_eval'


def vis_cross_eval(exp_cfgs: List[ClArgsConfig]):
    """
    Visualize the results of a set of experiments.

    Args:
        exp_cfgs (list): A list of experiment configs.
    """
    batch_exp_stats = []
    filtered_exp_cfgs = []
    for exp_cfg in exp_cfgs:
        try:
            with open(exp_cfg.log_dir + '/test_stats.json', 'r') as f:
                batch_exp_stats.append(json.load(f))
                filtered_exp_cfgs.append(exp_cfg)
        except FileNotFoundError as e:
            print(f"Stats not found. Error:\n {e}\n Skipping experiment stats.")
    
    batch_exp_cfgs = filtered_exp_cfgs
    exp_names = [cfg.exp_name for cfg in batch_exp_cfgs]
    
    # Assuming all stats have the same set of evaluated metrics.
    col_headers = list(batch_exp_stats[0].keys())

    data_rows = []
    for exp_stats in batch_exp_stats:
        data_rows.append([exp_stats[k] if k in exp_stats else None for k in col_headers])

    # Hierarchical rows, separating experiments according to relevant hyperparameters
    ignored_hyperparams = set({
        # 'minibatch_size',
    })
    hyperparams = [h for h in list(yaml.safe_load(open('configs/batch.yaml', 'r')).keys()) if h not in ignored_hyperparams]
    row_tpls = []
    for exp_cfg in batch_exp_cfgs:
        row_tpls.append([getattr(exp_cfg, h) for h in hyperparams])
    hyperparams = [h.replace('_', ' ') for h in hyperparams]
    row_indices = pd.MultiIndex.from_tuples(row_tpls, names=hyperparams)

    col_headers = [c.replace('_', ' ') for c in col_headers]
    df = pd.DataFrame(data_rows, columns=col_headers, index=row_indices)
    df.sort_index(inplace=True)
    # df.to_latex(os.path.join(EVAL_DIR, 'cross_eval.tex'), multirow=True)
    pandas_to_latex(df, os.path.join(EVAL_DIR, 'cross_eval.tex'), multirow=True, index=True, vertical_bars=True,
                     columns=col_headers)
    proj_dir = os.curdir
    os.system(f'cd {EVAL_DIR}; pdflatex tables.tex; cd {proj_dir}')
    

def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, header=True, index=False,
                    escape=False, multicolumn=False, **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.

    Args:
        df_table: dataframe
        latex_file: filename to write latex table code to
        vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
        right_align_first_column: Allows option to turn off right-aligned first column
        header: Whether or not to display the header
        index: Whether or not to display the index labels
        escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
        multicolumn: Enable better handling for multi-index column headers - adds midrules
        kwargs: additional arguments to pass through to DataFrame.to_latex()
    """
    n = len(df_table.columns) + len(df_table.index[0])

#   if right_align_first_column:
    cols = 'c' + 'r' * (n - 1)
#   else:
#       cols = 'r' * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
                              **kwargs)
    latex = latex.replace('\\begin{table}', '\\begin{table*}')
    latex = latex.replace('\end{table}', '\end{table*}')

    if vertical_bars:
        # Remove the booktabs rules since they are incompatible with vertical lines
        latex = re.sub(r'\\(top|mid|bottom)rule', r'\\hline', latex)

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{r}')
        latex = latex.replace(r'{c}', r'{r}')

        offset = len(df_table.index[0])
        #       offset = 1
        midrule_str = ''

        # Find horizontal start and end indices of multicols
        # Here's a hack for the evo cross eval table.
        hstart = 1 + offset
        hend = offset + len(kwargs['columns'])
        midrule_str += rf'\cline{{{hstart}-{hend}}}'

        # Ensure that headers don't get colored by row highlighting
        #       midrule_str += r'\rowcolor{white}'

        latex_lines = latex.splitlines()
        # FIXME: ad-hoc row indices here
        # TODO: get row index automatically then iterate until we get to the end of the multi-cols
        # latex_lines.insert(7, midrule_str)
        # latex_lines.insert(9, midrule_str)

        # detect start of multicol then add horizontal line between multicol levels
        for i, l in enumerate(latex_lines):
            if '\multicolumn' in l:
                mc_start = i
                break
        for i in range(len(df_table.columns[0]) - 1):
            latex_lines.insert(mc_start + i + 1, midrule_str)
        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)
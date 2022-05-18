import json
import os
from pathlib import Path
from pdb import set_trace as TT
import pickle
import re
from typing import List

import matplotlib.pyplot as plt
import yaml
from configs.config import BatchConfig, Config

import pandas as pd

from mazes import load_dataset


EVAL_DIR = os.path.join(Path(__file__).parent, 'runs_eval')

model_stat_keys = set({
    'n params'
})

names_to_hyperparams = {
    'batch': [
        "task",
        "model",
        "env_generation",
        "n_layers",
        "max_pool",
        "kernel_size",
        "loss_interval",
        "n_hid_chan",
        "shared_weights",
        "skip_connections",
        "cut_conv_corners",
        "sparse_update",
        "n_data",
        "learning_rate",
        "exp_name",
    ],
    'n_hid_chan': [
        'model',
        'n_hid_chan',
    ],
    'loss_interval': [
        'model',
        'shared_weights',
        'loss_interval',
    ],
    'weight_sharing': [
        'model',
        'shared_weights',
        'n_hid_chan',
    ],
    'cut_corners': [
        'model',
        'cut_conv_corners',
        'n_hid_chan',
    ],
    'models': [
        'model',
        'n_layers',
        'n_hid_chan',
    ],
    'diam_max_pool': [
        'model',
        'max_pool',
        'n_hid_chan',
    ],
    'diam_kernel': [
        'model',
        'kernel_size',
        'n_hid_chan',
    ],
}
names_to_cols = {
    'default': [
        'n_params',
        'TRAIN_pct_complete',
        'TRAIN_completion_time',
        'TEST_pct_complete',
        'TEST_completion_time',
    ],
    'loss_interval': [
        'TRAIN_pct_complete',
        'TRAIN_completion_time',
        'TEST_pct_complete',
        'TEST_completion_time',
    ],
    'models': [
        'n_params',
        # 'TRAIN_accs',
        'TRAIN_pct_complete',
        'TEST_accs',
        'TEST_pct_complete',
        'TEST_32_pct_complete',
    ],
    'diam_max_pool': [
        'n_params',
        'TRAIN_pct_complete',
        # 'TRAIN_completion_time',
        'TEST_pct_complete',
        # 'TEST_completion_time',
        'TEST_32_accs',
        'TEST_32_pct_complete',
    ]
}
hyperparam_renaming = {
    'n hid chan': 'n. hid chan',
    'cut conv corners': 'cut corners',
}
col_renaming = {
    'n params': 'n. params',
    'n updates': 'n. updates',
    'pct complete': 'pct. complete',
}


def vis_cross_eval(exp_cfgs: List[Config], batch_cfg: BatchConfig, task: str):
    """
    Visualize the results of a set of experiments.

    Args:
        exp_cfgs (list): A list of experiment configs.
    """
    name = batch_cfg.sweep_name
    if not batch_cfg.load_pickle:
        batch_exp_stats = []
        filtered_exp_cfgs = []
        for exp_cfg in exp_cfgs:
            try:
                with open(exp_cfg.log_dir + '/stats.json', 'r') as f:
                    exp_general_stats = json.load(f)
                with open(exp_cfg.log_dir + '/test_stats.json', 'r') as f:
                    exp_test_stats = json.load(f)
                    exp_test_stats = {f'TEST_{k}': v for k, v in exp_test_stats.items()}
                if exp_cfg.model != "MLP":
                    with open(exp_cfg.log_dir + '/test_32_stats.json', 'r') as f:
                        exp_test_32_stats = json.load(f)
                        exp_test_32_stats = {f'TEST_32_{k}': v for k, v in exp_test_32_stats.items()}
                else:
                    exp_test_32_stats = {}
                with (open(exp_cfg.log_dir + '/train_stats.json', 'r')) as f:
                    exp_train_stats = json.load(f)
                    exp_train_stats = {f'TRAIN_{k}': v for k, v in exp_train_stats.items()}
                exp_stats = {**exp_general_stats, **exp_train_stats, **exp_test_stats, **exp_test_32_stats}
                batch_exp_stats.append(exp_stats)
                filtered_exp_cfgs.append(exp_cfg)
            except FileNotFoundError as e:
                print(f"Stats not found. Error:\n {e}\n Skipping experiment stats.")
        
        batch_exp_cfgs = filtered_exp_cfgs
        
        if batch_cfg.selective_table:
            if name in names_to_cols:
                col_headers = names_to_cols[name]
            else:
                col_headers = names_to_cols['default']
        else:
            # Assuming all stats have the same set of evaluated metrics. (E.g., need to not put MLPs first, if including
            # other models, since they are not evaluated on larger mazes.)
            col_headers = list(batch_exp_stats[1].keys())

        data_rows = []
        for exp_stats in batch_exp_stats:
            data_rows.append([exp_stats[k] if k in exp_stats else None for k in col_headers])

        if batch_cfg.selective_table and name in names_to_hyperparams:
            hyperparams = names_to_hyperparams[name]
        else:
            hyperparams = names_to_hyperparams['batch']
        # Hierarchical rows, separating experiments according to relevant hyperparameters
        # hyperparams = [h for h in list(yaml.safe_load(open('configs/batch.yaml', 'r')).keys()) if h not in ignored_hyperparams]
        row_tpls = []
        for exp_cfg in batch_exp_cfgs:
            # row_tpls.append([getattr(exp_cfg, h) for h in hyperparams])
            row_tpl = []
            for h in hyperparams:
                if h == 'env_generation':
                    row_tpl.append(getattr(exp_cfg, h) is not None)
                # elif h == 'exp_name':
                    # FIXME: hack
                    # row_tpl.append(getattr(exp_cfg, h).split('_')[-1])
                else:
                    row_tpl.append(getattr(exp_cfg, h))
            row_tpls.append(row_tpl)
        hyperparams = [h.replace('_', ' ') for h in hyperparams]
        hyperparams = [hyperparam_renaming[h] if h in hyperparam_renaming else h for h in hyperparams]
        row_indices = pd.MultiIndex.from_tuples(row_tpls, names=hyperparams)

        col_headers = [c.replace('_', ' ') for c in col_headers]
        col_tpls = []
        for c in col_headers:
            if 'TEST' in c:
                if 'TEST 32' in c:
                    col_tpls.append(('test', '32x32', c.replace('TEST 32 ', '')))
                else:
                    col_tpls.append(('test', '16x16', c.replace('TEST ', '')))
            elif 'TRAIN' in c:
                col_tpls.append(('train', '16x16', c.replace('TRAIN ', '')))
            elif c in model_stat_keys:
                col_tpls.append(('model', '---', c))
            else:
                col_tpls.append(('experiment', '---', c))
            # col_tpl = ('test' if 'TEST' in c else 'train', c.replace('TEST ', '').replace('TRAIN ', ''))
            # col_tpls.append(col_tpl)
        col_tpls = [(c[0], c[1], col_renaming[c[-1]] if c[-1] in col_renaming else c[-1]) for c in col_tpls]
        col_indices = pd.MultiIndex.from_tuples(col_tpls)  #, names=['type', 'metric'])

        df = pd.DataFrame(data_rows, columns=col_indices, index=row_indices)
        # df.sort_index(inplace=True)

        for k in col_indices:
            if k in df:
                print(k[-1])
                df[k] = df[k].apply(
                    lambda data: preprocess_values(data, 
                                    col_name=k)
                )
        df.to_csv(os.path.join(EVAL_DIR, f'{task}_{name}_cross_eval.csv'))

        # Save dataframe using pickle
        with open(os.path.join(EVAL_DIR, f'{task}_{name}_cross_eval.pkl'), 'wb') as f:
            pickle.dump(df, f)

    else: 
        with open(os.path.join(EVAL_DIR, f'{batch_cfg.batch_hyperparams}_cross_eval.pkl'), 'rb') as f:
            df = pickle.load(f)
        col_indices = df.columns

    for k in col_indices:
        if k in df:
            print(k[-1])
            df[k] = df[k].apply(
                lambda data: bold_extreme_values(data, 
                                data_max=(max if ('completion time' not in k[-1] and 'losses' not in k[-1]) else min)\
                                    ([(d[0] if isinstance(d, tuple) else d) for d in df[k]]), 
                                col_name=k)
            )
    
    raw_tbl_tex_name = f'{task}_{name}_cross_eval{("_selective" if batch_cfg.selective_table else "")}.tex'

    # df.to_latex(os.path.join(EVAL_DIR, 'cross_eval.tex'), multirow=True)
    pandas_to_latex(
        df, 
        os.path.join(EVAL_DIR, raw_tbl_tex_name), 
        multirow=True, 
        index=True, 
        header=True,
        vertical_bars=True,
        columns=col_indices, 
        multicolumn=True, 
        multicolumn_format='c|',
        right_align_first_column=False,
        bold_rows=True,
        )
    proj_dir = os.curdir

    # Read in the latex template
    with open(os.path.join(proj_dir, EVAL_DIR, 'table_template.tex'), 'r') as f:
        temp = f.read()

    # Replace the template input with path to file
    temp = temp.replace('INPUT', raw_tbl_tex_name)

    tables_tex_name = f"{task}_{name}_table{('_selective' if batch_cfg.selective_table else '')}.tex"

    # Write the output to a file.
    with open(os.path.join(proj_dir, EVAL_DIR, tables_tex_name), 'w') as f:
        f.write(temp)

    os.system(f'cd {EVAL_DIR}; pdflatex {tables_tex_name}; cd {proj_dir}')


def preprocess_values(data, col_name=None):
    "Preprocessing values in dataframe (output floats)."

    # hack
    if isinstance(data, int) or isinstance(data, float):
        return data

    if data is None:
        return 0 

    # Assume (mean, std)
    data, err = data
    # Other preprocessing here
    if col_name[-1] not in set({"completion time", "n_params"}):
        data *= 100
        err *= 100
    return data, err


def bold_extreme_values(data, data_max=-1, col_name=None):
    "Process dataframe values from floats into strings. Bold extreme (best) values."

    # hack
    if col_name[-1] in set({"n. params", "n. updates"}):
        return '{:,}'.format(data)

    if isinstance(data, int):
        assert data == 0
        return '---'

    else:
        # TODO: get standard deviation in a smart way.
        data, err = data

    if data == data_max:
        bold = True
    else: 
        bold = False

    data = '{:.2f}'.format(data)

    if err is not None:
        err = '{:.2f}'.format(err)
    
    if bold:
        data = '\\textbf{' + str(data) + '}'
        if err is not None:
            data += ' ± ' + str(err)
    else:
        data = f'{data}'
        if err is not None:
            data += f' ± {err}'
    return data


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
        k = 0
        for i, l in enumerate(latex_lines):
            if '\multicolumn' in l:
                mc_start = i
                for j in range(len(df_table.columns[0]) - 1):
                    latex_lines.insert(mc_start + j + 1, midrule_str)
                k += 1
        # for j in range(len(df_table.columns[0]) - 1):
            # latex_lines.insert(mc_start + j + 1 + k, midrule_str)
        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)


plot_label_map = {
    'completion time': 'Completion Time',
    'n. hid chan': 'Number of hidden channels',
    'loss interval': 'Loss interval',
}


def plot_column(df, row_name, col_name):
    # Get all `row_name` row names
    xs = df.index.get_level_values(row_name).unique()
    # Get all `col_name` values
    # Drop row names and convert values to numpy array
    ys_errs_train = df[('train', col_name)].values
    ys_train, errs_train = zip(*ys_errs_train)
    ys_errs_test = df[('test', col_name)].values
    ys_test, errs_test = zip(*ys_errs_test)
    # Plot as a histogram with error bars
    from matplotlib import rc
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(['science','no-latex'])
    plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":11}) 

    #rc('text', usetex=True)
    #rc('font', family='serif')
    plt.errorbar(xs, ys_train, yerr=errs_train,linewidth=2, 
        label=r'train')
    plt.errorbar(xs, ys_test, yerr=errs_test,linewidth=2,
        label=r'test')
    plt.xlabel(rf'{plot_label_map[row_name]}', color="black", size=10)
    plt.ylabel(r'Percentage', color="black", size=10)
    plt.title(r'Path-finding percentage mean and variance', color="black", size=10)
    plt.legend()
    # Save
    plt.savefig(os.path.join(EVAL_DIR, f'{row_name}_{col_name}.jpg'), dpi=2000)
    plt.close()
    

if __name__ == '__main__':
    # Load maze data and compute average path length.
    train_data, val_data, test_data = load_dataset()
    train_trgs, val_trgs, test_trgs = train_data.target_paths, val_data.target_paths, test_data.target_paths
    train_path_lengths, val_path_lengths, test_path_lengths = train_trgs.sum((1,2)).float().cpu().numpy(), \
        val_trgs.sum((1,2)).float().cpu().numpy(), test_trgs.sum((1,2)).float().cpu().numpy()
    with open(os.path.join(EVAL_DIR, 'path_lengths.json'), 'w') as f:
        json.dump({'train': train_path_lengths.mean().item(), 'val': val_path_lengths.mean().item(), 
        'test': test_path_lengths.mean().item()}, f)

    # Load dataframe using pickle
    df = pd.read_pickle(os.path.join(EVAL_DIR, 'n_hid_chan_cross_eval.pkl'))
    plot_column(df, 'n. hid chan', 'pct. complete')
    plot_column(df, 'n. hid chan', 'completion time')

    df = pd.read_pickle(os.path.join(EVAL_DIR, 'loss_interval_cross_eval.pkl'))
    plot_column(df, 'loss interval', 'pct. complete')
    plot_column(df, 'loss interval', 'completion time')
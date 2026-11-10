"""Run the supplied RSM notebooks without requiring a Jupyter installation."""
import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_table(value):
    print(value.to_string() if hasattr(value, 'to_string') else value)


def run_one(notebook, output):
    output.mkdir(parents=True, exist_ok=True)
    namespace={'__name__':'__main__','display':display_table}
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    old_cwd=Path.cwd()
    original_show=plt.show
    plt.show=lambda *args,**kwargs: None
    try:
        os.chdir(notebook.parent)
        with (output/'report.txt').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
            for i,cell in enumerate(cells):
                if cell['cell_type']=='code':
                    exec(compile(''.join(cell.get('source',[])),f'{notebook.name}:cell{i}','exec'),namespace)
        model=namespace['ols_model']
        metrics={'n':int(model.nobs),'r_squared':float(namespace['r2']),
                 'adjusted_r_squared':float(namespace['adj_r2']),
                 'rmse':float(namespace['rmse']),
                 'evaluation':'In-sample fit of all observations; no held-out test set.'}
        model.params.rename('coefficient').to_csv(output/'coefficients.csv',index_label='term')
        frame=namespace['df']
        pd.DataFrame({'PC1':frame['x'],'PC2':frame['y'],'observed':frame['z'],
                      'fitted':namespace['z_pred'],'residual':namespace['residual']}).to_csv(output/'fitted_values.csv',index=False)
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
        namespace['fig1'].savefig(output/'response_surface.png',dpi=180,bbox_inches='tight')
        print(notebook.name,json.dumps(metrics))
    finally:
        plt.close('all');plt.show=original_show;os.chdir(old_cwd)


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=root/'results')
    args=parser.parse_args()
    output=args.output.resolve()
    start=time.perf_counter()
    for stem in ['Cap','RE']:
        run_one(root/'scientificProject'/f'PC1-PC2-{stem}.ipynb',output/stem)
    report={'python':platform.python_version(),'platform':platform.platform(),
            'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib','openpyxl','scikit-learn']},
            'notebook_execution_and_export_seconds':time.perf_counter()-start}
    (output/'environment_and_timing.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()

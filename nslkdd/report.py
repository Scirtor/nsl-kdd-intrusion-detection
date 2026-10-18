"""Regenerate figures from saved metrics without fitting or selecting models."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from nslkdd.data import ROOT


def plot_report(report, output_dir):
    output_dir=Path(output_dir)
    rows=report['validation_results']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.barh([r['model'] for r in rows],[r['f1'] for r in rows],color=['#888888']+['#a9d94e']*(len(rows)-1))
    ax.set(xlim=(0,1.12),xlabel='Attack-class F1',title='Model selection on training-file validation data')
    for i,r in enumerate(rows):ax.text(r['f1']+.01,i,f'{r["f1"]:.4f}',va='center')
    fig.tight_layout();fig.savefig(output_dir/'validation-comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,5.3))
    palette=LinearSegmentedColormap.from_list('pink_lime',['#f8e4ee','#a9d94e'])
    display=ConfusionMatrixDisplay(np.asarray(report['official_test']['confusion_matrix']),display_labels=['Normal','Attack'])
    display.plot(ax=ax,cmap=palette,colorbar=False)
    for annotation in display.text_.ravel():annotation.set_color('#17201b')
    ax.set_title(f'{report["selected_model"]} on official KDDTest+');fig.tight_layout()
    fig.savefig(output_dir/'test-confusion.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/baseline')
    args=parser.parse_args()
    plot_report(json.loads((args.run_dir/'metrics.json').read_text()),args.run_dir)

"""Render measured results as standalone research figures (no network)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('runs/pilot-v1');s=json.loads((root/'summary.json').read_text());b=json.loads((root/'jevbench-public/summary.json').read_text())
    pub=json.loads(Path('reports/published-public-comparison.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),layout='constrained')
    fig.suptitle('OpenJeff • first trained pilot',fontsize=21,fontweight='bold',ha='left',x=.015)
    colors=['#728392','#cf303a']
    vals=[100*s['models'][k]['splits']['final']['raw']['accuracy'] for k in ('base','adapter')]
    axes[0].bar(['Base 12B','OpenJeff'],vals,color=colors,width=.6)
    axes[0].set(title='Synthetic final set · 600 decisions',ylim=(0,108),ylabel='Accuracy (%)')
    for i,v in enumerate(vals):axes[0].text(i,v+1.5,f'{v:.1f}%',ha='center',fontweight='bold')
    vals=[100*b['models'][k]['hard']['accuracy'] for k in ('base','adapter')]+[100*pub['systems']['jev-1.13.0']['tiers']['hard']['accuracy']]
    axes[1].bar(['Base 12B','OpenJeff','Jev¹'],vals,color=colors+['#1e3c36'],width=.6)
    axes[1].set(title='Public JevBench hard · 111 decisions',ylim=(0,108))
    for i,v in enumerate(vals):axes[1].text(i,v+1.5,f'{v:.1f}%',ha='center',fontweight='bold')
    vals=[s['models'][k]['splits']['final']['calibrated']['ece'] for k in ('base','adapter')]
    axes[2].bar(['Base 12B','OpenJeff'],vals,color=colors,width=.6)
    axes[2].set(title='Synthetic final calibration error',ylabel='ECE · 15 bins · lower is better',ylim=(0,max(vals)*1.45+.01))
    for i,v in enumerate(vals):axes[2].text(i,v+.002,f'{v:.4f}',ha='center',fontweight='bold')
    for a in axes:a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.supxlabel('197/600 synthetic final states match training after ID normalization; see the full overlap audit.\nPublic benchmark excluded from training and selection.\n¹Publisher result on the same item IDs; different runtime and prompt. This is not an official benchmark rank.',fontsize=9)
    out=Path('reports/figures');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'pilot-results.png',dpi=160);fig.savefig(out/'pilot-results.svg');plt.close(fig)

if __name__=='__main__':main()

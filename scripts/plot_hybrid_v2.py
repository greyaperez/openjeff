"""Create a static, exportable figure from validated experiment metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
methods=['ar','ar_view','diffusion','diffusion_guided_ar','ar_guided_ar','fusion']
labels=['AR','AR + map','Diffusion','Diffusion → AR','AR → AR','Probability blend']

def main():
    report=json.loads((ROOT/'reports/hybrid-v2-results.json').read_text())
    assert json.loads((ROOT/'reports/hybrid-v2-validation.json').read_text())['status']=='passed'
    fig,axes=plt.subplots(2,2,figsize=(14,10));fig.patch.set_facecolor('#fafafa')
    colors=['#174f69','#d99a28'];x=np.arange(len(methods));width=.36
    panels=[('Synthetic accuracy','final','final_no_normalized_train_match','All 600','403 without normalized training match'),('Public JevBench accuracy','public','public_hard','Public 231','Hard 111')]
    for ax,(title,s1,s2,l1,l2) in zip(axes[0],panels):
        for j,(split,name) in enumerate(((s1,l1),(s2,l2))):
            values=[100*report['methods'][m]['splits'][split]['raw']['accuracy'] for m in methods]
            bars=ax.bar(x+(j-.5)*width,values,width,label=name,color=colors[j]);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
        ax.set_ylim(0,122);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Correct (%)');ax.set_title(title,loc='left',fontweight='bold');ax.legend(loc='upper left',frameon=False,fontsize=8)
    ax=axes[1,0]
    for j,(key,name) in enumerate((('raw','Raw'),('calibrated','Temperature calibrated'))):
        values=[report['methods'][m]['splits']['public'][key]['nll'] for m in methods]
        bars=ax.bar(x+(j-.5)*width,values,width,label=name,color=colors[j]);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set_title('Public probability quality',loc='left',fontweight='bold');ax.set_ylabel('Mean negative log likelihood · lower is better');ax.legend(frameon=False,fontsize=8);ax.margins(y=.30)
    ax=axes[1,1]
    for j,(key,name) in enumerate((('p50_s','Median'),('p95_s','95th percentile'))):
        values=[1000*report['methods'][m]['splits']['public']['serial_latency'][key] for m in methods]
        bars=ax.bar(x+(j-.5)*width,values,width,label=name,color=colors[j]);ax.bar_label(bars,fmt='%.0f',fontsize=8,padding=3)
    ax.set_title('Public tasks: warmed stage times',loc='left',fontweight='bold');ax.set_ylabel('Milliseconds · separate GPU phases');ax.legend(frameon=False,fontsize=8);ax.margins(y=.18)
    for ax in axes.flat:
        ax.set_xticks(x,labels,rotation=25,ha='right',fontsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.suptitle('OpenJeff · Does diffusion improve the frozen AR model?',fontsize=19,fontweight='bold',x=.055,ha='left')
    fig.text(.055,.028,'No model retraining or latent bridge. Synthetic partitions share rule templates and repeated logical states.\nPublic results cover 231 items, not the full official benchmark. Stage times exclude model loading, GPU swaps, queues and concurrency.\nThe development-selected probability blend assigns diffusion zero weight and reduces to AR alone.',fontsize=9,color='#444')
    fig.tight_layout(rect=(.035,.085,.99,.95));dest=ROOT/'reports/figures/hybrid-v2-results.png';dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest,dpi=180,facecolor=fig.get_facecolor());fig.savefig(dest.with_suffix('.pdf'),facecolor=fig.get_facecolor());print(dest)

if __name__=='__main__':main()

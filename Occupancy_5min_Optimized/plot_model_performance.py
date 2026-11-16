"""Run: python plot_model_performance.py (requires matplotlib)."""
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/test_metrics.json'
m = json.loads(SOURCE.read_text(encoding='utf-8'))
tn, fp = m['confusion_matrix'][0]
fn, tp = m['confusion_matrix'][1]
home_f1 = 2 * tp / (2 * tp + fp + fn)
away_f1 = 2 * tn / (2 * tn + fp + fn)
assert abs((home_f1 + away_f1) / 2 - m['macro_f1']) < 1e-10
# KTH-inspired blue palette; aggregate measures lead each metric group.
navy, blue, light = '#000061', '#004791', '#6298D2'
macro_precision = (m['home_precision'] + m['away_precision']) / 2
labels = ['Balanced accuracy', 'Home recall', 'Away recall',
          'Macro precision', 'Home precision', 'Away precision',
          'Macro F1', 'Home F1', 'Away F1', 'Accuracy']
values = [m['balanced_accuracy'], m['home_recall'], m['away_recall'],
          macro_precision, m['home_precision'], m['away_precision'],
          m['macro_f1'], home_f1, away_f1, m['accuracy']]
positions = [0, 1, 2, 3.7, 4.7, 5.7, 7.4, 8.4, 9.4, 11.1]
colors = [navy, blue, light] * 3 + [navy]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(13, 10), facecolor='white')
fig.subplots_adjust(left=.28, right=.93, top=.79, bottom=.19)
fig.add_artist(plt.Rectangle((0, .975), 1, .025, transform=fig.transFigure, color=blue, linewidth=0))
ax.barh(positions, values, color=colors, height=.65)
ax.set_yticks(positions, labels)
ax.invert_yaxis()
ax.set_xlim(0, 1.06)
ax.set_xticks([0, .2, .4, .6, .8, 1])
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.xaxis.grid(True, color='#DEE7F0')
ax.set_axisbelow(True)
ax.tick_params(axis='both', length=0, pad=10, labelcolor=navy)
for spine in ax.spines.values():
    spine.set_visible(False)
for i, label in enumerate(ax.get_yticklabels()):
    if i in [0, 3, 6, 9]:
        label.set_weight('bold')
for y, value in zip(positions, values):
    ax.text(value + .012, y, f'{value:.1%}', va='center', fontsize=12, weight='bold', color=navy)
for y in [2.85, 6.55, 10.25]:
    ax.axhline(y, color='#D8E3EE', linewidth=.8)
fig.text(.055, .915, 'Occupancy model performance', fontsize=26, weight='bold', color=navy)
fig.text(.055, .87, 'Optimized 5-minute model | Historical test: 25–31 May 2020', fontsize=13, color=blue)
fig.text(.055, .833, f"{m['test_samples']} readings · {m['home_count']} Home · Validation-selected threshold: {m['threshold']:.2f}", fontsize=11, color='#475569')
fig.text(.055, .115, 'Balanced accuracy = macro recall. Macro measures give Home and Away equal weight.', fontsize=11, color=navy)
fig.text(.055, .079, f"Always-Home baseline: {m['home_count']/m['test_samples']:.1%} accuracy / 50.0% balanced accuracy.", fontsize=10, color='#475569')
fig.text(.055, .045, 'Target: current occupancy at each five-minute reading. Historical test week previously inspected; not an untouched test.', fontsize=9, color='#64748B')
for extension in ['png', 'svg', 'pdf']:
    path = ROOT / f'Model_Performance_5min.{extension}'
    fig.savefig(path, dpi=240, facecolor='white')
    print(path)
plt.close(fig)
with (ROOT / 'Model_Performance_5min.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Score', 'Percent'])
    writer.writerows((label, value, value * 100) for label, value in zip(labels, values))

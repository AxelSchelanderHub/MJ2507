"""Run: python plot_model_performance.py (requires matplotlib)."""
from pathlib import Path
import json
import argparse
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
parser.add_argument('--output-dir', type=Path, default=ROOT)
args = parser.parse_args()
SOURCE = args.results_dir / 'test_metrics.json'
ROOT = args.output_dir
ROOT.mkdir(parents=True, exist_ok=True)
m = json.loads(SOURCE.read_text(encoding='utf-8'))
protocol = json.loads((SOURCE.parent/'protocol.json').read_text(encoding='utf-8'))
period_start = pd.Timestamp(protocol['final_split']['score_start'])
period_end = pd.Timestamp(protocol['final_split']['score_end']) - pd.Timedelta(seconds=1)
period = f'{period_start:%d %b %Y} – {period_end:%d %b %Y}'
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
fig.text(.055, .87, f'Optimized 5-minute model | Historical test: {period}', fontsize=13, color=blue)
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

# Read saved predictions only: plotting never fits a model or selects a threshold.
predictions = pd.read_csv(SOURCE.parent / 'test_predictions.csv', parse_dates=['Timestep'])
actual = predictions['Presence (1 = presence)'].to_numpy(dtype=int)
predicted = predictions['predicted'].to_numpy(dtype=int)
times = predictions['Timestep']
threshold = float(m['threshold'])
cm = np.array([[np.sum((actual == a) & (predicted == b)) for b in (0, 1)] for a in (0, 1)])
assert np.array_equal(cm, m['confusion_matrix'])
np.testing.assert_array_equal(predicted, (predictions['score_home'] >= threshold).astype(int))

def save_figure(figure, name):
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(ROOT / f'{name}.{extension}', dpi=220, facecolor='white')
    plt.close(figure)

def draw_confusion(axis):
    fractions = cm / cm.sum(axis=1, keepdims=True)
    axis.imshow(fractions, cmap='Blues', vmin=0, vmax=1)
    axis.set_xticks([0, 1], ['Away', 'Home'])
    axis.set_yticks([0, 1], ['Away', 'Home'])
    axis.set_xlabel('Predicted occupancy')
    axis.set_ylabel('Actual occupancy')
    axis.set_title('Confusion matrix', color=navy, weight='bold', pad=16)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f'{cm[i,j]:,}\n{fractions[i,j]:.1%}\nof actual {("Away", "Home")[i]}',
                      ha='center', va='center', fontsize=10,
                      color='white' if fractions[i,j] > .55 else navy)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

import matplotlib.dates as mdates
def draw_timeline(axis):
    axis.plot(times, predictions['score_home'], color=blue, linewidth=1.3, label='Predicted P(Home)')
    axis.step(times, actual, where='post', color='#374151', linewidth=1.1, alpha=.8,
              label='Actual presence (1 = Home)')
    axis.axhline(threshold, color='#B42318', linestyle='--', linewidth=1.6,
                 label=f'Classification threshold = {threshold:.2f}')
    axis.set_ylim(-.06, 1.06)
    axis.set_ylabel('Probability / actual state')
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axis.grid(alpha=.2)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=3, frameon=False, fontsize=10)
    axis.set_xlim(times.iloc[0], times.iloc[-1])

fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(top=.83, bottom=.18)
fig.suptitle(f'5-minute model | {period}', fontsize=18, color=navy, weight='bold')
draw_confusion(ax)
fig.text(.08, .06, f'Historical test: {len(actual):,} readings. Colours show recall within each actual class.', fontsize=10, color=navy)
save_figure(fig, 'Confusion_Matrix_5min')

fig = plt.figure(figsize=(16, 11))
grid = fig.add_gridspec(3, 2, height_ratios=[2.1, .8, 2.4], hspace=.65, wspace=.35)
fig.subplots_adjust(top=.84, bottom=.10, left=.07, right=.96)
ax = fig.add_subplot(grid[0, :])
draw_timeline(ax)
states = fig.add_subplot(grid[1, :], sharex=ax)
states.imshow(np.vstack([actual, predicted]), aspect='auto', interpolation='nearest',
              cmap=matplotlib.colors.ListedColormap(['#E5EDF5', blue]), vmin=0, vmax=1,
              extent=[mdates.date2num(times.iloc[0]), mdates.date2num(times.iloc[-1]), 1.5, -.5])
states.set_yticks([0, 1], ['Actual', 'Predicted'])
states.xaxis.set_major_locator(mdates.DayLocator())
states.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
states.set_title('Occupancy states — blue: Home; pale blue: Away', loc='left', fontsize=11, color=navy)
states.set_xlabel('Date (historical test week)')
draw_confusion(fig.add_subplot(grid[2, 0]))
scores = fig.add_subplot(grid[2, 1])
summary_labels = ['Balanced accuracy', 'Home recall', 'Away recall', 'Accuracy', 'Macro precision', 'Macro F1']
summary_values = [m['balanced_accuracy'],m['home_recall'],m['away_recall'],m['accuracy'],m['macro_precision'],m['macro_f1']]
scores.barh(summary_labels, summary_values, color=[navy,blue,light,navy,navy,navy], height=.65)
scores.invert_yaxis()
scores.set_xlim(0,1.13)
scores.set_xticks([0,.25,.5,.75,1])
scores.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
scores.grid(axis='x',alpha=.18)
scores.set_axisbelow(True)
scores.set_title('Test performance', color=navy, weight='bold', pad=16)
for i,v in enumerate(summary_values):
    scores.text(v+.015,i,f'{v:.1%}',va='center',color=navy,weight='bold')
for spine in scores.spines.values():
    spine.set_visible(False)
fig.suptitle('5-minute occupancy model | Test-week evaluation', fontsize=24,color=navy,weight='bold',y=.975)
fig.text(.07,.925,f'{period} · Current occupancy · Saved predictions · Threshold selected on validation only',fontsize=12,color=blue)
fig.text(.07,.035,'Historical dates previously inspected; not an untouched test. Plotting does not retrain the model or adjust its threshold.',fontsize=10,color='#475569')
save_figure(fig,'Test_Week_Dashboard_5min')

fig, ax = plt.subplots(figsize=(16, 5.5))
fig.subplots_adjust(top=.70,bottom=.17,left=.065,right=.98)
draw_timeline(ax)
ax.set_xlabel(f'Date | {period} (historical test)')
fig.suptitle('5-minute model | Predicted probability versus actual presence',fontsize=21,color=navy,weight='bold',y=.96)
save_figure(fig,'Test_Timeline_5min')
print('Also saved confusion matrix, test timeline and combined dashboard as PNG, SVG and PDF.')

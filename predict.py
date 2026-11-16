"""Predict current occupancy from chronological five-minute sensor readings."""
import argparse
from pathlib import Path
from train_model import c
p=argparse.ArgumentParser()
p.add_argument('data',type=Path)
p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'results/model_5min.npz')
p.add_argument('--output',type=Path,default=Path('predictions.csv'))
a=p.parse_args()
c.predict_saved(c.load_data(a.data,labels=False),a.model).to_csv(a.output,index=False)
print(a.output)

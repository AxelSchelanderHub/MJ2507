import copy
import json
from pathlib import Path
import unittest
from configuration import validate, candidates, make_splits

class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).parent/'config.json').read_text())

    def test_original_candidates(self):
        validate(self.cfg)
        models=candidates(self.cfg)
        self.assertEqual(len(models),24)
        self.assertEqual(models[11].lr,.01)
        self.assertEqual(models[11].family,'levels_clock')

    def test_grid_and_splits(self):
        self.cfg['search']['mode']='grid'
        self.assertEqual(len(candidates(self.cfg)),9)
        self.cfg['model']['patience_checks']=4
        self.cfg['split']['calibration_hours']=24
        validate(self.cfg)
        self.assertTrue(all(m.patience_checks==4 for m in candidates(self.cfg)))
        _,split=make_splits(self.cfg)
        self.assertEqual((split['cal_end']-split['cal_start']).total_seconds(),86400)

    def test_invalid_settings(self):
        for key,value in [('hidden',0),('lr',-1),('activation','invalid'),('patience_checks',0)]:
            cfg=copy.deepcopy(self.cfg)
            cfg['model'][key]=value
            with self.assertRaises(ValueError): validate(cfg)
        self.cfg['split']['validation_starts']=['2020-05-26']
        with self.assertRaises(ValueError): validate(self.cfg)

if __name__=='__main__': unittest.main()

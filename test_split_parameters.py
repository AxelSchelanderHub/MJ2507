"""Check parameterized temporal splits without rerunning hyperparameter search."""
import unittest
import model_core as c

class SplitTests(unittest.TestCase):
    def test_defaults_preserve_original(self):
        for split in [*c.folds(),c.final_fold()]:
            s=split['score_start']
            for name,days in [('train_end',7),('stop_start',6),('stop_end',4),('cal_start',3),('cal_end',1)]:
                self.assertEqual(split[name],s-c.pd.Timedelta(days=days))

    def test_custom_windows(self):
        for stop,gap in [(24,24),(72,12),(12,0),(36,6)]:
            for split in [*c.folds(stop,gap),c.final_fold(stop,gap)]:
                self.assertEqual(split['stop_end']-split['stop_start'],c.pd.Timedelta(hours=stop))
                for left,right in [('train_end','stop_start'),('stop_end','cal_start'),('cal_end','score_start')]:
                    self.assertEqual(split[right]-split[left],c.pd.Timedelta(hours=gap))
                self.assertLessEqual(split['score_end'],c.TEST_END)

    def test_invalid(self):
        for stop,gap in [(0,24),(-1,24),(48,-1),(float('nan'),24),(48,float('inf'))]:
            with self.assertRaises(ValueError):
                c.final_fold(stop,gap)

if __name__=='__main__':
    unittest.main()

import unittest

import run_experiment as rexp
class Tests(unittest.TestCase):
    
    def test_user_rec(self):
        test, mae = rexp.run_user_rec_experiment(3)
        
        self.assertEqual(test["user_id"], 130, "Should be 130")
        self.assertEqual(test["movie_id"], 109, "Should be 109")
        self.assertEqual(test["rating"], 3.0, "Should be 3.0")
        self.assertEqual(test["pred_rating"], 3.35, "Should be 3.35")
        self.assertEqual(mae, 0.77459, "Should be 0.77459")


    def test_item_rec(self):
        test, mae = rexp.run_item_rec_experiment(3)
        
        self.assertEqual(test["user_id"], 221, "Should be 221")
        self.assertEqual(test["movie_id"], 943, "Should be 943")
        self.assertEqual(test["rating"], 4.0, "Should be 4.0")
        self.assertEqual(test["pred_rating"], 4.3, "Should be 4.3")
        self.assertEqual(mae, 0.66193, "Should be 0.66193")
        
        
    def test_bootstrap_rec(self):
        test, mae = rexp.run_bootstrap_rec_experiment(3)
        
        self.assertEqual(test["user_id"], 222, "Should be 222")
        self.assertEqual(test["movie_id"], 366, "Should be 366")
        self.assertEqual(test["rating"], 4.0, "Should be 4.0")
        self.assertEqual(test["pred_rating"], 3.68, "Should be 3.68")
        self.assertEqual(mae, 1.11184, "Should be 1.11184")
        
        
    def test_pearlpu_rec(self):
        test, mae = rexp.run_pearlpu_rec_experiment(3)
        
        self.assertEqual(test["user_id"], 160, "Should be 160")
        self.assertEqual(test["movie_id"], 174, "Should be 174")
        self.assertEqual(test["rating"], 5.0, "Should be 5.0")
        self.assertEqual(test["pred_rating"], 4.62, "Should be 4.62")
        self.assertEqual(mae, 0.69621, "Should be 0.69621")
        
        
    '''def test_corec_rec(self):
        test, mae = rexp.run_corec_rec_experiment(3)
        
        self.assertEqual(test["user_id"], 130, "Should be 130")
        self.assertEqual(test["movie_id"], 109, "Should be 109")
        self.assertEqual(test["rating"], 3.0, "Should be 3.0")
        self.assertEqual(test["pred_rating"], 3.35, "Should be 3.35")
        self.assertEqual(mae, 0.77459, "Should be 0.77459")'''

if __name__ == "__main__":
    unittest.main()    
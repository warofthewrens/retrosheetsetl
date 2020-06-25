import unittest
from extract import extract_team, extract_roster, extract_roster_team
from raw_schemas.play import Play

class TestExtract(unittest.TestCase):

    def test_basic_outs(self):
        '''
        test the result of loading a basic out into Play
        '''
        play = {'play':'53/G.1-3'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'53(B)/G.1-3'})

        play = {'play':'7/F7'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'7(B)/F7'})

        play = {'play':'64(1)3/GDP/G6'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'64(1)3(B)/GDP/G6'})

        # maybe make B-1 explicit
        play = {'play':'54(1)/FO/G5.3-H'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'54(1)/FO/G5.3-H'})
    
    def test_basic_errors(self):
        play = {'play':'E6.1-3'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'E6(B).1-3'})
    
    def test_thrown_outs(self):
        play = {'play':'K/DP.1X2(26)'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'K/DP.1X2(26(1))'})

        play = {'play':'S7/L7LD.3-H;2-H;BXH(7E4)(NR)(TUR)'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'S7/L7LD.3-H;2-H;BXH(7E4(B))(NR)(TUR)'})

        play = {'play':'S5/G5.1-3(TH)(E5/TH)(UR)'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'S5/G5.1-3(TH)(E5(1))(UR)'})
    
    def test_missing_runner(self):
        play = {'play':'46(1)/FO/G.3-H;2-H(E6)(E6/TH)(NR)'}
        result = Play().load(play)
        self.assertEqual(result, {'play':'46(1)/FO/G.3-H;2-H(E6(2))(E6(2))(NR)'})
    
    def test_lots_of_parens(self):
        play = {'play': 'D7/L+.1-H;B-H(TH)(E2/TH)(UR)(NR)'}
        result = Play().load(play)
        self.assertEqual(result, {'play': 'D7/L+.1-H;B-H(TH)(E2(B))(UR)(NR)'})

if __name__ == '__main__':
    unittest.main()
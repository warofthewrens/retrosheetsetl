import unittest
from extract import extract_team, extract_roster, extract_roster_team
from raw_schemas.play import Play

class TestExtract(unittest.TestCase):
    def test_basic_outs(self):
        '''
        test the result of loading a basic out into Play
        '''
        play = '53/G.1-3'
        result = Play().load(play)
        self.assertEqual(result, '53(B)/G.1-3')

        play = '7/F7'
        result = Play().load(play)
        self.assertEqual(result, '7(B)/F7')

        play = '64(1)3/GDP/G6'
        result = Play().load(play)
        self.assertEqual(result, '64(1)3(B)/GDP/G6')

        # maybe make B-1 explicit
        play = '54(1)/FO/G5.3-H'
        result = Play().load(play)
        self.assertEqual(result, '54(1)/FO/G5.3-H')
    
    def test_basic_errors(self):
        play = 'E6.1-3'
        result = Play().load(play)
        self.assertEqual(result, 'E6(B).1-3')
    
    def test_thrown_outs(self):
        play = 'K/DP.1X2(26)'
        result = Play().load(play)
        self.assertEqual(result, 'K/DP.1X2(26(1))')

        play = 'S7/L7LD.3-H;2-H;BX2(7E4)'
        result = Play().load(play)
        self.assertEqual(result, 'S7/L7LD.3-H;2-H;BX2(7E4(B))')

        play = 'S5/G5.1-3(E5/TH)'
        result = Play().load(play)
        self.assertEqual(result, 'S5/G5.1-3(E5(B)/TH)')
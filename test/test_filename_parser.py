# test/test_filename_parser.py
# -*- coding: utf-8 -*-
"""
Filename parser unit tests
"""

import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filename_parser import parse_filename, group_videos, _detect_separator


class TestDetectSeparator(unittest.TestCase):

    def test_underscore(self):
        self.assertEqual(_detect_separator("FIVB_BVB_WT19_Edmonton"), '_')

    def test_hyphen(self):
        self.assertEqual(_detect_separator("FIVB-BVB-WT18-Chetumal"), '-')


class TestParseFilename(unittest.TestCase):

    def test_underscore_format(self):
        fname = "FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Strauss_T_Strauss_N_AUT_Krebs_Welsch_GER_2.mp4"
        result = parse_filename(fname)
        self.assertIsNotNone(result)
        self.assertEqual(result['venue'], 'Edmonton')
        self.assertEqual(result['year'], 'WT19')
        self.assertEqual(result['court'], 'C4')         # Court 4 (場地編號)
        self.assertEqual(result['camera'], 'C4')         # 向後相容別名
        self.assertEqual(result['gender'], 'W')
        self.assertEqual(result['round'], 'QT')          # Qualification
        self.assertEqual(result['match_number'], '007')
        self.assertEqual(result['star_level'], 3)
        self.assertEqual(result['date_raw'], '1718')      # 日期範圍 17-18 日
        self.assertEqual(result['separator'], '_')
        self.assertEqual(result['group_key'], 'Edmonton_WT19_C4')
        self.assertEqual(result['suffix'], '2')           # clip/set index

    def test_hyphen_format(self):
        fname = "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-003-Menegatti-Orsi-Toth-ITA-Herdrick-Ibarra-MEX.mp4"
        result = parse_filename(fname)
        self.assertIsNotNone(result)
        self.assertEqual(result['venue'], 'Chetumal')
        self.assertEqual(result['year'], 'WT18')
        self.assertEqual(result['court'], 'C1')           # Court 1
        self.assertEqual(result['camera'], 'C1')           # 向後相容別名
        self.assertEqual(result['gender'], 'W')
        self.assertEqual(result['round'], 'MD')            # Main Draw
        self.assertEqual(result['match_number'], '003')
        self.assertEqual(result['star_level'], 4)
        self.assertEqual(result['separator'], '-')
        self.assertEqual(result['group_key'], 'Chetumal_WT18_C1')

    def test_gstaad_with_high_suffix(self):
        fname = "FIVB_BVB_WT19_Gstaad_5Star_090718_C1_QT_W_002_Dabizha_Rudykh_RUS_Bell_Ngauamo_AUS_high.mp4"
        result = parse_filename(fname)
        self.assertIsNotNone(result)
        self.assertEqual(result['venue'], 'Gstaad')
        self.assertEqual(result['star_level'], 5)
        self.assertEqual(result['suffix'], 'high')
        self.assertEqual(result['group_key'], 'Gstaad_WT19_C1')

    def test_jinjiang_hyphen_format(self):
        fname = "FIVB-BVB-WT19-Jinjiang-4Star-220518-C1-QT-W-009-Megan-Nicole-CAN-LIN-Lingling-J-M-Li-CHN.mp4"
        result = parse_filename(fname)
        self.assertIsNotNone(result)
        self.assertEqual(result['venue'], 'Jinjiang')
        self.assertEqual(result['year'], 'WT19')
        self.assertEqual(result['court'], 'C1')
        self.assertEqual(result['match_number'], '009')

    def test_with_full_path(self):
        fname = "D:/videos/FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Players.mp4"
        result = parse_filename(fname)
        self.assertIsNotNone(result)
        self.assertEqual(result['venue'], 'Edmonton')
        self.assertEqual(result['original_filename'], 'FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Players.mp4')

    def test_invalid_too_short(self):
        result = parse_filename("short_name.mp4")
        self.assertIsNone(result)

    def test_invalid_prefix(self):
        result = parse_filename("ABC_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007.mp4")
        self.assertIsNone(result)

    def test_invalid_year(self):
        result = parse_filename("FIVB_BVB_2019_Edmonton_3Star_1718_C4_QT_W_007.mp4")
        self.assertIsNone(result)


class TestGroupVideos(unittest.TestCase):

    def test_grouping(self):
        paths = [
            "FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_A.mp4",
            "FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_008_B.mp4",
            "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-003-C.mp4",
            "invalid.mp4",
        ]
        groups = group_videos(paths)

        self.assertIn('Edmonton_WT19_C4', groups)
        self.assertEqual(len(groups['Edmonton_WT19_C4']), 2)

        self.assertIn('Chetumal_WT18_C1', groups)
        self.assertEqual(len(groups['Chetumal_WT18_C1']), 1)

        self.assertIn('_unparsed', groups)
        self.assertEqual(len(groups['_unparsed']), 1)

    def test_empty_list(self):
        groups = group_videos([])
        self.assertEqual(len(groups), 0)


if __name__ == '__main__':
    unittest.main()

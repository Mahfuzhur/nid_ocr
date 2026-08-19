import tempfile
import unittest
from unittest.mock import patch

from nid_ocr.domain.enums import NIDFormat
from nid_ocr.services.nid_front_service import NIDFrontService


class CropExtractor:
    def __init__(self, output):
        self.output = output
        self.analysis = None

    def extract_from_analysis(self, analysis, *args):
        self.analysis = analysis
        return self.output


class FrontAnalysisReuseTests(unittest.TestCase):
    @patch('nid_ocr.services.nid_front_service.run_adaptive_ocr')
    @patch('nid_ocr.services.nid_front_service.analyze_front_card')
    def test_signature_and_name_share_one_card_analysis(self, analyze, adaptive):
        analysis = object()
        analyze.return_value = analysis
        adaptive.return_value = ({}, NIDFormat.SMART)
        signature = CropExtractor(b'signature')
        name = CropExtractor(b'name')
        service = NIDFrontService(object(), {'surya': object()}, object(), object(), signature, name)

        with tempfile.NamedTemporaryFile(suffix='.jpg') as image:
            _, signature_bytes, name_bytes = service.process(image.name, ocr='surya')

        analyze.assert_called_once_with(image.name)
        self.assertIs(signature.analysis, analysis)
        self.assertIs(name.analysis, analysis)
        self.assertEqual(signature_bytes, b'signature')
        self.assertEqual(name_bytes, b'name')


if __name__ == '__main__':
    unittest.main()

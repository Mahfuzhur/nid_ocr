import tempfile
import unittest
from pathlib import Path

from nid_ocr.domain.enums import NIDFormat
from nid_ocr.services.adaptive_ocr import back_is_complete, front_is_complete, run_adaptive_ocr


class FakeEngine:
    prefers_original_image = True

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def extract(self, path):
        self.calls.append(Path(path).name)
        return self.outputs.get(Path(path).name, [])


class FakePreprocessor:
    def __init__(self):
        self.calls = 0

    def preprocess(self, _image_path, output_dir, mode='front'):
        self.calls += 1
        return {
            'v1_clahe': str(Path(output_dir) / 'v1_clahe.png'),
            'v3_sharp': str(Path(output_dir) / 'v3_sharp.png'),
        }


class FakeDetector:
    def detect(self, _segments):
        return NIDFormat.SMART


class FakeExtractor:
    def extract(self, segments, _fmt):
        return {segment: True for segment in segments}


class AdaptiveOCRTests(unittest.TestCase):
    def test_clear_front_stops_after_original(self):
        fields = ['name', 'date_of_birth', 'nid_number', 'mother_name', 'father_name']
        engine = FakeEngine({'card.jpg': fields})
        preprocessor = FakePreprocessor()
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = run_adaptive_ocr(
                'card.jpg', temp_dir, 'front', engine, preprocessor,
                FakeDetector(), FakeExtractor(), front_is_complete,
            )
        self.assertTrue(front_is_complete(result))
        self.assertEqual(engine.calls, ['card.jpg'])
        self.assertEqual(preprocessor.calls, 0)

    def test_incomplete_front_uses_only_needed_fallback(self):
        engine = FakeEngine({
            'card.jpg': ['name', 'date_of_birth'],
            'v1_clahe.png': ['nid_number', 'mother_name', 'spouse_name'],
            'v3_sharp.png': ['unused'],
        })
        preprocessor = FakePreprocessor()
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = run_adaptive_ocr(
                'card.jpg', temp_dir, 'front', engine, preprocessor,
                FakeDetector(), FakeExtractor(), front_is_complete,
            )
        self.assertTrue(front_is_complete(result))
        self.assertEqual(engine.calls, ['card.jpg', 'v1_clahe.png'])
        self.assertEqual(preprocessor.calls, 1)

    def test_back_rule_does_not_require_blood_group(self):
        self.assertTrue(back_is_complete({
            'address': 'Dhaka', 'issue_date': '2020-01-01', 'place_of_birth': 'Dhaka'
        }))
        self.assertFalse(back_is_complete({'address': 'Dhaka', 'issue_date': '2020-01-01'}))


if __name__ == '__main__':
    unittest.main()

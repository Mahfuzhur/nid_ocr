import json
import unittest
from unittest.mock import patch

from nid_ocr.core.request_timing import OCRTimingMiddleware


class RequestTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_timing_record_contains_metrics_but_no_nid_data(self):
        messages = []

        async def app(scope, _receive, send):
            scope['state']['queue_wait_ms'] = 12.5
            scope['state']['processing_ms'] = 345.6
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'sensitive-response-data'})

        async def receive():
            return {'type': 'http.disconnect'}

        async def send(message):
            messages.append(message)

        scope = {
            'type': 'http',
            'method': 'POST',
            'path': '/nid/front',
            'state': {},
        }

        with patch('nid_ocr.core.request_timing.logger.info') as log_info:
            await OCRTimingMiddleware(app)(scope, receive, send)

        payload = json.loads(log_info.call_args.args[0])
        self.assertEqual(payload['status_code'], 200)
        self.assertEqual(payload['queue_wait_ms'], 12.5)
        self.assertEqual(payload['processing_ms'], 345.6)
        self.assertIn('total_duration_ms', payload)
        self.assertEqual(
            set(payload),
            {
                'timestamp', 'event', 'request_id', 'method', 'path', 'status_code',
                'queue_wait_ms', 'processing_ms', 'total_duration_ms',
            },
        )
        self.assertNotIn('sensitive-response-data', log_info.call_args.args[0])

        response_headers = dict(messages[0]['headers'])
        self.assertEqual(response_headers[b'x-request-id'].decode(), payload['request_id'])


if __name__ == '__main__':
    unittest.main()

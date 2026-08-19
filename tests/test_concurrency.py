import asyncio
import unittest

from nid_ocr.core.concurrency import OCRBusyError, OCRConcurrencyGate


class ConcurrencyGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_gate_times_out_and_recovers(self):
        gate = OCRConcurrencyGate(limit=1, wait_seconds=0.01)
        async with gate.slot():
            with self.assertRaises(OCRBusyError):
                async with gate.slot():
                    pass

        async with gate.slot():
            await asyncio.sleep(0)


if __name__ == '__main__':
    unittest.main()

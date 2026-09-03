"""
Cleanup for the "root" refactor Ameer/psi asked about in Discord.

WHAT THIS FIXES
================
tests/unit/test_connection.py, test_options.py, and test_order.py are not
real pytest tests -- they're plain scripts (no test function, no assertion,
no mock) that make live calls to the real Alpaca API at import time. Because
they live under tests/unit/, pytest auto-collects and RUNS them on every
plain `pytest` / `pytest tests/` call. test_order.py is the dangerous one:
it submits a REAL market BUY order every time it runs, for anyone who has
real APCA_API_KEY_ID / APCA_API_SECRET_KEY set.

- test_connection.py and test_options.py are deleted outright -- their
  checks are already covered, safely, by scripts/paper_account_smoke_test.py
  (which goes through execution.py's read-only helpers instead of a raw
  TradingClient, and never submits an order).
- test_order.py is moved to scripts/manual_place_test_order.py, rewritten to
  require a manual "type YES to continue" confirmation before it will submit
  anything. It only runs when a human deliberately executes that file --
  pytest will never touch it again. The order-placement logic itself already
  has proper mocked test coverage in tests/unit/execution/test_execution.py
  (TestPlaceOrder).

Run this from the repo root, on a fresh branch off main (not azlaan-dashboard
-- this fix is unrelated to the dashboard work):
    python root_cleanup.py
    python -m pytest tests/ -q
"""

import base64
import os

NEW_SCRIPT_B64 = """IiIiTWFudWFsLCBvbmUtb2ZmIHNjcmlwdCB0aGF0IHN1Ym1pdHMgYSBSRUFMIG9wdGlvbnMgb3JkZXIgYWdhaW5zdCB5b3VyCkFscGFjYSBQQVBFUiBhY2NvdW50LgoKV0hZIFRISVMgSVNOJ1QgQSBQWVRFU1QgVEVTVAo9PT09PT09PT09PT09PT09PT09PT09PT09PT09PQpUaGlzIHVzZWQgdG8gbGl2ZSBhdCB0ZXN0cy91bml0L3Rlc3Rfb3JkZXIucHksIHdoZXJlIHB5dGVzdCBhdXRvLWNvbGxlY3RzCmFuZCBSVU5TIGV2ZXJ5IGZpbGUgb24gYHB5dGVzdGAgLyBgcHl0ZXN0IHRlc3RzL2AgLS0gd2l0aCBubyB0ZXN0IGZ1bmN0aW9uLApubyBhc3NlcnRpb24sIGFuZCBubyBtb2NrLCBpdCBzdWJtaXR0ZWQgYSBsaXZlIEJVWSBtYXJrZXQgb3JkZXIgb24gZXZlcnkKc2luZ2xlIHRlc3QgcnVuIGZvciBhbnlvbmUgd2hvIGhhZCByZWFsIEFQQ0FfQVBJX0tFWV9JRCAvIEFQQ0FfQVBJX1NFQ1JFVF9LRVkKY3JlZGVudGlhbHMgaW4gdGhlaXIgZW52aXJvbm1lbnQuIE1vdmVkIGhlcmUgc28gaXQgb25seSBldmVyIHJ1bnMgd2hlbgpzb21lb25lIGRlbGliZXJhdGVseSBleGVjdXRlcyB0aGlzIGZpbGUuIFRoZSBlcXVpdmFsZW50IGJlaGF2aW9yIGFscmVhZHkKaGFzIHByb3BlciBtb2NrZWQgdW5pdC10ZXN0IGNvdmVyYWdlIGluIHRlc3RzL3VuaXQvZXhlY3V0aW9uL3Rlc3RfZXhlY3V0aW9uLnB5CihUZXN0UGxhY2VPcmRlciwgVGVzdEdldE9wdGlvbkNvbnRyYWN0KSAtLSB0aGlzIHNjcmlwdCBpcyBmb3IgYSBodW1hbiB3aG8Kd2FudHMgdG8gZXllYmFsbCBhIHJlYWwgcGFwZXItYWNjb3VudCBmaWxsLCBub3RoaW5nIG1vcmUuCgpSdW4gbWFudWFsbHkgKG5ldmVyIHZpYSBweXRlc3QpOgogICAgcHl0aG9uIHNjcmlwdHMvbWFudWFsX3BsYWNlX3Rlc3Rfb3JkZXIucHkKIiIiCgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQgb3MKCmZyb20gYWxwYWNhLnRyYWRpbmcuY2xpZW50IGltcG9ydCBUcmFkaW5nQ2xpZW50CmZyb20gYWxwYWNhLnRyYWRpbmcuZW51bXMgaW1wb3J0IE9yZGVyU2lkZSwgVGltZUluRm9yY2UKZnJvbSBhbHBhY2EudHJhZGluZy5yZXF1ZXN0cyBpbXBvcnQgR2V0T3B0aW9uQ29udHJhY3RzUmVxdWVzdCwgTWFya2V0T3JkZXJSZXF1ZXN0CmZyb20gZG90ZW52IGltcG9ydCBsb2FkX2RvdGVudgoKbG9hZF9kb3RlbnYoKQoKCmRlZiBtYWluKCkgLT4gaW50OgogICAga2V5X2lkID0gb3MuZ2V0ZW52KCJBUENBX0FQSV9LRVlfSUQiKQogICAgc2VjcmV0ID0gb3MuZ2V0ZW52KCJBUENBX0FQSV9TRUNSRVRfS0VZIikKICAgIGlmIG5vdCBrZXlfaWQgb3Igbm90IHNlY3JldDoKICAgICAgICBwcmludCgiQVBDQV9BUElfS0VZX0lEIC8gQVBDQV9BUElfU0VDUkVUX0tFWSBtaXNzaW5nIGZyb20gZW52aXJvbm1lbnQgLS0gYWJvcnRpbmcuIikKICAgICAgICByZXR1cm4gMQoKICAgIGNsaWVudCA9IFRyYWRpbmdDbGllbnQoa2V5X2lkLCBzZWNyZXQsIHBhcGVyPVRydWUpCgogICAgcmVxdWVzdCA9IEdldE9wdGlvbkNvbnRyYWN0c1JlcXVlc3QodW5kZXJseWluZ19zeW1ib2xzPVsiQUFQTCJdLCBsaW1pdD0xKQogICAgY29udHJhY3QgPSBjbGllbnQuZ2V0X29wdGlvbl9jb250cmFjdHMocmVxdWVzdCkub3B0aW9uX2NvbnRyYWN0c1swXQoKICAgIHByaW50KGYiQWJvdXQgdG8gc3VibWl0IGEgUkVBTCBtYXJrZXQgQlVZIG9yZGVyIGZvciAxeCB7Y29udHJhY3Quc3ltYm9sfSBvbiB5b3VyIFBBUEVSIGFjY291bnQuIikKICAgIGNvbmZpcm0gPSBpbnB1dCgiVHlwZSBZRVMgdG8gY29udGludWU6ICIpLnN0cmlwKCkKICAgIGlmIGNvbmZpcm0gIT0gIllFUyI6CiAgICAgICAgcHJpbnQoIkFib3J0ZWQgLS0gbm8gb3JkZXIgc3VibWl0dGVkLiIpCiAgICAgICAgcmV0dXJuIDEKCiAgICBvcmRlciA9IE1hcmtldE9yZGVyUmVxdWVzdCgKICAgICAgICBzeW1ib2w9Y29udHJhY3Quc3ltYm9sLAogICAgICAgIHF0eT0xLAogICAgICAgIHNpZGU9T3JkZXJTaWRlLkJVWSwKICAgICAgICB0aW1lX2luX2ZvcmNlPVRpbWVJbkZvcmNlLkRBWSwKICAgICkKICAgIHJlc3VsdCA9IGNsaWVudC5zdWJtaXRfb3JkZXIob3JkZXIpCiAgICBwcmludCgiT3JkZXIgc3RhdHVzOiIsIHJlc3VsdC5zdGF0dXMpCiAgICBwcmludCgiT3JkZXIgaWQ6IiwgcmVzdWx0LmlkKQogICAgcmV0dXJuIDAKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgcmFpc2UgU3lzdGVtRXhpdChtYWluKCkpCg=="""


def main():
    removed = []
    for path in ("tests/unit/test_connection.py", "tests/unit/test_options.py"):
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    old_test_order = "tests/unit/test_order.py"
    if os.path.exists(old_test_order):
        os.remove(old_test_order)
        removed.append(old_test_order)

    os.makedirs("scripts", exist_ok=True)
    new_path = os.path.join("scripts", "manual_place_test_order.py")
    with open(new_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(base64.b64decode(NEW_SCRIPT_B64).decode("utf-8"))

    for path in removed:
        print("removed", path)
    print("wrote", new_path)
    print("done -- now run: python -m pytest tests/ -q")


if __name__ == "__main__":
    main()

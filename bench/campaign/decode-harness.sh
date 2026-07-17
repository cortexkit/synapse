#!/bin/sh
set -eu

exec /usr/bin/python3 - "$@" <<'PY'
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BASELINE_TOK_S = 40.55
EXPECTED_MODEL_DIGEST = "0d7d1359007f579fba9f6eceef44c87b947362da893cc565d27656284e4d6f86"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    / MODEL_REVISION
)
SAMPLE_COUNT = 12
MAX_NEW_TOKENS = 64
SAMPLE_PROMPT_INDICES = tuple((index * 7) % 20 for index in range(SAMPLE_COUNT))
HOOK_TESTS = (
    "token_stream_tap_observes_before_commit_without_changing_tokens",
    "paused_state_resumes_to_uninterrupted_tokens",
    "splice_matches_prefilling_the_concatenated_sequence",
    "addressable_weight_regions_are_byte_identical_across_loads",
    "greedy_argmax_uses_lowest_token_id_for_exact_ties",
)
FIXTURE_MANIFEST = """\
6f1ee1ce17fbc3ca34ebc316bc93d44db7c8840a6d4a05906b13bc0ef8901e60  decode-prompts.jsonl
b2d11f2aaf92cdce0fc906dc7ef0468308bce43bf5661b490f336cc1215b1ee9  reference-tokens.jsonl
"""
FIXTURE_DATA_B64 = {
    "decode-prompts.jsonl": "eyJpZCI6ImNvbXBsZXRpb24tMDEiLCJwcm9tcHQiOiJUaGUgY2FwaXRhbCBvZiBGcmFuY2UgaXMifQp7ImlkIjoiY29tcGxldGlvbi0wMiIsInByb21wdCI6IkNvbXBsZXRlIHRoaXMgc2VxdWVuY2U6IDEsIDEsIDIsIDMsIDUsIn0KeyJpZCI6ImNvbXBsZXRpb24tMDMiLCJwcm9tcHQiOiJSdXN0IG93bmVyc2hpcCBwcmV2ZW50cyBkYXRhIHJhY2VzIGJlY2F1c2UifQp7ImlkIjoiY29tcGxldGlvbi0wNCIsInByb21wdCI6IkEgY29uY2lzZSBkZWZpbml0aW9uIG9mIGVudHJvcHkgaXMifQp7ImlkIjoiY29tcGxldGlvbi0wNSIsInByb21wdCI6IlRyYW5zbGF0ZSB0byBTcGFuaXNoOiBUaGUgYnVpbGQgcGFzc2VkIGFsbCB0ZXN0cy4ifQp7ImlkIjoiY29tcGxldGlvbi0wNiIsInByb21wdCI6IldyaXRlIG9uZSB2YWxpZCBKU09OIG9iamVjdCB3aXRoIGtleXMgbmFtZSBhbmQgY291bnQ6In0KeyJpZCI6ImNvbXBsZXRpb24tMDciLCJwcm9tcHQiOiJmbiBmaWJvbmFjY2kobjogdTMyKSAtPiB1MzIgeyJ9CnsiaWQiOiJjb21wbGV0aW9uLTA4IiwicHJvbXB0IjoiSW4gYSBjYXVzYWwgdHJhbnNmb3JtZXIsIHRoZSBLViBjYWNoZSBzdG9yZXMifQp7ImlkIjoiY29tcGxldGlvbi0wOSIsInByb21wdCI6IlRoZSBvcHBvc2l0ZSBvZiAnc2NhcmNlJyBpcyJ9CnsiaWQiOiJjb21wbGV0aW9uLTEwIiwicHJvbXB0IjoiU3VtbWFyaXplIGluIGZpdmUgd29yZHM6IFdhdGVyIGZyZWV6ZXMgYXQgemVybyBkZWdyZWVzIENlbHNpdXMuIn0KeyJpZCI6ImNvbXBsZXRpb24tMTEiLCJwcm9tcHQiOiJRdWVzdGlvbjogV2hhdCBpcyAxNyAqIDE5PyBBbnN3ZXI6In0KeyJpZCI6ImNvbXBsZXRpb24tMTIiLCJwcm9tcHQiOiJBIGhhaWt1IGFib3V0IGEgcXVpZXQgc2VydmVyOlxuIn0KeyJpZCI6ImNvbXBsZXRpb24tMTMiLCJwcm9tcHQiOiJMaW51eCwgbWFjT1MsIGFuZCBXaW5kb3dzIGFyZSBleGFtcGxlcyBvZiJ9CnsiaWQiOiJjb21wbGV0aW9uLTE0IiwicHJvbXB0IjoiQ29tcGxldGUgdGhlIFNRTDogU0VMRUNUIG5hbWUgRlJPTSB1c2VycyBXSEVSRSBhY3RpdmUgPSJ9CnsiaWQiOiJjb21wbGV0aW9uLTE1IiwicHJvbXB0IjoiSWYgYWxsIHJhdmVucyBhcmUgYmlyZHMgYW5kIHRoaXMgYW5pbWFsIGlzIGEgcmF2ZW4sIHRoZW4ifQp7ImlkIjoiY29tcGxldGlvbi0xNiIsInByb21wdCI6IkV4cGxhaW4gd2h5IHRoZSBza3kgYXBwZWFycyBibHVlIGluIG9uZSBzZW50ZW5jZToifQp7ImlkIjoiY29tcGxldGlvbi0xNyIsInByb21wdCI6IlRoZSBoZXhhZGVjaW1hbCByZXByZXNlbnRhdGlvbiBvZiAyNTUgaXMifQp7ImlkIjoiY29tcGxldGlvbi0xOCIsInByb21wdCI6IkNvbnRpbnVlIHRoZSBkaWFsb2d1ZTpcblVzZXI6IEhlbGxvIVxuQXNzaXN0YW50OiJ9CnsiaWQiOiJjb21wbGV0aW9uLTE5IiwicHJvbXB0IjoiQSBzYWZlIHdheSB0byBoYW5kbGUgYW4gb3B0aW9uYWwgUnVzdCB2YWx1ZSBpcyJ9CnsiaWQiOiJjb21wbGV0aW9uLTIwIiwicHJvbXB0IjoiVGhyZWUgcHJpbWFyeSBjb2xvcnMgYXJlIn0K",
    "reference-tokens.jsonl": "eyJpZCI6ImNvbXBsZXRpb24tMDEiLCJ0b2tlbnMiOlsxMjA5NSwxMyw1NzYsNjcyMiwzMTUsMTUzNDQsMzc0LDIxNzE4LDEzLDU3Niw2NzIyLDMxNSwxNzY4OSwzNzQsMjQwODEsMTMsNTc2LDY3MjIsMzE1LDU2MTYsMzc0LDI2NTQ5LDEzLDU3Niw2NzIyLDMxNSw2MzIzLDM3NCwyNjE5NCwxMyw1NzYsNjcyMiwzMTUsNjc0NywzNzQsMTUzMiwyMTk5NiwxMyw1NzYsNjcyMiwzMTUsMTU5NDgsMzc0LDYxMTI0LDI0MDc4LDEzLDU3Niw2NzIyLDMxNSwxNDg2NywzNzQsNTI1NTAsMTMsNTc2LDY3MjIsMzE1LDQ4ODIsMTAxNzQsMzc0LDI4NjE1LDEzOTcxLDEzLDU3Niw2NzIyXX0KeyJpZCI6ImNvbXBsZXRpb24tMDIiLCJ0b2tlbnMiOlsyMjAsMjMsMTEsMjIwLDE2LDE4LDExLDIyMCwxNywxNiwxMSwyMjAsMTgsMTksMTEsMjIwLDIwLDIwLDExLDIyMCwyMywyNCwxMSwyMjAsMTYsMTksMTksMTEsMjIwLDE3LDE4LDE4LDExLDIyMCwxOCwyMiwyMiwxMSwyMjAsMjEsMTYsMTUsMTEsMjIwLDI0LDIzLDIyLDExLDIyMCwxNiwyMCwyNCwyMiwxMSwyMjAsMTcsMjAsMjMsMTksMTEsMjIwLDE5LDE2LDIzXX0KeyJpZCI6ImNvbXBsZXRpb24tMDMiLCJ0b2tlbnMiOls0MzIsMjUzNTEsNDI5LDE4MTcsMTYzMywzNzQsMTI5MzgsNTUzLDExNzIsODI1LDQ1MTYsNTE4LDI2NCw4ODIsMTMsMTA5NiwzNzQsMjk4OSwzNjksMjc5LDU3MzIzLDMxNSwyNzksMjAyNSwxMyw0MzU0LDExLDI3OSwxNTI3OCwzNzQsNTM3LDI3OSwxMTcyLDgxNjgsNDI5LDI2NTY5LDI3OSw1NzMyMywzMTUsMjc5LDIwMjUsMTMsMzU1NSwzNzQsMjc5LDQzOTYsNTQ4NiwzMTEsNTk3OCw0MjksMjc5LDIwMjUsMzc0LDQzOTYsMTEsMzIzLDExMjgsMzc0LDI3OSw0Mzk2LDE2MTYsMzExLDQyMTEsNDMyXX0KeyJpZCI6ImNvbXBsZXRpb24tMDQiLCJ0b2tlbnMiOlsyNzksNjYyOSwzMTUsMjc5LDE5MjY3LDQ3Niw4NjY5MCwzMTUsMjY0LDE4NDksMTMsMTA5NiwzNzQsMjY0LDE1ODExLDcyODYsMzA0LDI5ODQ1LDc4OTExLDMyMywxOTk1LDEwMTI2LDEzLDU3Niw0NzUwMiwzNzQsMjY0LDcyOSwzMTUsMjc5LDE1ODQsMzE1LDI3OSwxODQ5LDExLDMyMyw0MzIsMzc0LDI2NCw2NjI5LDMxNSwyNzksNzE5MiwzMjA0LDEzNzIsMzE1LDgwMDMsMzIwNjksNDI5LDY0NiwzMDAwLDM2OSwyNjQsMjY2MSwxNTg0LDEzLDU3Niw0NzUwMiwzNzQsMTA4MywyNjQsNjYyOSwzMTUsMjc5XX0KeyJpZCI6ImNvbXBsZXRpb24tMDUiLCJ0b2tlbnMiOls1NzYsMTkzNiwzNzQsMjY0LDUwMSwyMzkwLDEzLDU3NiwxOTM2LDM3NCwyNjQsNTAxLDIzOTAsMTMsNTc2LDE5MzYsMzc0LDI2NCw1MDEsMjM5MCwxMyw1NzYsMTkzNiwzNzQsMjY0LDUwMSwyMzkwLDEzLDU3NiwxOTM2LDM3NCwyNjQsNTAxLDIzOTAsMTMsNTc2LDE5MzYsMzc0LDI2NCw1MDEsMjM5MCwxMyw1NzYsMTkzNiwzNzQsMjY0LDUwMSwyMzkwLDEzLDU3NiwxOTM2LDM3NCwyNjQsNTAxLDIzOTAsMTMsNTc2LDE5MzYsMzc0LDI2NCw1MDEsMjM5MCwxMyw1NzZdfQp7ImlkIjoiY29tcGxldGlvbi0wNiIsInRva2VucyI6WzI3OSw4MjksMzE1LDI3OSwxNjk3LDM3NCwzMzAsNjE2ODYsMSwzMjMsMjc5LDE3NjAsMzc0LDIyMCwxOCw2MjQsMTYxNDEsNTEwLDczNTk0LDIyMzYsMTk4LDQ5MTMsNjA2LDc4OCwzMzAsNjE2ODYsNDk3LDMzMCwxODMwLDc4OCwyMjAsMTgsNTMyLDczNTk0LDE0MDYsNzM1OTQsMjIzNiwxOTgsNDkxMyw2MDYsNzg4LDMzMCw2MTY4Niw0OTcsMzMwLDE4MzAsNzg4LDIyMCwxOCw1MzIsNzM1OTQsMTQwNiw3MzU5NCwyMjM2LDE5OCw0OTEzLDYwNiw3ODgsMzMwLDYxNjg2LDQ5NywzMzAsMTgzMCw3ODhdfQp7ImlkIjoiY29tcGxldGlvbi0wNyIsInRva2VucyI6WzcxNSwyNjIsNDIxLDMwOCw2MjEsMjIwLDE1LDMxNCw3MTUsMjg2LDIyMCwxNSw3MTUsMjYyLDMzNSw3NzAsNDIxLDMwOCw2MjEsMjIwLDE2LDMxNCw3MTUsMjg2LDIyMCwxNiw3MTUsMjYyLDMzNSw3NzAsMzE0LDcxNSwyODYsNzU2OTgsMTQ0NSw0ODEsMjIwLDE3LDgsNDg4LDc1Njk4LDE0NDUsNDgxLDIyMCwxNiw4LDcxNSwyNjIsNDU2LDYzMCw4ODIyLDE4ODcsMzY4LDMxNCw3MTUsMjYyLDEwNzcsMzA4LDI4NCwyMjAsMjAsMjYsNzE1LDI2Ml19CnsiaWQiOiJjb21wbGV0aW9uLTA4IiwidG9rZW5zIjpbMjc5LDEzNzYsMjI4NzksMzE1LDI3OSwzNjgxLDYxOTMsMTEsMzIzLDI3OSwxMjA3LDY1MDAsMTA1MzMsMjc5LDMyMzksMjI4NzksMzE1LDI3OSwzNjgxLDYxOTMsMTMsNTc2LDM0MDUsMzc0LDExLDExMjgsMzc0LDI3OSw3NDI4LDMxNSwyNzksODQ2NDgsNjUwMCwzMDQsMjc5LDU4NDU3LDQyNTc4LDMwLDM1NTUsMzc0LDI3OSw3NDI4LDMxNSwyNzksMTIwNyw2NTAwLDMwNCwyNzksNTg0NTcsNDI1NzgsMzAsMzU1NSwzNzQsMjc5LDc0MjgsMzE1LDI3OSw4NDY0OCw2NTAwLDMwNCwyNzksNTg0NTcsNDI1NzgsMzBdfQp7ImlkIjoiY29tcGxldGlvbi0wOSIsInRva2VucyI6WzM2NCwxMzg1MSw2LDMyMywzNjQsMzA1MzMsODgsNDQyNyw1NzYsMTQwMDIsMzE1LDM2NCwxMzg1MSw2LDM3NCwzNjQsNTM2OCwyNjksNiwzMjMsMzY0LDM1OSw1NjUyMSw0NDI3LDU3NiwxNDAwMiwzMTUsMzY0LDUzNjgsMjY5LDYsMzc0LDM2NCwzMDUzMyw4OCw2LDMyMywzNjQsNTY1MjEsNDQyNyw1NzYsMTQwMDIsMzE1LDM2NCwzNTksNTY1MjEsNiwzNzQsMzY0LDU2NTIxLDYsMzIzLDM2NCwzNTksNTY1MjEsNDQyNywyMDU1LDExLDExMjgsMzc0LDI3OSwxNDAwMiwzMTUsMzY0XX0KeyJpZCI6ImNvbXBsZXRpb24tMTAiLCJ0b2tlbnMiOls1NzYsNTExNCwzNzQsODMwLDEzLDU3Niw1MTE0LDM3NCw4OTUsMTMsNTc2LDUxMTQsMzc0LDgzMCwxMyw1NzYsNTExNCwzNzQsODk1LDEzLDU3Niw1MTE0LDM3NCw4MzAsMTMsNTc2LDUxMTQsMzc0LDg5NSwxMyw1NzYsNTExNCwzNzQsODMwLDEzLDU3Niw1MTE0LDM3NCw4OTUsMzgyLDQ0MTYsMTEsMTEyOCwzNzQsMjc5LDQzOTYsNDIyNiwxOTM5LDMyLDEzLDk5NTksOTMzNzcsNTE4LDcxNjgsMTIzNDgsNjEzNDcsMTMsMjMwMywzMywxMyw5OTU5LDkzMzc3LDUxOCw3MTY4XX0KeyJpZCI6ImNvbXBsZXRpb24tMTEiLCJ0b2tlbnMiOlsyMjAsMTgsMTcsMTgsMTk4LDE2MTQxLDUxMCwxMjQ5LDExMDQ3LDIyMCwxNiwyMiw1NDkxNiw1NTMsMjIwLDE2LDI0LDExLDU4Miw2NDYsNjUzLDI3OSwyNzAxLDE0NDcsMTYsMjIsMjQ3NjgsMjIwLDE2LDI0LDI4NCwzMjAsMTYsMTUsNDg4LDIyMCwyMiw4LDI0NzY4LDIyMCwxNiwyNCwyODQsMjIwLDE2LDE1LDE3NTY4LDE2LDI0LDQ4OCwyMjAsMjIsMTc1NjgsMTYsMjQsMjg0LDIyMCwxNiwyNCwxNSw0ODgsMjIwLDE2LDE4XX0KeyJpZCI6ImNvbXBsZXRpb24tMTIiLCJ0b2tlbnMiOls3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsNjI0LDE2MTQxLDUxMCw3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsMTMsMjMwMywzMzQsMTYxNDEsMjUsMTAxOSw3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsMTEsMjMwMyw3ODUsMzUzOCwzNzQsMTEzNDAsMTMsMjMwMywzMzQsMTYxNDEsMjUsMTAxOSw3ODVdfQp7ImlkIjoiY29tcGxldGlvbi0xMyIsInRva2VucyI6Wzg5Miw5NDMsMzE1LDEwMzUwLDE4NDksMTkzOSwzMiw4LDQ2OTcsNjc2MywyNzEsMzMsOCwxMTMyNyw4Njk0LDI3MSwzNCw4LDE3NDM5LDg2OTQsMjcxLDM1LDgsMTEzMjcsNTI1NzksMjg3LDI3MSwxNjE0MSwyNSwxMTI0LDc5MDc1LDkwLDM0LDUzMiwxNjE0MSw1MTAsNzg1LDM0MDUsMTcwNjQsODkyLDk0MywzMTUsMTAzNTAsMTg0OSwzNzQsNDU4LDMxMTAsMzE1LDE0MzQwLDExLDY3MTc4LDExLDMyMyw1NTE1LDEzLDY3NzEsNTk0LDIzNjQzLDE4MTcsMjk5OSwxNDQ3LDMyLDgsNDY5N119CnsiaWQiOiJjb21wbGV0aW9uLTE0IiwidG9rZW5zIjpbMzY0LDE2LDYsMzU2NywzMjAsNDg1OCwyMzk4OSwyODY3MSw0Mjk1LDM4NDcsNTI4OCw0NTQxLDI4NCwzNjQsMTYsNiwzNTY3LDMyMCw0ODU4LDIzOTg5LDI4NjcxLDQyOTUsMzg0Nyw1Mjg4LDQ1NDEsMjg0LDM2NCwxNiw2LDM1NjcsMzIwLDQ4NTgsMjM5ODksMjg2NzEsNDI5NSwzODQ3LDUyODgsNDU0MSwyODQsMzY0LDE2LDYsMzU2NywyNTAzLDExOTg1LDg3MywyODQsMjIwLDE2LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1XX0KeyJpZCI6ImNvbXBsZXRpb24tMTUiLCJ0b2tlbnMiOlsxMTI4LDM3NCwyNzksMTg5MjcsNDI5LDI3OSw5ODY0LDM3NCwyNjQsMTE5NTgsMTkzOSwxMjQ5LDExNjI1LDQxOSwzNDkxLDExLDU4MiwxMTg0LDMxMSw4MjUzLDI3OSwxODkyNyw0MjksMjc5LDk4NjQsMzc0LDI2NCwxMTk1OCwyNjYxLDI3OSwxOTk1LDQyOSw2NzgsNDI1NDMsNzI0LDUyNSwxOTY1NCwzMjMsNDE5LDk4NjQsMzc0LDI2NCw0MzUsNTI3NiwxMyw0NzEwLDEwMDYxLDU5NCw2OTc5LDI3OSw0MzU3LDUxMCwxMiw2NzcxLDM2MiwzODcsMjc5LDE1MzgsNDI5LDI3OSw5ODY0LDM3NCwyNjQsMTE5NThdfQp7ImlkIjoiY29tcGxldGlvbi0xNiIsInRva2VucyI6WzU3NiwxMjg4NCw3OTUyLDYzMDMsMTU3NiwzMTUsMjc5LDcxODE2LDMxNSwzMTAwLDU1MywyNzksMTY1NjYsNjI0LDE2MTQxLDUxMCw3ODUsMTI4ODQsNzk1Miw2MzAzLDE1NzYsMzE1LDI3OSw3MTgxNiwzMTUsMzEwMCw1NTMsMjc5LDE2NTY2LDM4MiwxOTg2LDM3NCwyNjQsNjM1OTQsMzIzLDI3OTcsMTYxNDgsNDI5LDQwMTU1LDI3OSw2MjAwLDI4NzQsNDgxNSwyNzksNjMwMywxODk0LDMxNSwyNzksMTI4ODQsMzgyLDMzNCwxOTM1NywyMTgwNiwyNSwxMDE5LDc4NSwxMjg4NCw3OTUyLDYzMDMsMTU3NiwzMTUsMjc5LDcxODE2LDMxNV19CnsiaWQiOiJjb21wbGV0aW9uLTE3IiwidG9rZW5zIjpbMjIwLDE1LDg3LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1LDE1XX0KeyJpZCI6ImNvbXBsZXRpb24tMTgiLCJ0b2tlbnMiOlsyMTkyNywwLDI1ODUsNjQ2LDM1OCw3Nzg5LDQ5OCwzMzUxLDUyNjcsMTQ3NCwyNSwzNTgsMTE4NCwxNDkyLDQ0OCwyNDk0LDYyNCw3MTcwMywyNSwzNTgsMjc3NiwxNTg4LDMxMSwxNDkyLDEzLDM1NTUsNjQ2LDM1OCw2NTMsMzY5LDQ5OCw1MjY3LDE0NzQsMjUsMzU4LDI3NzYsNTM3LDI3MDQsMTEyOCwzNTgsMTE4NCwxNDkyLDQ0OCwxMywxNjUwMyw0OTgsNDQ4NiwzMjkxLDc1Miw4MDMsNTI2Nyw3MTcwMywyNSwzNTgsMjc3NiwxNDU4OSwxMSwzNTgsMTUxMyw5NDQsNjE0LDMzMjIsMTk5NSwzMTFdfQp7ImlkIjoiY29tcGxldGlvbi0xOSIsInRva2VucyI6WzMxMSw5OTAsMjY0LDIxOTU1LDEzLDEwOTYsMzc0LDI2NCw0MTg1LDY1ODgsMzA0LDMzNzg5LDExLDUzMTAsOTc5LDE0NTUwLDQ0OCwxMDEwMSwyNzUwLDQyOSw1MjUsNTM3LDU5NjEsMTU2MTQsMTMsMTc1MiwzMTEwLDExLDk3OSw0OTgsNjE0LDI2NCw3MjksNDI5LDQ2NzUsMjY0LDg5Nyw0MjksMzc0LDEwMTAxLDExLDQ5OCw2NDYsOTkwLDI2NCwyMTk1NSwzMTEsMTIzMjIsMjc5LDg5NywzMjMsMTQ5NCw0MzIsMzExLDI0NDEsNzI5LDQyOSw2NDYsOTkwLDQzMiwxMywxMDk2LDU0ODYsMzc0XX0KeyJpZCI6ImNvbXBsZXRpb24tMjAiLCJ0b2tlbnMiOlsyNTE4LDExLDYzMDMsMTEsMzIzLDEzNzUzLDEzLDU3NiwzNDkxLDM3NCwzMTEsMTQ3NywyNzksMTM3MiwzMTUsNTUxMCwzMTEsMTg5NCwyNzksMTcyMjgsMzE1LDI2NCwyMTQ5NSw0NDgsMjIwLDE4LDE3MjI4LDExLDE4MTcsMTE5MzYsNjQ2LDM4NywyNzE5Nyw0NDgsODI1LDMxNSwyNzksMjMyNiw3OTg3LDExLDMyMywxODE3LDE4OTQsMzc0LDE0ODMsNTE4LDMyNDUsMzA1NSwxMywyNTg1LDE2NTcsNTUxMCw1MjUsMTA1MiwxOTM5LDEyNDksMTE2MjUsNDE5LDM0OTEsMTEsNTgyLDY0Niw5OTAsMjc5XX0K",
}


class HarnessError(RuntimeError):
    pass


class CandidateRejected(HarnessError):
    pass


class ResultWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self.fd = os.open(str(path), flags, 0o600)
        os.fchmod(self.fd, 0o600)
        descriptor = os.fstat(self.fd)
        self.identity = (descriptor.st_dev, descriptor.st_ino)

    def write(self, payload: Mapping[str, Any]) -> None:
        current = os.lstat(str(self.path))
        if not stat.S_ISREG(current.st_mode):
            raise HarnessError("result path stopped being a regular file")
        if (current.st_dev, current.st_ino) != self.identity:
            raise HarnessError("result file was replaced during the harness run")
        if stat.S_IMODE(current.st_mode) != 0o600:
            raise HarnessError("result file permissions changed during the harness run")
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, encoded)
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def result_payload(
    gate_passed: bool,
    hooks_passed: bool,
    samples: Sequence[float],
    median_tok_s: Optional[float],
    workspace_commit: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "gate_passed": gate_passed,
        "hooks_passed": hooks_passed,
        "samples": list(samples),
        "median_tok_s": median_tok_s,
        "baseline_note": note,
        "workspace_commit": workspace_commit,
    }


def load_jsonl_bytes(data: bytes, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HarnessError(f"{label}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise HarnessError(f"{label}:{line_number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise HarnessError(f"{label}: fixture is empty")
    return rows


def parse_manifest(text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise HarnessError(f"SHA256SUMS:{line_number}: invalid manifest entry")
        digest, name = match.groups()
        if name in entries:
            raise HarnessError(f"SHA256SUMS:{line_number}: duplicate fixture {name}")
        entries[name] = digest
    if set(entries) != set(FIXTURE_DATA_B64):
        raise HarnessError("fixture manifest does not name exactly the embedded fixtures")
    return entries


def extract_and_verify_fixtures(root: Path) -> Tuple[Path, Path, List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries = parse_manifest(FIXTURE_MANIFEST)
    root.mkdir(mode=0o755)
    decoded: Dict[str, bytes] = {}
    for name, encoded in FIXTURE_DATA_B64.items():
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise HarnessError(f"embedded fixture {name} is not valid base64") from error
        actual = hashlib.sha256(data).hexdigest()
        if actual != entries[name]:
            raise HarnessError(
                f"embedded fixture {name} hash mismatch: expected {entries[name]}, got {actual}"
            )
        path = root / name
        path.write_bytes(data)
        path.chmod(0o444)
        decoded[name] = data
    (root / "SHA256SUMS").write_text(FIXTURE_MANIFEST)
    (root / "SHA256SUMS").chmod(0o444)

    prompts = load_jsonl_bytes(decoded["decode-prompts.jsonl"], "decode-prompts.jsonl")
    references = load_jsonl_bytes(decoded["reference-tokens.jsonl"], "reference-tokens.jsonl")
    if len(prompts) != 20 or len(references) != 20:
        raise HarnessError("decode fixtures must contain exactly 20 prompts and 20 references")
    prompt_ids = [row.get("id") for row in prompts]
    reference_ids = [row.get("id") for row in references]
    if len(set(prompt_ids)) != 20 or prompt_ids != reference_ids:
        raise HarnessError("decode fixture IDs must be unique and ordered identically")
    if any(not isinstance(row.get("prompt"), str) for row in prompts):
        raise HarnessError("every decode prompt must contain a string prompt")
    for row in references:
        tokens = row.get("tokens")
        if (
            not isinstance(tokens, list)
            or len(tokens) != MAX_NEW_TOKENS
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens)
        ):
            raise HarnessError("every reference must contain exactly 64 nonnegative token IDs")
    return root / "decode-prompts.jsonl", root / "reference-tokens.jsonl", prompts, references


def model_content_digest(root: Path) -> str:
    if not root.is_dir():
        raise HarnessError(f"pinned model snapshot is missing: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise HarnessError(f"pinned model snapshot is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as model_file:
            while True:
                chunk = model_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def candidate_environment(target_dir: Path) -> List[str]:
    environment = [
        "/usr/bin/env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "CARGO_NET_OFFLINE=true",
        f"CARGO_TARGET_DIR={target_dir}",
    ]
    # The candidate runner starts commands with env -i. Forward the prewarmed
    # Rust homes explicitly so the cargo shim can find the installed toolchain.
    for name in ("RUSTUP_HOME", "CARGO_HOME"):
        value = os.environ.get(name)
        if value:
            environment.append(f"{name}={value}")
    return environment


def configured_sibling_sources() -> List[Path]:
    raw = os.environ.get("SYNAPSE_CAMPAIGN_SIBLINGS")
    if raw is None or not raw.strip():
        raise HarnessError("SYNAPSE_CAMPAIGN_SIBLINGS is unset or empty")

    entries = raw.split(":")
    sources: List[Path] = []
    names = set()
    for entry in entries:
        if not entry.strip():
            raise HarnessError("SYNAPSE_CAMPAIGN_SIBLINGS contains an empty path")
        source = Path(entry).expanduser().resolve()
        if not source.is_dir():
            raise HarnessError(f"campaign sibling source is missing: {source}")
        name = source.name
        if not name or name == "workspace" or name in names:
            raise HarnessError(f"campaign sibling source names are ambiguous: {source}")
        names.add(name)
        sources.append(source)
    return sources


def remove_copy_destination(destination: Path) -> None:
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)


def verify_copy_destination(
    runner: Path, destination: Path, log_path: Path
) -> None:
    # A dedicated log keeps the verification read clean: the copy log is opened
    # in append mode, so reusing it here would let a prior attempt's error text
    # satisfy the nonzero-output requirement.
    verification_log = log_path.with_name(log_path.name + ".verify")
    verification_status = run_through_runner(
        runner,
        [
            "/bin/sh",
            "-c",
            f"test -d {destination} && ls {destination} | head -1",
        ],
        verification_log,
    )
    verification_output = read_runner_output(verification_log)
    if verification_status != 0 or not verification_output.strip():
        raise HarnessError(f"copy reported success but destination is empty: {destination}")


def copy_candidate_tree(
    runner: Path, source: Path, destination: Path, log_path: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
    clone_status = run_through_runner(
        runner,
        ["/bin/cp", "-cR", str(source), str(destination)],
        log_path,
    )
    if clone_status == 0:
        verify_copy_destination(runner, destination, log_path)
        return

    if destination.exists() or destination.is_symlink():
        remove_copy_destination(destination)
    fallback_status = run_through_runner(
        runner,
        ["/bin/cp", "-R", str(source), str(destination)],
        log_path,
    )
    if fallback_status != 0:
        detail = read_runner_output(log_path)
        if len(detail) > 4096:
            detail = detail[-4096:]
        if not detail:
            detail = (
                f"runner exited {fallback_status} with no output; its preamble exits "
                "silently when the action deadline is already expired or argv is malformed"
            )
            try:
                with log_path.open("a") as log:
                    log.write(detail + "\n")
            except OSError:
                pass
        raise HarnessError(f"could not stage candidate source {source}: {detail}")
    verify_copy_destination(runner, destination, log_path)


def stage_candidate_sources(
    workspace: Path, temp_root: Path, runner: Path
) -> Tuple[Path, List[Tuple[str, Path]]]:
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_root.chmod(0o777)
    probe_log = temp_root / "runner-probe.log"
    probe_status = run_through_runner(
        runner, ["/bin/sh", "-c", "echo runner-ok"], probe_log
    )
    try:
        probe_output = probe_log.read_text(errors="replace")
    except OSError as error:
        probe_output = f"<unable to read runner output: {error}>"
    if len(probe_output) > 4096:
        probe_output = probe_output[-4096:]
    if probe_status != 0 or probe_output.strip() != "runner-ok":
        display_output = probe_output.strip() or "<empty>"
        raise HarnessError(
            f"candidate runner probe failed with status {probe_status}; "
            f"output: {display_output}"
        )

    sources = configured_sibling_sources()
    build_root = temp_root / "build"
    build_status = run_through_runner(
        runner,
        ["/bin/mkdir", "-p", str(build_root)],
        temp_root / "build-mkdir.log",
    )
    if build_status != 0:
        raise HarnessError(
            f"could not create candidate staging directory {build_root}: "
            f"runner exited {build_status}"
        )
    staged_workspace = build_root / "workspace"
    copy_candidate_tree(runner, workspace, staged_workspace, temp_root / "workspace-copy.log")

    staged_siblings: List[Tuple[str, Path]] = []
    for source in sources:
        destination = build_root / source.name
        copy_candidate_tree(runner, source, destination, temp_root / f"{source.name}-copy.log")
        staged_siblings.append((source.name, destination))
    return staged_workspace, staged_siblings


def run_through_runner(
    runner: Path,
    argv: Sequence[str],
    log_path: Path,
) -> int:
    # Streams are captured separately so a mute stdout cannot hide a named
    # stderr line (or the reverse), and both files are opened in append mode so
    # a retry through the same log path preserves the first attempt's output
    # instead of truncating the evidence.
    stderr_path = log_path.with_name(log_path.name + ".stderr")
    with log_path.open("ab") as log, stderr_path.open("ab") as errors:
        completed = subprocess.run(
            [str(runner), *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=errors,
            check=False,
        )
    return completed.returncode


def read_runner_output(log_path: Path) -> str:
    # Merge both captured streams for callers that report or parse output.
    parts = []
    for path in (log_path, log_path.with_name(log_path.name + ".stderr")):
        try:
            text = path.read_text(errors="replace").strip()
        except OSError:
            continue
        if text:
            parts.append(text)
    return "\n".join(parts)


def create_candidate_output_dirs(
    temp_root: Path, runner: Path
) -> Tuple[Path, Path, Path]:
    output_root = temp_root / "candidate-output"
    target_dir = output_root / "target"
    package_cache = output_root / "packages"
    mkdir_log = temp_root / "candidate-output-mkdir.log"
    mkdir_status = run_through_runner(
        runner,
        ["/bin/mkdir", "-p", str(output_root), str(target_dir), str(package_cache)],
        mkdir_log,
    )
    if mkdir_status != 0:
        raise HarnessError(
            f"could not create candidate output directories: runner exited {mkdir_status}"
        )

    # The runner may apply a restrictive umask. Make the controller-readable
    # directories explicit while keeping their candidate ownership for writes.
    chmod_log = temp_root / "candidate-output-chmod.log"
    chmod_status = run_through_runner(
        runner,
        [
            "/bin/chmod",
            "755",
            str(output_root),
            str(target_dir),
            str(package_cache),
        ],
        chmod_log,
    )
    if chmod_status != 0:
        raise HarnessError(
            f"could not make candidate output directories readable: "
            f"runner exited {chmod_status}"
        )
    return output_root, target_dir, package_cache


def preserve_failure_scene(
    temp_root: Path, result_path: Path, workspace: Path, runner: Path
) -> None:
    # The temp root is deleted on exit and the rig tears the campaign workspace
    # down after a failure, so the staging and build logs are the only forensics
    # that can survive. The results directory is the one location that persists
    # across teardown; copy controller logs and candidate-output logs there
    # together with the environment facts a post-mortem needs.
    try:
        scene_dir = result_path.parent / "failure-scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for log in sorted(temp_root.glob("*.log*")):
            try:
                shutil.copy2(log, scene_dir / log.name)
            except OSError:
                pass
        candidate_output = temp_root / "candidate-output"
        for log in sorted(candidate_output.glob("*.log*")):
            try:
                destination = scene_dir / "candidate-output" / log.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, destination)
            except OSError:
                pass
        runner_digest = ""
        try:
            runner_digest = hashlib.sha256(runner.read_bytes()).hexdigest()
        except OSError:
            pass
        listing: Dict[str, Any] = {
            "workspace": str(workspace),
            "workspace_exists": workspace.is_dir(),
            "runner": str(runner),
            "runner_sha256": runner_digest,
            "euid": os.geteuid(),
            "cwd": os.getcwd(),
            "env_deadline_ms": os.environ.get("ALFONSO_CANDIDATE_DEADLINE_MS", ""),
        }
        try:
            listing["workspace_stat"] = repr(os.stat(workspace))
            listing["workspace_entries"] = sorted(
                entry.name for entry in workspace.iterdir()
            )[:20]
        except OSError as error:
            listing["workspace_stat_error"] = str(error)
        (scene_dir / "scene.json").write_text(json.dumps(listing, indent=1))
    except OSError:
        pass


def cleanup_staging_tree(temp_root: Path, runner: Path) -> None:
    cleanup_log = temp_root / "staging-cleanup.log"
    try:
        cleanup_status = run_through_runner(
            runner,
            ["/bin/rm", "-rf", str(temp_root / "build")],
            cleanup_log,
        )
    except OSError as error:
        print(f"warning: candidate staging cleanup could not run: {error}", file=sys.stderr)
        return
    if cleanup_status != 0:
        print(
            f"warning: candidate staging cleanup exited with status {cleanup_status}",
            file=sys.stderr,
        )


def runner_stdout(runner: Path, argv: Sequence[str], log_path: Path, limit: int = 4096) -> str:
    status = run_through_runner(runner, argv, log_path)
    output = log_path.read_bytes()
    if len(output) > limit:
        raise CandidateRejected(f"runner output exceeded {limit} bytes for {' '.join(argv[:2])}")
    if status != 0:
        raise CandidateRejected(f"runner command failed with status {status}: {' '.join(argv[:2])}")
    return output.decode("utf-8", errors="strict").strip()


def parse_workspace_commit(output: str) -> str:
    lines = output.splitlines()
    if len(lines) != 1 or re.fullmatch(r"[0-9a-f]{40}", lines[0]) is None:
        raise CandidateRejected("candidate workspace did not report one full Git commit SHA")
    return lines[0]


def sibling_head(
    runner: Path, sibling: Path, log_path: Path
) -> str:
    status = run_through_runner(
        runner,
        [
            "/usr/bin/git",
            "-c",
            f"safe.directory={sibling}",
            "-C",
            str(sibling),
            "rev-parse",
            "HEAD",
        ],
        log_path,
    )
    if status != 0:
        return "non-git"
    output = log_path.read_text(errors="replace").strip()
    if re.fullmatch(r"[0-9a-f]{40}", output) is None:
        return "non-git"
    return output


def sibling_provenance(siblings: Sequence[Tuple[str, str]]) -> str:
    return "Sibling HEADs: " + ", ".join(
        f"{name}={head}" for name, head in siblings
    ) + "."


def load_result(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateRejected(f"candidate decode output is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CandidateRejected("candidate decode output must be a JSON object")
    return payload


def validate_decode_result(payload: Mapping[str, Any], references: Sequence[Mapping[str, Any]]) -> float:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(references):
        raise CandidateRejected("candidate output has the wrong number of prompt results")
    if payload.get("prompts") != len(references) or payload.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise CandidateRejected("candidate output changed the prompt or generation count")
    if payload.get("exact_prompts") != len(references):
        raise CandidateRejected("candidate did not report every prompt as token-exact")
    if payload.get("accepted_near_ties") != 0:
        raise CandidateRejected("near-tie exemptions are forbidden by the campaign gate")

    token_count = 0
    for index, (actual, expected) in enumerate(zip(results, references), start=1):
        if not isinstance(actual, dict):
            raise CandidateRejected(f"candidate prompt result {index} is not an object")
        if actual.get("id") != expected.get("id") or actual.get("tokens") != expected.get("tokens"):
            raise CandidateRejected(f"token-exact decode mismatch at fixture row {index}")
        if actual.get("exact_reference") is not True:
            raise CandidateRejected(f"candidate prompt result {index} lacks an exact-reference verdict")
        token_count += len(expected["tokens"])

    decode_wall_s = payload.get("decode_wall_s")
    reported_tok_s = payload.get("decode_tok_per_s")
    if (
        not isinstance(decode_wall_s, (int, float))
        or isinstance(decode_wall_s, bool)
        or not math.isfinite(decode_wall_s)
        or decode_wall_s <= 0
    ):
        raise CandidateRejected("candidate reported an invalid decode wall time")
    computed_tok_s = token_count / float(decode_wall_s)
    if (
        not isinstance(reported_tok_s, (int, float))
        or isinstance(reported_tok_s, bool)
        or not math.isfinite(reported_tok_s)
        or not math.isclose(float(reported_tok_s), computed_tok_s, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise CandidateRejected("candidate throughput is inconsistent with tokens divided by decode time")
    return computed_tok_s


def parse_hook_log(text: str) -> None:
    for test_name in HOOK_TESTS:
        pattern = rf"(?m)^test qwen3_decode::tests::{re.escape(test_name)} \.\.\. ok$"
        if re.search(pattern, text) is None:
            raise CandidateRejected(f"required hook test did not pass: {test_name}")


def write_sample_fixture(path: Path, row: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    path.chmod(0o444)


def decode_command(
    binary: Path,
    model: Path,
    prompts: Path,
    references: Path,
    package_cache: Path,
    output: Path,
) -> List[str]:
    return [
        str(binary),
        "--model",
        str(model),
        "--tokenizer",
        str(model / "tokenizer.json"),
        "--generate-prompts",
        str(prompts),
        "--decode-reference",
        str(references),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--decode-cache-bucket",
        "512",
        "--decode-top-k",
        "5",
        "--device",
        "metal",
        "--dtype",
        "f16",
        "--execution",
        "explicit",
        "--package-cache",
        str(package_cache),
        "--out",
        str(output),
    ]


def configured_constants() -> Tuple[float, str]:
    baseline_text = os.environ.get("SYNAPSE_CAMPAIGN_BASELINE_TOK_S", str(BASELINE_TOK_S))
    digest = os.environ.get("SYNAPSE_CAMPAIGN_MODEL_SHA256", EXPECTED_MODEL_DIGEST)
    try:
        baseline = float(baseline_text)
    except ValueError as error:
        raise HarnessError("configured baseline is not numeric") from error
    if baseline != BASELINE_TOK_S:
        raise HarnessError("campaign registration baseline disagrees with the pinned harness")
    if digest != EXPECTED_MODEL_DIGEST:
        raise HarnessError("campaign registration model digest disagrees with the pinned harness")
    return baseline, digest


def run_harness(workspace_arg: str, runner_arg: str, result_arg: str) -> int:
    workspace = Path(workspace_arg).resolve()
    runner = Path(runner_arg).resolve()
    result_path = Path(result_arg).resolve()
    if not workspace.is_dir():
        raise HarnessError(f"candidate workspace is not a directory: {workspace}")
    if not runner.is_file() or not os.access(str(runner), os.X_OK):
        raise HarnessError(f"candidate runner is not executable: {runner}")

    baseline, expected_digest = configured_constants()
    model = Path(os.environ.get("SYNAPSE_CAMPAIGN_MODEL", str(DEFAULT_MODEL))).resolve()
    writer = ResultWriter(result_path)
    workspace_commit = ""
    gate_passed = False
    hooks_passed = False
    sibling_note = ""
    baseline_note = (
        f"Frozen master baseline: {baseline:.2f} tok/s on locked M1; measurement not completed."
    )
    writer.write(result_payload(False, False, [], None, "", baseline_note))

    temp_root = Path(tempfile.mkdtemp(prefix="synapse-decode-campaign-", dir="/tmp"))
    temp_root.chmod(0o755)
    try:
        fixture_root = temp_root / "fixtures"
        prompts_path, references_path, prompts, references = extract_and_verify_fixtures(fixture_root)
        actual_digest = model_content_digest(model)
        if actual_digest != expected_digest:
            raise HarnessError(
                f"model snapshot digest mismatch: expected {expected_digest}, got {actual_digest}"
            )

        # The runner creates these directories so Cargo can write as the
        # candidate identity. The controller only needs read/traverse access.
        temp_root.chmod(0o777)
        output_root, target_dir, package_cache = create_candidate_output_dirs(
            temp_root, runner
        )
        cargo = os.environ.get("SYNAPSE_CAMPAIGN_CARGO") or shutil.which("cargo")
        if not cargo:
            raise HarnessError("cargo is not available to build the candidate")

        staged_workspace, staged_siblings = stage_candidate_sources(workspace, temp_root, runner)
        sibling_heads = [
            (
                name,
                sibling_head(runner, sibling, temp_root / f"{name}-head.log"),
            )
            for name, sibling in staged_siblings
        ]
        sibling_note = sibling_provenance(sibling_heads)
        baseline_note = (
            f"Frozen master baseline: {baseline:.2f} tok/s on locked M1; "
            f"measurement not completed. {sibling_note}"
        )
        writer.write(result_payload(False, False, [], None, "", baseline_note))

        workspace_commit = parse_workspace_commit(
            runner_stdout(
                runner,
                [
                    "/usr/bin/git",
                    "-c",
                    f"safe.directory={staged_workspace}",
                    "-C",
                    str(staged_workspace),
                    "rev-parse",
                    "HEAD",
                ],
                temp_root / "workspace-commit.log",
            )
        )

        manifest = staged_workspace / "bench/spikes/unified-rt/Cargo.toml"
        candidate_env = candidate_environment(target_dir)
        build_status = run_through_runner(
            runner,
            [
                *candidate_env,
                cargo,
                "build",
                "--locked",
                "--offline",
                "--release",
                "--manifest-path",
                str(manifest),
                "--bin",
                "spike-unified-rt",
            ],
            temp_root / "build.log",
        )
        if build_status != 0:
            raise CandidateRejected(f"candidate release build failed with status {build_status}")
        binary = target_dir / "release/spike-unified-rt"

        gate_output = output_root / "gate.json"
        gate_status = run_through_runner(
            runner,
            [
                *candidate_env,
                *decode_command(
                    binary,
                    model,
                    prompts_path,
                    references_path,
                    package_cache,
                    gate_output,
                ),
            ],
            temp_root / "gate.log",
        )
        if gate_status != 0:
            raise CandidateRejected(f"20-prompt correctness gate failed with status {gate_status}")
        validate_decode_result(load_result(gate_output), references)
        gate_passed = True
        writer.write(
            result_payload(True, False, [], None, workspace_commit, baseline_note)
        )

        hooks_log = temp_root / "hooks.log"
        hooks_status = run_through_runner(
            runner,
            [
                *candidate_env,
                cargo,
                "test",
                "--locked",
                "--offline",
                "--release",
                "--manifest-path",
                str(manifest),
                "qwen3_decode::tests::",
                "--",
                "--nocapture",
            ],
            hooks_log,
        )
        if hooks_status != 0:
            raise CandidateRejected(f"decode hook suite failed with status {hooks_status}")
        parse_hook_log(hooks_log.read_text(errors="strict"))
        hooks_passed = True
        writer.write(
            result_payload(True, True, [], None, workspace_commit, baseline_note)
        )

        samples: List[float] = []
        for sample_number, fixture_index in enumerate(SAMPLE_PROMPT_INDICES, start=1):
            sample_prompt = fixture_root / f"sample-{sample_number:02d}-prompt.jsonl"
            sample_reference = fixture_root / f"sample-{sample_number:02d}-reference.jsonl"
            write_sample_fixture(sample_prompt, prompts[fixture_index])
            write_sample_fixture(sample_reference, references[fixture_index])
            sample_output = output_root / f"sample-{sample_number:02d}.json"
            sample_status = run_through_runner(
                runner,
                [
                    *candidate_env,
                    *decode_command(
                        binary,
                        model,
                        sample_prompt,
                        sample_reference,
                        package_cache,
                        sample_output,
                    ),
                ],
                temp_root / f"sample-{sample_number:02d}.log",
            )
            if sample_status != 0:
                raise CandidateRejected(
                    f"measurement sample {sample_number} failed with status {sample_status}"
                )
            samples.append(
                validate_decode_result(load_result(sample_output), [references[fixture_index]])
            )

        median_tok_s = statistics.median(samples)
        if not math.isfinite(median_tok_s) or median_tok_s <= 0:
            raise CandidateRejected("measurement median is not finite and positive")
        baseline_note = (
            f"Frozen master baseline: {baseline:.2f} tok/s on locked M1 "
            f"(DECODE-WAVE1.md); N={SAMPLE_COUNT} fresh processes with varied prompts. "
            f"{sibling_note}"
        )
        writer.write(
            result_payload(
                True,
                True,
                samples,
                median_tok_s,
                workspace_commit,
                baseline_note,
            )
        )
        return 0
    except CandidateRejected as error:
        note = (
            f"Frozen master baseline: {baseline:.2f} tok/s on locked M1. "
            f"{sibling_note} Candidate rejected: {error}"
        )
        writer.write(
            result_payload(gate_passed, hooks_passed, [], None, workspace_commit, note)
        )
        print(f"decode campaign candidate rejected: {error}", file=sys.stderr)
        preserve_failure_scene(temp_root, result_path, workspace, runner)
        # The campaign gate reads only the exit code and the numeric sample
        # field; a zero exit here would surface as an invalid measurement
        # instead of a rejected proposal.
        return 3
    except Exception as error:
        note = f"Frozen master baseline: {baseline:.2f} tok/s on locked M1. Harness refused: {error}"
        writer.write(
            result_payload(gate_passed, hooks_passed, [], None, workspace_commit, note)
        )
        print(f"decode campaign harness refused to run: {error}", file=sys.stderr)
        preserve_failure_scene(temp_root, result_path, workspace, runner)
        return 1
    finally:
        cleanup_staging_tree(temp_root, runner)
        shutil.rmtree(temp_root, ignore_errors=True)
        writer.close()


def expect_rejection(action: Any) -> None:
    try:
        action()
    except CandidateRejected:
        return
    raise AssertionError("expected CandidateRejected")


def expect_harness_error(action: Any) -> None:
    try:
        action()
    except HarnessError:
        return
    raise AssertionError("expected HarnessError")


def self_test() -> int:
    global model_content_digest, run_through_runner

    root = Path(tempfile.mkdtemp(prefix="synapse-decode-self-test-", dir="/tmp"))
    previous_siblings = os.environ.get("SYNAPSE_CAMPAIGN_SIBLINGS")
    try:
        mini_workspace = root / "workspace-source"
        mini_manifest = mini_workspace / "bench/spikes/unified-rt/Cargo.toml"
        mini_manifest.parent.mkdir(parents=True)
        mini_manifest.write_text("[package]\\nname = \\\"self-test\\\"\\n")
        # Use a pass-through binary for successful runner calls; Python's local macOS
        # runtime does not reliably reap temporary shell-script runners.
        fake_runner = Path("/usr/bin/env")
        silent_runner = Path("/usr/bin/false")
        try:
            copy_candidate_tree(
                silent_runner,
                mini_workspace,
                root / "silent-copy",
                root / "silent-copy.log",
            )
        except HarnessError as error:
            message = str(error)
            assert "runner exited 1 with no output" in message
            assert (
                "preamble exits silently when the action deadline is already expired "
                "or argv is malformed"
            ) in message
        else:
            raise AssertionError("expected empty runner log to produce a HarnessError")
        try:
            stage_candidate_sources(mini_workspace, root / "probe-failure", silent_runner)
        except HarnessError as error:
            message = str(error)
            assert "candidate runner probe failed with status 1" in message
            assert "output: <empty>" in message
        else:
            raise AssertionError("expected a silent runner probe to fail")

        original_run_through_runner = run_through_runner
        initial_runner_records: List[Tuple[str, ...]] = []

        def record_initial_runner(
            _runner: Path, argv: Sequence[str], log_path: Path
        ) -> int:
            initial_runner_records.append(tuple(argv))
            return original_run_through_runner(fake_runner, argv, log_path)

        run_through_runner = record_initial_runner
        os.environ.pop("SYNAPSE_CAMPAIGN_SIBLINGS", None)
        expect_harness_error(
            lambda: stage_candidate_sources(mini_workspace, root / "without-siblings", fake_runner)
        )

        sibling_sources = []
        for name in ("subconscious", "commons"):
            source = root / "sibling-sources" / name
            source.mkdir(parents=True)
            (source / "marker.txt").write_text(name)
            sibling_sources.append(source)
        os.environ["SYNAPSE_CAMPAIGN_SIBLINGS"] = ":".join(
            str(source) for source in sibling_sources
        )
        staged_workspace, staged_siblings = stage_candidate_sources(
            mini_workspace, root / "with-siblings", fake_runner
        )
        assert staged_workspace == root / "with-siblings/build/workspace"
        assert (staged_workspace / "bench/spikes/unified-rt/Cargo.toml").is_file()
        assert [name for name, _ in staged_siblings] == ["subconscious", "commons"]
        assert all((path / "marker.txt").read_text() == name for name, path in staged_siblings)
        expected_copy_commands = {
            ("/bin/cp", "-cR", str(mini_workspace), str(root / "with-siblings/build/workspace")),
            ("/bin/cp", "-cR", str(sibling_sources[0].resolve()), str(root / "with-siblings/build/subconscious")),
            ("/bin/cp", "-cR", str(sibling_sources[1].resolve()), str(root / "with-siblings/build/commons")),
        }
        assert expected_copy_commands.issubset(set(initial_runner_records))
        assert ("/bin/mkdir", "-p", str(root / "with-siblings/build")) in initial_runner_records
        expected_verification_commands = {
            (
                "/bin/sh",
                "-c",
                f"test -d {destination} && ls {destination} | head -1",
            )
            for destination in (
                root / "with-siblings/build/workspace",
                root / "with-siblings/build/subconscious",
                root / "with-siblings/build/commons",
            )
        }
        assert expected_verification_commands.issubset(set(initial_runner_records))

        empty_copy_records: List[Tuple[str, ...]] = []

        def empty_copy_runner(
            _runner: Path, argv: Sequence[str], log_path: Path
        ) -> int:
            empty_copy_records.append(tuple(argv))
            if list(argv) == ["/bin/sh", "-c", "echo runner-ok"]:
                log_path.write_text("runner-ok\n")
                return 0
            if argv and argv[0] in {"/bin/mkdir", "/bin/cp"}:
                log_path.write_text("")
                return 0
            if (
                len(argv) >= 3
                and list(argv[:2]) == ["/bin/sh", "-c"]
                and argv[2].startswith("test -d ")
            ):
                log_path.write_text("")
                return 0
            raise AssertionError(f"unexpected empty-copy runner command: {argv}")

        run_through_runner = empty_copy_runner
        try:
            stage_candidate_sources(mini_workspace, root / "empty-copy", fake_runner)
        except HarnessError as error:
            assert "copy reported success but destination is empty" in str(error)
        else:
            raise AssertionError("expected an empty successful copy to fail verification")
        assert ("/bin/mkdir", "-p", str(root / "empty-copy/build")) in empty_copy_records
        assert any(
            len(call) >= 3
            and call[:2] == ("/bin/sh", "-c")
            and call[2].startswith("test -d ")
            for call in empty_copy_records
        )
        run_through_runner = original_run_through_runner

        fake_build_manifests: List[str] = []
        fake_runner_calls: List[Tuple[str, ...]] = []
        original_run_through_runner = run_through_runner
        original_model_content_digest = model_content_digest
        previous_model = os.environ.get("SYNAPSE_CAMPAIGN_MODEL")
        previous_cargo = os.environ.get("SYNAPSE_CAMPAIGN_CARGO")
        previous_rustup_home = os.environ.get("RUSTUP_HOME")
        previous_cargo_home = os.environ.get("CARGO_HOME")

        def fake_run_through_runner(
            _runner: Path, argv: Sequence[str], log_path: Path
        ) -> int:
            fake_runner_calls.append(tuple(argv))
            if list(argv) == ["/bin/sh", "-c", "echo runner-ok"]:
                log_path.write_text("runner-ok\n")
                return 0
            if argv and argv[0] in {"/bin/cp", "/bin/chmod", "/bin/mkdir", "/bin/rm"}:
                return original_run_through_runner(fake_runner, argv, log_path)
            if (
                len(argv) >= 3
                and list(argv[:2]) == ["/bin/sh", "-c"]
                and argv[2].startswith("test -d ")
            ):
                return original_run_through_runner(fake_runner, argv, log_path)
            if "rev-parse" in argv:
                checkout = Path(argv[argv.index("-C") + 1])
                if "/build/workspace" in checkout.as_posix():
                    log_path.write_text("a" * 40 + "\n")
                    return 0
                log_path.write_text("")
                return 1
            if "build" in argv:
                fake_build_manifests.append(argv[argv.index("--manifest-path") + 1])
                candidate_log = log_path.parent / "candidate-output" / "candidate-build.log"
                candidate_log.write_text("candidate cargo diagnostics\n")
                log_path.write_text("cargo build failed\n")
                log_path.with_name(log_path.name + ".stderr").write_text("cargo stderr\n")
                return 1
            raise AssertionError(f"unexpected self-test runner command: {argv}")

        try:
            model_content_digest = lambda _model: EXPECTED_MODEL_DIGEST
            run_through_runner = fake_run_through_runner
            os.environ["SYNAPSE_CAMPAIGN_MODEL"] = str(root / "fake-model")
            os.environ["SYNAPSE_CAMPAIGN_CARGO"] = "/bin/false"
            os.environ["RUSTUP_HOME"] = str(root / "fake-rustup")
            os.environ["CARGO_HOME"] = str(root / "fake-cargo")
            fake_result = root / "fake-result.json"
            assert (
                run_harness(str(mini_workspace), str(fake_runner), str(fake_result))
                == 3
            )
            assert len(fake_build_manifests) == 1
            assert fake_build_manifests[0].endswith(
                "/build/workspace/bench/spikes/unified-rt/Cargo.toml"
            )
            assert any(
                call[:3] == ("/bin/cp", "-cR", str(mini_workspace.resolve()))
                and Path(call[3]).name == "workspace"
                for call in fake_runner_calls
            )
            assert any(
                call[:3] == ("/bin/cp", "-cR", str(sibling_sources[0].resolve()))
                and Path(call[3]).name == "subconscious"
                for call in fake_runner_calls
            )
            assert any(
                call[:3] == ("/bin/cp", "-cR", str(sibling_sources[1].resolve()))
                and Path(call[3]).name == "commons"
                for call in fake_runner_calls
            )
            assert any(
                call[:2] == ("/bin/mkdir", "-p")
                and Path(call[2]).name == "build"
                for call in fake_runner_calls
            )
            assert any(
                call[:2] == ("/bin/mkdir", "-p")
                and len(call) == 5
                and Path(call[2]).name == "candidate-output"
                and Path(call[3]).name == "target"
                and Path(call[4]).name == "packages"
                for call in fake_runner_calls
            )
            assert any(
                call[:2] == ("/bin/chmod", "755")
                and len(call) == 5
                and Path(call[2]).name == "candidate-output"
                and Path(call[3]).name == "target"
                and Path(call[4]).name == "packages"
                for call in fake_runner_calls
            )
            assert any(
                "RUSTUP_HOME=" + str(root / "fake-rustup") in call
                and "CARGO_HOME=" + str(root / "fake-cargo") in call
                and "build" in call
                for call in fake_runner_calls
            )
            assert any(
                len(call) >= 3
                and call[:2] == ("/bin/sh", "-c")
                and call[2].startswith("test -d ")
                for call in fake_runner_calls
            )
            assert any(
                call[:2] == ("/bin/rm", "-rf")
                and Path(call[2]).name == "build"
                for call in fake_runner_calls
            )
            failure_scene = fake_result.parent / "failure-scene"
            assert (failure_scene / "build.log").read_text() == "cargo build failed\n"
            assert (failure_scene / "build.log.stderr").read_text() == "cargo stderr\n"
            assert (
                failure_scene / "candidate-output" / "candidate-build.log"
            ).read_text() == "candidate cargo diagnostics\n"
            assert "Sibling HEADs: subconscious=non-git, commons=non-git." in fake_result.read_text()
        finally:
            model_content_digest = original_model_content_digest
            run_through_runner = original_run_through_runner
            if previous_model is None:
                os.environ.pop("SYNAPSE_CAMPAIGN_MODEL", None)
            else:
                os.environ["SYNAPSE_CAMPAIGN_MODEL"] = previous_model
            if previous_cargo is None:
                os.environ.pop("SYNAPSE_CAMPAIGN_CARGO", None)
            else:
                os.environ["SYNAPSE_CAMPAIGN_CARGO"] = previous_cargo
            if previous_rustup_home is None:
                os.environ.pop("RUSTUP_HOME", None)
            else:
                os.environ["RUSTUP_HOME"] = previous_rustup_home
            if previous_cargo_home is None:
                os.environ.pop("CARGO_HOME", None)
            else:
                os.environ["CARGO_HOME"] = previous_cargo_home

        _, _, _, references = extract_and_verify_fixtures(root / "fixtures")
        expected = references[0]
        wall = 1.6
        payload: Dict[str, Any] = {
            "prompts": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "exact_prompts": 1,
            "accepted_near_ties": 0,
            "decode_wall_s": wall,
            "decode_tok_per_s": MAX_NEW_TOKENS / wall,
            "results": [
                {
                    "id": expected["id"],
                    "tokens": list(expected["tokens"]),
                    "exact_reference": True,
                }
            ],
        }
        assert validate_decode_result(payload, [expected]) == MAX_NEW_TOKENS / wall
        wrong_token = json.loads(json.dumps(payload))
        wrong_token["results"][0]["tokens"][0] += 1
        expect_rejection(lambda: validate_decode_result(wrong_token, [expected]))
        dishonest_speed = json.loads(json.dumps(payload))
        dishonest_speed["decode_tok_per_s"] += 1
        expect_rejection(lambda: validate_decode_result(dishonest_speed, [expected]))

        complete_log = "\n".join(
            f"test qwen3_decode::tests::{name} ... ok" for name in HOOK_TESTS
        )
        parse_hook_log(complete_log)
        expect_rejection(lambda: parse_hook_log(complete_log.splitlines()[0]))
        assert parse_workspace_commit("a" * 40) == "a" * 40
        expect_rejection(lambda: parse_workspace_commit("not-a-commit"))
        print("decode-harness self-test passed")
        return 0
    finally:
        if previous_siblings is None:
            os.environ.pop("SYNAPSE_CAMPAIGN_SIBLINGS", None)
        else:
            os.environ["SYNAPSE_CAMPAIGN_SIBLINGS"] = previous_siblings
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    arguments = sys.argv[1:]
    if arguments == ["--self-test"]:
        return self_test()
    if len(arguments) != 3:
        print(
            "usage: decode-harness.sh {workspace} {candidate_runner} {result}",
            file=sys.stderr,
        )
        return 2
    return run_harness(arguments[0], arguments[1], arguments[2])


if __name__ == "__main__":
    raise SystemExit(main())
PY

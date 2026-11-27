"""Download one JSON file per day of train running from Fintraffic's open API.

Usage: python fetch_days.py <output-dir> <first-date YYYY-MM-DD> <number-of-weekdays>
Data: https://rata.digitraffic.fi (CC-BY 4.0). The API asks clients to send a
Digitraffic-User header naming the application.
"""
import datetime
import gzip
import os
import sys
import time
import urllib.request

out, d, n = sys.argv[1], datetime.date.fromisoformat(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
got = 0
while got < n:
    if d.weekday() < 5:
        path = os.path.join(out, f"{d.isoformat()}.json")
        if not os.path.exists(path):
            req = urllib.request.Request(f"https://rata.digitraffic.fi/api/v1/trains/{d.isoformat()}",
                                         headers={"Digitraffic-User": "research-script", "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            open(path, "wb").write(raw)
            time.sleep(1)
        got += 1
    d += datetime.timedelta(days=1)
print(f"{got} days in {out}")

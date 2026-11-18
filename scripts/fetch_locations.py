"""Download the METR-LA sensor coordinates (latitude, longitude) used by the map.

Source: the DCRNN repository (Li et al., ICLR 2018), MIT license: https://github.com/liyaguang/DCRNN
Run once:  python3 scripts/fetch_locations.py
"""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv"
out = Path(__file__).resolve().parent.parent / "data" / "sensor_locations.csv"
out.write_bytes(urlopen(URL).read())
print("wrote", out)

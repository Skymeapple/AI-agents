#!/usr/bin/env bash
# Find the Columbia Basin DART csv layout for daily river conditions (flow, spill, dissolved gas).
set -u
B="https://www.cbr.washington.edu/dart/cs/php/rpt"
try() { echo "--- $1"; curl -sSL -m 120 -A "research-script" "$1" | head -c 1500; echo; }
try "$B/river_daily.php?sc=1&outputFormat=csv&year=2023&proj=LWG&span=no&startdate=4%2F1&enddate=4%2F10"
try "$B/river_daily.php?sc=1&outputFormat=csv&year=2023&proj=LGNW&span=no&startdate=4%2F1&enddate=4%2F10"
try "$B/river_daily.php?sc=1&outputFormat=csv&year=2023&proj=BON&span=no&startdate=4%2F1&enddate=4%2F10"
try "$B/river_daily.php?sc=1&outputFormat=csv&year=2023&proj=CCIW&span=no&startdate=4%2F1&enddate=4%2F10"
try "$B/mg.php?sc=1&mgconfig=river&outputFormat=csvSingle&year%5B%5D=2023&loc%5B%5D=LWG&loc%5B%5D=LGNW&data%5B%5D=Dissolved+Gas+Percent&data%5B%5D=Spill&data%5B%5D=Outflow&startdate=4%2F1&enddate=4%2F10&avgyear=0&consolidate=1&grid=1&y1min=&y1max=&y2min=&y2max=&size=medium"
try "$B/adult_daily.php?sc=1&outputFormat=csv&year=2023&proj=LWG&span=no&startdate=4%2F1&enddate=4%2F5&run=" 
try "https://www.cbr.washington.edu/dart/cs/php/rpt/pit_obs_de.php?sc=1&outputFormat=csv&year=2023&proj=LWG&species=1&run=1&rear_type=Null&stage=J&span=no&startdate=4%2F1&enddate=4%2F5&syear=2023&eyear=2023&summary=yes"
try "https://www.cbr.washington.edu/dart/cs/php/rpt/smolt_index.php?sc=1&outputFormat=csv&year=2023&proj=LWG&span=no&startdate=4%2F1&enddate=4%2F10"

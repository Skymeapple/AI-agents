#!/usr/bin/env bash
# Downloads the public inputs (about 1.5 GB streamed; only 2023 files are kept).
set -euo pipefail
cd "$(dirname "$0")/../data"
# 1) gridemissions bulk dataset (de Chalendar et al.): EIA-930 reconciled generation by fuel + CO2 intensity
curl -s "https://s3.us-east-2.amazonaws.com/gridemissions/processed.tar.gz" | \
  tar -xz --wildcards 'processed/EIA930_2023_*_elec.csv' 'processed/EIA930_2023_*_co2i.csv'
# 2) NOAA ISD-Lite hourly weather, 2023
mkdir -p isd
for s in 724030-93738 722590-03927 724945-23293 722780-23183 726988-24219 725500-14942 723560-13968 \
         722190-13874 723340-13893 723860-23169 725720-24127 723140-13881 725090-14739; do
  curl -s -o isd/$s-2023.gz "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2023/$s-2023.gz"
done

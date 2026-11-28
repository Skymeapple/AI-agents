# Air-quality sensors: download SensEURCity (5.7 GB, Zenodo 17858205) and describe its
# files, so the calibration-transfer experiment can be written against the real layout.
set -e
cd "$DATA"
curl -sS -L -f -m 3000 -o sens.zip "https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content"
ls -la sens.zip
unzip -l sens.zip | awk '{print $1, $4, $5, $6}' | sort -k2 | head -150
echo "== file count by folder and extension"
unzip -Z1 sens.zip | sed -E 's|/[^/]*\.([A-Za-z0-9]+)$| .\1|' | sort | uniq -c | sort -rn | head -40
for f in $(unzip -Z1 sens.zip | grep -iE "\.(csv|txt|md)$" | grep -iE "readme|meta|location|deploy|descr" | head -8); do echo "== head of $f"; unzip -p sens.zip "$f" | head -c 2500; echo; done
first=$(unzip -Z1 sens.zip | grep -iE "\.csv$" | grep -viE "meta|location|deploy" | head -1)
echo "== first data file: $first"; unzip -p sens.zip "$first" | head -5 | cut -c1-3000
unzip -p sens.zip "$first" | wc -l

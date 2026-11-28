# Intraday-overlay stand-in on Bybit public trades: 24 perpetuals, 1 April - 21 June 2024 (last 21 days tested).
set -e
pip -q install numpy scipy scikit-learn pandas
SYMS="BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT LTCUSDT BCHUSDT TRXUSDT NEARUSDT ATOMUSDT UNIUSDT FILUSDT APTUSDT ARBUSDT OPUSDT SUIUSDT INJUSDT AAVEUSDT ETCUSDT XLMUSDT"
DAYS=$(python3 -c "import datetime as d; s=d.date(2024,4,1); print(' '.join(str(s+d.timedelta(i)) for i in range(82)))")
A=research/hfm-projects/quant-finance/sp500-activity-ranking/activity_floor.py
for s in $SYMS; do
  mkdir -p "$DATA/raw"
  for day in $DAYS; do curl -sS -L -f -m 600 -o "$DATA/raw/$s$day.csv.gz" "https://public.bybit.com/trading/$s/$s$day.csv.gz" || echo "missing $s $day"; done
  python3 $A build "$DATA/raw" "$DATA/$s.npz" $DAYS -- $s
  rm -rf "$DATA/raw"
done
python3 - "$DATA" $SYMS <<'PY'
import sys, numpy as np
d, syms = sys.argv[1], sys.argv[2:]
parts = [np.load(f"{d}/{s}.npz") for s in syms]
np.savez_compressed(f"{d}/bars.npz", vol=np.concatenate([p["vol"] for p in parts]), cnt=np.concatenate([p["cnt"] for p in parts]),
                    close=np.concatenate([p["close"] for p in parts]), syms=np.array(syms), days=parts[0]["days"])
print("merged", len(syms))
PY
timeout 200m python3 -u research/hfm-projects/quant-finance/ashare-intraday-overlay/overlay_floor.py "$DATA/bars.npz" 21

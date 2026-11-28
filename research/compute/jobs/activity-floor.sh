# Activity-surprise ranking stand-in on Bybit public trades: 24 perpetuals, 1-21 June 2024 (last 7 days tested).
set -e
pip -q install numpy scipy scikit-learn pandas
SYMS="BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT LTCUSDT BCHUSDT TRXUSDT NEARUSDT ATOMUSDT UNIUSDT FILUSDT APTUSDT ARBUSDT OPUSDT SUIUSDT INJUSDT AAVEUSDT ETCUSDT XLMUSDT"
DAYS=$(python3 -c "print(' '.join(f'2024-06-{d:02d}' for d in range(1, 22)))")
cd "$DATA"
for s in $SYMS; do for d in $DAYS; do
  curl -sS -L -f -m 600 -o $s$d.csv.gz "https://public.bybit.com/trading/$s/$s$d.csv.gz" || echo "missing $s $d"
done; done
du -sh .
cd "$GITHUB_WORKSPACE"
F=research/hfm-projects/quant-finance/sp500-activity-ranking/activity_floor.py
python3 -u $F build "$DATA" "$DATA/bars.npz" $DAYS -- $SYMS
timeout 200m python3 -u $F "$DATA/bars.npz" 7

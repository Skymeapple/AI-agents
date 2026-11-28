# Cross-venue routing replay on five days of public BTCUSDT perpetual trades (Binance and Bybit).
set -e
pip -q install numpy pandas
cd "$DATA"; args=""
for d in 2025-03-03 2025-03-04 2025-03-05 2025-03-06 2025-03-07; do
  curl -sS -L -f -m 1200 -o bn-$d.zip "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-$d.zip"
  unzip -q -o bn-$d.zip && mv BTCUSDT-aggTrades-$d.csv bn-$d.csv && rm bn-$d.zip
  curl -sS -L -f -m 1200 -o bb-$d.csv.gz "https://public.bybit.com/trading/BTCUSDT/BTCUSDT$d.csv.gz" && gunzip -f bb-$d.csv.gz
  args="$args $DATA/bn-$d.csv $DATA/bb-$d.csv"
done
ls -la; head -2 bn-2025-03-03.csv; head -2 bb-2025-03-03.csv
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/quant-finance/cross-venue-routing/routing_replay.py $args

# Mapping C test: path (memory) features against state features for depth depletion, Bybit BTCUSDT, 3-5 June 2024.
set -e
pip -q install numpy scipy scikit-learn
cd "$DATA"
for d in 2024-06-03 2024-06-04 2024-06-05; do
  curl -sS -L -f -m 1200 -o ${d}_BTCUSDT_ob500.data.zip "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/${d}_BTCUSDT_ob500.data.zip"
  curl -sS -L -f -m 1200 -o BTCUSDT${d}.csv.gz "https://public.bybit.com/trading/BTCUSDT/BTCUSDT${d}.csv.gz"
done
ls -la
cd "$GITHUB_WORKSPACE"
timeout 300m python3 -u research/hfm-projects/quant-finance/order-book-depletion/memory_floor.py "$DATA" BTCUSDT 2024-06-03 2024-06-04 2024-06-05

# Order-book stand-in on Bybit public data: ETHUSDT perpetual, 3-5 June 2024 (fit on two days, test on the third).
set -e
pip -q install numpy scipy scikit-learn
cd "$DATA"
for d in 2024-06-03 2024-06-04 2024-06-05; do
  curl -sS -L -f -m 1200 -o ${d}_ETHUSDT_ob500.data.zip "https://quote-saver.bycsi.com/orderbook/linear/ETHUSDT/${d}_ETHUSDT_ob500.data.zip"
  curl -sS -L -f -m 1200 -o ETHUSDT${d}.csv.gz "https://public.bybit.com/trading/ETHUSDT/ETHUSDT${d}.csv.gz"
done
ls -la
cd "$GITHUB_WORKSPACE"
timeout 300m python3 -u research/hfm-projects/quant-finance/order-book-depletion/bybit_book_floor.py "$DATA" ETHUSDT 2024-06-03 2024-06-04 2024-06-05 -- 20000 200000

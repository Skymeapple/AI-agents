# Order-book stand-in on Bybit public data: XRPUSDT perpetual, 3-5 June 2024 (fit on two days, test on the third).
set -e
pip -q install numpy scipy scikit-learn
cd "$DATA"
for d in 2024-06-03 2024-06-04 2024-06-05; do
  curl -sS -L -f -m 1200 -o ${d}_XRPUSDT_ob500.data.zip "https://quote-saver.bycsi.com/orderbook/linear/XRPUSDT/${d}_XRPUSDT_ob500.data.zip"
  curl -sS -L -f -m 1200 -o XRPUSDT${d}.csv.gz "https://public.bybit.com/trading/XRPUSDT/XRPUSDT${d}.csv.gz"
done
ls -la
cd "$GITHUB_WORKSPACE"
timeout 300m python3 -u research/hfm-projects/quant-finance/order-book-depletion/bybit_book_floor.py "$DATA" XRPUSDT 2024-06-03 2024-06-04 2024-06-05 -- 20000 200000

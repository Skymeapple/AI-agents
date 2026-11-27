# LOBSTER samples now need an application. Probe Bybit's public historical order-book files instead.
for u in "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2024-06-03_BTCUSDT_ob500.data.zip" \
         "https://quote-saver.bycsi.com/orderbook/linear/ETHUSDT/2024-06-03_ETHUSDT_ob500.data.zip" \
         "https://quote-saver.bycsi.com/orderbook/spot/BTCUSDT/2024-06-03_BTCUSDT_ob200.data.zip" \
         "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2024-06-03.csv.gz"; do
  echo "== $u"; curl -sS -L -m 300 -o f.bin -w "%{http_code} %{size_download}\n" "$u"; file f.bin
  if unzip -l f.bin >/dev/null 2>&1; then unzip -p f.bin | head -c 700; echo; fi
done

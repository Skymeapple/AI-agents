# Order-book stand-in on LOBSTER's free 2012-06-21 sample (five stocks, ten levels): depth forecasting and execution replay.
set -e
pip -q install numpy scipy scikit-learn
cd "$DATA"
for s in AMZN AAPL GOOG INTC MSFT; do
  curl -sS -L -f -m 900 -o $s.zip "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_${s}_2012-06-21_10.zip" && unzip -q -o $s.zip || echo "FAILED $s"
done
ls -la
cd "$GITHUB_WORKSPACE"
for q in 500 5000; do
  timeout 150m python3 -u research/hfm-projects/quant-finance/order-book-depletion/lobster_floor.py "$DATA" $q || echo "stopped early"
done

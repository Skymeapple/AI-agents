# Find the real download links for LOBSTER's free sample files.
curl -sS -L -m 120 -o a.bin -D h.txt "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip"; cat h.txt; ls -la a.bin; head -c 600 a.bin; echo
for u in https://lobsterdata.com/info/DataSamples.php https://lobsterdata.com/info/DataStructure.php https://lobsterdata.com/; do
  echo "== $u"; curl -sS -L -m 120 "$u" | grep -io 'href="[^"]*"' | grep -i 'sample\|zip\|7z' | sort -u | head -40
done

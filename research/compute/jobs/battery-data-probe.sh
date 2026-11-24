# Find a working download for the public Panasonic 18650PF / LG HG2 battery datasets
# (Kollmeyer, Mendeley Data) and describe their contents, so the cold-battery
# experiment can be written against the real file layout.
cd "$DATA"
for u in \
  "https://data.mendeley.com/public-files/datasets/wykht8y7tg/files/96dcfe69-2a3c-4f29-a5e3-0b3b6b3b4a3e/file_downloaded" \
  "https://data.mendeley.com/api/datasets-v2/datasets/wykht8y7tg/zip/download?version=1" \
  "https://data.mendeley.com/public-api/zip/wykht8y7tg/download/1" \
  "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wykht8y7tg-1.zip" \
  "https://data.mendeley.com/public-api/zip/cp3473x7xv/download/3" \
  "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cp3473x7xv-3.zip" ; do
  code=$(curl -sS -L -m 900 -o probe.bin -w "%{http_code} %{size_download} %{content_type}" "$u" || true)
  echo "$code  $u"
  if file probe.bin | grep -qi zip; then
    name=$(echo "$u" | grep -oE '(wykht8y7tg|cp3473x7xv)')
    mv probe.bin "$name.zip"; echo "== contents of $name.zip"; unzip -l "$name.zip" | head -80; unzip -l "$name.zip" | tail -3
  fi
done
curl -sS -m 60 "https://data.mendeley.com/public-api/datasets/wykht8y7tg" | head -c 3000; echo
curl -sS -m 60 "https://data.mendeley.com/public-api/datasets/wykht8y7tg/files?folder_id=root&version=1" | head -c 4000; echo

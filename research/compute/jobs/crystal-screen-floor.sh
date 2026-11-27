# Crystal screening stand-in on JARVIS-DFT 3D (2022 release): screen vs random vs oracle selection in held-out chemical systems.
set -e
pip -q install numpy scikit-learn
cd "$DATA"
curl -sS -L -f -m 1800 -o jdft.zip "https://ndownloader.figshare.com/files/38521619"
python3 -c "import zipfile; z=zipfile.ZipFile('jdft.zip'); print(z.namelist()); z.extractall('.')"
ls -la
cd "$GITHUB_WORKSPACE"
timeout 300m python3 -u research/hfm-projects/materials/crystal-generation/screen_floor.py "$DATA/jdft_3d-12-12-2022.json" 0.0 0.05 0.1

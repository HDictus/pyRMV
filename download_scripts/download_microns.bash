cd download_scripts/data_store
mkdir microns
cd microns
#python3 -m venv .venv
source .venv/bin/activate
#pip install ../../../
#pip install Connectome-Utilities
#pip install pyarrow
if [ ! -f microns_mm3_connectome_v1181.h5 ]; then
    wget https://zenodo.org/records/13849415/files/microns_mm3_connectome_v1181.h5
fi

python ../../format_microns.py
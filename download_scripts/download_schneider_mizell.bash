mkdir -p download_scripts/data_store/schneider-mizell
cd download_scripts/data_store/schneider-mizell
wget https://zenodo.org/records/7641781/files/cell_types.csv
wget https://zenodo.org/records/7641781/files/inhibitory_synapses_onto_column.csv
wget https://zenodo.org/records/10463756/files/inhibitory_edgelist.csv

python3 -m venv .venv
.venv/bin/pip install ../../../
.venv/bin/python ../../create_schneider_mizell_dataframes.py
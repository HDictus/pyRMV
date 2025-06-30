mkdir -p download_scripts/data_store/ecephys
python -m venv .siegle_ecephys_venv
.siegle_ecephys_venv/bin/pip install allensdk analysis-neuro
.siegle_ecephys_venv/bin/python download_scripts/download_siegle_data.py
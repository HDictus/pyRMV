# required on EPFL supercomputer
#module purge
#module load gcc py-h5py
##############################
cd download_scripts/data_store
mkdir aisynphys
cd aisynphys
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
git clone https://github.com/AllenInstitute/aisynphys.git
pip install aisynphys/
pip install -e ../../../
# pip install h5py==2.9
pip install SQLAlchemy==1.3.11
pip install pyyaml
pip install numba
git clone https://github.com/alleninstitute/neuroanalysis
pip install neuroanalysis

#pip install \
#    h5py==2.9\
#    sqlalchemy==1.3.11\
#    git+https://github.com/pyqtgraph/pyqtgraph\
#    git+https://github.com/alleninstitute/neuroanalysis\
#    git+https://github.com/alleninstitute/aisynphys
#pip install pillow\
#    psycopg2\
#    tqdm\
#    pandas\
#    matplotlib\
#    jupyter\
#    pandas\
#    seaborn\
#    scipy\
#    pyyaml\
#    numba\
#    scikit-image\
#    scikit-learn\
#    statsmodels\
#    lmfit

pip install pyarrow fastparquet

python ../../download_aisynphys.py
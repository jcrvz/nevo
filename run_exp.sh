module load uv

# Load Python 3.10 module
module load python/3.10

# Tell uv to use Python 3.10
uv python pin 3.10

# Clean and resync
#rm -rf .venv uv.lock
uv sync

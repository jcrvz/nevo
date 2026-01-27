module load uv

# Load Python 3.10 module
#module load python/3.10

# Tell uv to use Python 3.10
uv python pin 3.10

# Clean and resync
#rm -rf .venv uv.lock
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate

# Running python from uv
echo "Running Python from uv:"
which python
#uv python --version
python --version
echo ""

# Ready to run
echo "Setup complete. You can now run your Python application within the uv environment."

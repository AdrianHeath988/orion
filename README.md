# Orion

Adding installation instructions and additional examples shortly.
sudo swapoff /swapfile \
sudo rm /swapfile \
sudo fallocate -l 100G /swapfile \
sudo chmod 600 /swapfile \
sudo mkswap /swapfile \
sudo swapon /swapfile \
sudo swapon --show


poetry install

eval $(poetry env activate)
or 
poetry shell
or
source $(poetry env info --path)/bin/activate
or
poetry run python3 run_lola.py
or
source /home/adrian/.cache/pypoetry/virtualenvs/orion-fhe-NaQa48nB-py3.10/bin/activate


python3 run_lola.py
or
python3 run_mlp.py # (smaller model)



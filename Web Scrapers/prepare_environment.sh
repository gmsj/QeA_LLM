virtualenv --always-copy -p python3 virtual/
source virtual/bin/activate 
python3 -m pip install --no-cache-dir -r requirements.txt --ignore-installed
deactivate
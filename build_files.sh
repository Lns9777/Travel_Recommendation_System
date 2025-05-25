# pip install -r requirements.txt
# python3.9 manage.py collectstatic
#!/bin/bash

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py collectstatic --noinput

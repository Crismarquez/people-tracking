# People Tracking
This repo contains an api that allow to obtain the route of the detected people.

## System requirements
ubuntu 18 - 20

python >= 3.7

## Clone repo
<pre>
https://github.com/bluelabsai/people_tracking.git
cd people_tracking
</pre> 

## Virtual enviroment
<pre>
python3 -m venv .venv
source .venv/bin/activate
</pre> 

## Install dependencies
<pre>
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:${PWD}"
</pre> 

## Run application
<pre>
python3 app.py
</pre> 

**In your browser: http://0.0.0.0:8000/docs*


# Final Year Project 

## To Run

### Create a Virtual Environment
Firstly create a new virtual environment in Python and install the required packages. We use Python 3.8.
To do that, run the following commands in the root of the folder containing the code:
`virtualenv -p /usr/bin/python3.8 venv`
`source venv/bin/activate`
`pip install -r requirements.txt`


### Run Experiments
By default, start.py runs the simple.yml config file
`python start.py`

To run a specific config file, create an `example_config.yml` file with the desired parameters and run the following command in the terminal:
`python start.py --config=example_config`


## Disclaimer
I was given everything in the first commit by my supervisor. Everything afterwards is my work.

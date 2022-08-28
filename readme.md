# Demo

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a python environment called *env*
```
virtualenv env
```
Secondly, we will login to the *env* environement
```
source ./env/bin/activate
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/ml-auto-app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

###  Launch the app

```
streamlit run main.py
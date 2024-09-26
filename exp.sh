#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# 1. setup anaconda environment
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=research
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

sleep 5

# 2. set up  mujoco
#source ./mujoco_setup.sh

# 3. misc
# missing osmesa file, corrected with anaconda
#conda install -y -c menpo osmesa
#export C_INCLUDE_PATH="$ENVDIR/include:$C_LIBRARY_PATH"
pip install stable-baselines3
WORKING_DIR="$(pwd)"
mkdir d4rl_temp
export D4RL_DATASET_DIR="$WORKING_DIR/d4rl_temp"
#pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

pip install gymnasium

pip install libsumo
export LIBSUMO_AS_TRACI=1

# launch code
python3 run_single_continual.py "$@"
#python3 run_traffic.py "$@"

unset D4RL_DATASET_DIR
rm -rf d4rl_temp


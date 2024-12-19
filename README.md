# On Continual Learning in Multiclass Queueing
Adapted from source code of the paper [***Learning to Stabilize Online Reinforcement Learning in Unbounded State Spaces***](https://arxiv.org/abs/2306.01896) by Brahma S. Pavse, Matthew Zurek, Yudong Chen, Qiaomin Xie, Josiah P. Hanna.

This is the source code for my Senior Honors Thesis.

## Setting up the environment
```
conda env create -f environment.yml
```

## Running the code
Generic command:
```
python run_single_continual.py  --outfile <result_file> --env_name <queue/nmodel/crisscross/nsqueue> --mdp_num <0/1/2> --deployed_interaction_steps 5_000_000  --exp_name <exp_name>  --reward_function <opt/stab>  --seed 0  --truncated_horizon 200 --algo_name <algo_name> --lr 3e-4 --state_transformation <state_trans> --lyp_power <p> --adam_betas 0.9 <beta2>
```
where,
- `exp_name` can be anything
- `reward_function` is either `opt` for optimal only or `stab` for optimal + stability
- `algo_name` is either MW, PPO, or STOP-suffix where suffix can be anything to uniquely identify the algorithm run based state transformation and lyp power. Example: STOP-SL-2, denotes STOP with symloge and p = 2
- `state_transformation` is either `id, sigmoid, symsqrt, symloge`
- `lyp_power` is any floating number (p from the paper)
- `beta2` is Adam's beta_2 hyper-parameter

Example command:
```
python run_single_continual.py  --outfile result_file --env_name queue --mdp_num 2 --deployed_interaction_steps 5_000_000  --exp_name test  --reward_function stab  --seed 0  --truncated_horizon 200 --algo_name STOP-3 --lr 3e-4 --state_transformation sigmoid --lyp_power 3 --adam_beta 0.9
```

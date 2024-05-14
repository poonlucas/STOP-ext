# Learning to Stabilize Online Reinforcement Learning in Unbounded State Spaces

This is the source code accompanying the paper [***Learning to Stabilize Online Reinforcement Learning in Unbounded State Spaces***](https://arxiv.org/abs/2306.01896) by Brahma S. Pavse, Matthew Zurek, Yudong Chen, Qiaomin Xie, Josiah P. Hanna.

## Setting up the environment
```
conda create --name <env> --file requirements.txt
```

## Running the code
To execute the code run:
```
python run_single_continual.py  --outfile <result_file> --env_name <queue/nmodel> --mdp_num <0/1/2> --deployed_interaction_steps 5_000_000  --exp_name <exp_name>  --reward_function <opt/stab>  --seed 0  --truncated_horizon 200 --algo_name <algo_name> --lr 3e-4 --state_transformation <state_trans> --lyp_power <p> --adam_beta 0.9
```
where,
- `exp_name` can be anything
- `reward_function` is either `opt` for optimal only or `stab` for optimal + stability
- `algo_name` is either MW, PPO, or STOP-suffix where suffix can be anything to uniquely identify the algorithm run based state transformation and lyp power. Example: STOP-SL-2, denotes STOP with symloge and p = 2
- `state_transformation` is either `id, sigmoid, symsqrt, symloge`
- `lyp_power` is any floating number (p from the paper)

## Citation
If you found any part of this code useful, please consider citing our paper:

```
@inproceedings{
    pavse2024unbounded,
    title={Learning to Stabilize Online Reinforcement Learning in Unbounded State Spaces},
    author={Brahma S. Pavse, Matthew Zurek, Yudong Chen, Qiaomin Xie, Josiah P. Hanna},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=64fdhmogiD}
}
```
## Contact

If you have any questions, please feel free to email: pavse@wisc.edu!

New version of Olivier Sigaud's code https://github.com/osigaud/Basic-Policy-Gradient-Labs that includes CEM (Cross Entropy Method), new visualisations, and a Beta policy. 
You can find slides of a short presentation about the kind of experiments you can do with this code in /RAPPORT/slides_soutenance.pf (in French).

# Code usage


## Comparison between Policy Gradient and Cross Entropy Method


To launch a comparison between Policy Gradient (reinforce or custom PG) and CEM you can use:

```
python3 main.py --experiment comparison --env_name Pendulum-v0 --policy_type normal --nb_cycles 100 --nb_repet 1 --nb_eval 1 --eval_freq 20 --nb_trajs_cem 1 --reinforce True --nb_trajs_pg 20 --population 15 --lr_actor 1e-4
```
Plots and models are found in the /data folder.

## Study of PG 

For classic reinforce use --reinforce True. Otherwise you can build your own policy gradient algorithm like this for exemple: 
```
python3 main.py --experiment pg --env_name Pendulum-v0 --policy_type normal --critic_update_method dataset --study_name discount --gamma 0.99 --lr_critic 1e-2 --gradients sum+baseline --critic_estim_method td --nb_trajs_pg 20
```

## Study of CEM

To study the CEM you can use:

```
python3 main.py --experiment cem --population 20 --elites_frac 0.2 --sigma 1 --nb_trajs_cem 2
```

## Simple expert policy on Pendulum-v0


```
python3 simple_eval_expert.py
```


## Dependencies

```
pip install -r requirements.txt
```

## TODO

* Compatibility between CEM and Bernoulli Policy.
* Fix plot axes and problem of last eval.
* Make a "CustomNetwork" class with dim of NN and policy type as arguments. 
* Compatibility of simple_rendering.py with all types of policy. 
* Rebuild code for comparison so it runs both algo independently and then build plots. 
* Rethink compatibility with Vignettes. 
* Make better wrapper for Beta policy actions. 
* Make sure everything is in English
* Translate slides of results in English


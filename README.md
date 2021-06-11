New version of Olivier Sigaud's code https://github.com/osigaud/Basic-Policy-Gradient-Labs that includes CEM (Cross Entropy Method), new visualisations, and a Beta policy.

# Code usage


## Comparison between Policy Gradient and Cross Entropy Method

Pour lancer une comparaison entre PG et CEM
Les plots et politiques peuvent être retrouvés dans le dossier **/data**


Pour reproduire les résultats du rapport :

```
python3 main.py --experiment comparison --env_name Pendulum-v0 --policy_type normal --nb_repet 1 --nb_eval 1 --eval_freq 20 --nb_trajs_cem 1 --reinforce True --nb_trajs_pg 20 --population 15 --lr_actor 1e-4
```

## Study of PG 

For classic reinforce use --reinforce True. Otherwise you can build your own policy gradient algorithm like this for exemple: 
```
python3 main.py --expriment pg --env_name Pendulum-v0 --policy_type normal --critic_update_method dataset --study_name discount --gamma 0.99 --lr_critic 1e-2 --gradients sum+baseline --critic_estim_method td --nb_trajs_pg 20
```

## Study of CEM

To study the CEM you can use:

```
python3 main.py --experiment cem --population 20 --elite_frac 0.2 --sigma 1 --nb_trajs_cem 2
```

## Etude sur Pendulum-v0 avec une politique experte simple

Pour lancer une etude avec une politique experte simple

Exemple de commande :

```
python3 simple_eval_expert.py
```


## Installation des dépendances

```
pip install -r requirements.txt
```

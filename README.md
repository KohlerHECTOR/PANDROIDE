# Présentation du PANDROIDE

Projet PANDROIDE : Master 1 Androide 2020/2021

Sujet : Comparaison de méthodes évolutionnaires et d’apprentissage par renforcement sur des benchmarks de contrôle classique.

Auteurs : Damien LEGROS, Hector Kohler

Encadrant : Olivier Sigaud

Ce répértoire comprendra :

- Le code mis à jour au long du projet
- Un cahier des charges a rendre au cours de février (Environ 4/5 pages)
- Un carnet de bord (Premiere version à rendre avant le 28 février)
- Un rapport du projet

# Utilisation du code


## Comparaison entre Policy Gradient et Cross Entropy Method

Pour lancer une comparaison entre PG et CEM
Les plots et politiques peuvent être retrouvés dans le dossier **/data**


Pour reproduire les résultats du rapport :

```
python3 main.py --study_name comparison --nb_trajs_cem 2 --nb_trajs_pg 50 --nb_cycles 1000 --nb_eval 500 --population 25 --nb_repet 5 --policy_type beta --start_from_same_policy True --env_name CartPoleContinuous-v0 --lr_actor 1e-4
```

```
python3 main.py --study_name comparison --nb_trajs_cem 2 --nb_trajs_pg 50 --nb_cycles 1000 --nb_eval 500 --population 25 --nb_repet 5 --policy_type beta --start_from_same_policy True --env_name Pendulum-v0 --lr_actor 1e-4
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

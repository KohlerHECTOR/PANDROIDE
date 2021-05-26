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

Exemple de commande :

```
python3 main.py --env_name CartPoleContinuous-v0 --study_name comparison --population 10 --nb_repet 1 --nb_cycles 10 --nb_trajs 10 --nb_evals 5 --lr_actor 0.0001
```

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

## Visualisation du Gradient

Permet de creer une image pour visualiser le gradient entre la ou les politiques politiques misent dans le dossier **/Models**
Ces images peuvent être retrouvées dans le dossier **/Gradient_output**
La vignette sauvegardée au nom de **--filename** peut être retrouvée dans **/SavedGradient**

Exemple de commande :

```
python3 main_gradient.py --filename gradient
```

## Visualisation du Paysage

Permet de creer le paysage autour d'une politique ou plusieurs politiques misent dans le dossier **/Models**
Les vignettes peuvent être retrouvées dans le dossier **/Vignette_output**
La vignette sauvegardée au nom de **--filename** peut être retrouvée dans **/SavedVignette**

Exemple de commande :

```
python3 main_vignettes.py --nb_lines 15 --nb_evals 5 --maxalpha 50 --stepalpha 1 --title "Landscape" --filename vignette
```

Permet de visualiser le paysage d'une vignette déjà faite, en partant d'un fichier de **/SavedVignette**
Les vignettes peuvent être retrouvées dans le dossier **/Vignette_output**

Exemple de commande :

```
python3 savedVignette.py --title "Landscape Test" --filename vignette_test
```

## Multi-Threading (UNSTABLE)

Ajouter l'argument **--multi_treading True** dans votre commande pour effectuer les evaluations en multi-threading

## Installation des dépendances

```
pip install -r requirements.txt
```

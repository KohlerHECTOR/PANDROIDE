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

Exemple de commande :

```
python3 main.py --env_name CartPoleContinuous-v0 --study_name comparison --population 10 --nb_repet 1 --nb_cycles 10 --nb_trajs 10 --nb_evals 5 --lr_actor 0.0001 --seed 12
```

## Visualisation du Paysage

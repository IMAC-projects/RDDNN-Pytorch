# Reducing the Dimensionality of Data with Neural Networks

by G. E. Hinton and R. R. Salakhutdinov



## Abstract

Des données en haute dimension peuvent être un problèmes et nécessite parfois d'être représentées avec moins de dimensions pour des applications multiples comme la visualisation, accélération algorithmique  ou encore le stockage et la compression de données. 

Nous allons explorer une méthode de réduction de dimension basé sur une architecture de neurone multicouche "auto-codeurs". Des algorithmes de descente de gradient sont généralement utilisées sur ces architectures mais cela ne fonctionne bien que si les poids initiaux sont proches d'une bonne solution.

Nous allons expliquer et implémenter une solution proposée par [G. E. Hinton et R. R. Salakhutdinov](www.cs.toronto.edu/~hinton/science.pdf)  qui permet d'initialiser efficacement les poids du réseaux d'auto-codage profonds à l'aide de plusieurs couches consécutives sur le modèle des *Restricted Boltzmann Machines*.  Le réseaux pourra ensuite   affiner ses poids à l'aide d'un algorithme de descente de gradient afin d'apprendre des codes de faible dimension qui fonctionnent beaucoup qu'un réseau d'auto-codage naïf.

Nous allons ensuite comparer les résultats obtenues avec une méthode classique et largement utilisée dans ce domaine à savoir : l'analyse en composantes principales.

<div style="page-break-after: always; break-after: page;"></div>

[TOC]

<div style="page-break-after: always; break-after: page;"></div>

## Introduction

- Presentation of the problem(s). 
- Previous works (at least a few citations). If relevant, include things that you have seen during the lectures.
- 
- Contributions. Why is the studied method different/better/worse/etc. than existing previous works. 



## Analyse en composantes principales

### L'idée

TODO

### Covariance

TODO

### Vecteurs propres

TODO

### Projection des données

 exemple 3D  données projetées

### le cas en hautes dimensions

TODO

expliquer comment on doit choisir à quelle dimension arrêter (explained variance)

Graphing Variance vs. Components

### Example: MNIST dataset

TODO

## Restricted Boltzmann Machines

### L'idée

TODO

### énergie d'activation

### Apprentissage

TODO



## Solution proposée

TODO 

ajout des photo tirées du papier, unfolding multicouche, ..



## Comparaison



## Conclusion

résultats obtenus, ..


limitations, erreurs, ouvertures, améliorations, papiers plus récents sur la même problématique

## Bibliographie


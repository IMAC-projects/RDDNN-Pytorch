# Reducing the Dimensionality of Data with Neural Networks

by G. E. Hinton and R. R. Salakhutdinov

## Abstract

Des données en haute dimension peuvent être un problème et nécessitent parfois d'être représentées avec moins de dimensions pour des applications multiples comme la visualisation, l'accélération algorithmique ou encore le stockage et la compression de données. 

Une méthode classique et largement utilisée dans ce domaine est l'analyse en composantes principales et nous allons expliquer comment l'appliquer.

Nous allons ensuite explorer une méthode de réduction de dimension basée sur une architecture de neurone multicouche "auto-codeurs". Des algorithmes de descente de gradient sont généralement utilisés sur ces architectures mais cela ne fonctionne bien que si les poids initiaux sont proches d'une bonne solution.

Nous allons expliquer et implémenter une solution proposée par [G. E. Hinton et R. R. Salakhutdinov](www.cs.toronto.edu/~hinton/science.pdf) qui permet d'initialiser efficacement les poids du réseau d'auto-encodage profond à l'aide de plusieurs couches consécutives sur le modèle des *Restricted Boltzmann Machines*. Le réseaux pourra ensuite affiner ses poids à l'aide d'un algorithme de descente de gradient afin d'apprendre des codes de faible dimension qui fonctionnent beaucoup mieux qu'un réseau d'auto-encodage naïf.

Enfin, nous allons comparer les résultats obtenus et conclure sur quelle est la plus efficace des solutions présentées.

<div style="page-break-after: always; break-after: page;"></div>

[TOC]

<div style="page-break-after: always; break-after: page;"></div>

## Introduction

- Presentation of the problem(s). 

- Previous works (at least a few citations). If relevant, include things that you have seen during the lectures.

- Contributions. Why is the studied method different/better/worse/etc. than existing previous works. 


## Analyse en composantes principales

### L'idée

L’analyse en composantes principales (ACP) est une technique très populaire.

En termes simples, l'ACP consiste à effectuer une transformation de coordonnées à partir des axes arbitraires avec lesquels nos données sont exprimées vers un ensemble d'axes "alignés avec les données elles-mêmes". C'est à dire, des axes qui expriment au mieux la dispersion des informations présentes. La dispersion n'est rien d'autre que la variance ou le fait de disposer d'une "information élevée". On peut dire en d'autres termes qu'une dispersion élevée contient une information élevée.

Par conséquent, si nos données sont exprimées par des axes maximisant la représentation de l'information, il est alors possible d'omettre les axes ou les dimensions ayant une variance moindre car ces "composantes" ne participent que très peu à la description des données.
Cela peut par exemple servir à améliorer la rapidité de certains algorithmes en épargnant de nombreux calculs sans trop souffrir de perte de précision.

Selon les points de vue, on peut considérer cette analyse comme une technique descriptive où l’on essaie de résumer les données selon ses dimensions les plus importantes. Cela peut être utilisé comme une technique de visualisation où l’on essaie de préserver la "proximité" entre les individus dans un espace de représentation réduit par exemple.

Nous définirons les termes suivants au fur et à mesure, mais voici le processus résumé :

- Trouver la matrice de covariance pour l'ensemble de données en question
- Trouver les vecteurs propres de cette matrice
- Trier les vecteurs propres/"dimensions" de la plus grande à la plus petite variance
- Projection / Réduction des données : Utiliser les vecteurs propres correspondant à la plus grande variance pour projeter l'ensemble de données dans un espace à dimensions réduites
- Vérification : combien avons-nous perdu en précision dans la représentation des données ?

### Variance et covariance

Techniquement, la variance est la moyenne des différences au carré par rapport à la moyenne. Si vous connaissez bien l'écart type, généralement noté $\sigma$, la variance est juste le carré de l'écart type. 
$$
V = \sigma^2 = {1\over N}\sum_{i=0}^N (x_i - \bar{x})^2
$$
Avec la moyenne notée $\bar{x}$:
$$
\bar{x} = {1\over N}\sum_{i=0}^N x_i
$$
La variance exprime la "propagation" ou "l'étendue" des données.

Exemple d'un jeu de données 2D :

![plot2D](./src/PCA/imgs/plot2D.png)

Il est possible de calculer la variance selon les deux axes : 

| axe      | x                  | y                   |
| -------- | ------------------ | ------------------- |
| variance | 1.2526627262767291 | 0.31870756461533817 |

On peut remarquer dans le graphique ci-dessus que $x$ varie "avec" $y$ à peu près. On dit alors que $y$ est "covariant" avec $x$ . 

La covariance indique le niveau auquel deux variables varient ensemble.
Pour la calculer, c'est un peu comme la variance régulière, sauf qu'au lieu d'élever au carré l'écart par rapport à la moyenne pour une variable, nous multiplions les écarts pour les deux variables.
$$
Cov(x,y) = {1\over N-1}\sum_{j=1}^N (x_j-\bar x)(y_j-\bar y)
$$

> Remarque : la covariance d'une variable avec elle même est sa variance.

>  <span style="color:red">Remarque :</span> On divise ici par $N-1$ au lieu de $N$, donc contrairement à la variance régulière, nous ne prenons pas tout à fait la moyenne. Pour un grand ensemble de données cela ne fait essentiellement aucune différence, mais pour un petit nombre de points de données, l'utilisation de 𝑁 peut donner des valeurs qui ont tendance à être trop petites et donc le $N-1$ a été introduit pour "réduire le biais des petits échantillons".

La covariance de $x$ en fonction de $y$ vaut donc dans notre exemple $0.6250769297631616$

### Matrice de covariance

La matrice de covariance est une matrice qui regroupe la variance et covariance de chaque variable ( ou dimension) avec chaque autres et s'exprime sous la forme :
$$
\begin{pmatrix}
   Cov(x, x) & Cov(x, y) \\
   Cov(x, y) & Cov(y, y) \\
\end{pmatrix}
$$
Le long de la diagonale se trouvera la variance de chaque variable (à ${1 \over N}$ près), et le reste de la matrice sera constituée des covariances. 

> Remarque : Puisque l'ordre des variables n'a pas d'importance lors du calcul de la covariance, la matrice sera *symétrique* et sera donc *carrée*.

On obtient dans notre exemple : 
$$
\begin{pmatrix}
1.25895751 & 0.62507693 \\
0.62507693 & 0.32030911 \\
\end{pmatrix}
$$
Nous avons donc maintenant une matrice de covariance. L'étape suivante de l'ACP consiste à trouver les "composantes principales". Cela signifie les directions dans lesquelles les données varient le plus. Cela nécessite de trouver des vecteurs propres pour la matrice de covariance de notre ensemble de données.

### Vecteurs propres

Par définition, étant donné une matrice (ou "opérateur linéaire")  ${\bf A}$ de taille $n\times n$ il existe un ensemble de $n$ vecteurs $\vec{v}_i$ de sorte que la multiplication d'un de ces vecteurs par ${\bf A}$ donne un vecteur proportionnel (d'un facteur $\lambda_i$) à $\vec{v}_i$.
$$
{\bf A} \vec{v}_i = \lambda_i \vec{v}_i
$$
On appelle ces vecteurs les vecteurs propres et les $\lambda_i$ leurs valeurs propres associées.

Nous ne rentrerons pas dans le détails de comment obtenir ces vecteurs et valeurs propres ici car de nombreuses librairies permettent de le faire et beaucoup mieux que nous.

Il s'agit en général pour les cas les plus simples de suivre les étapes suivantes: 

- Trouver les valeurs propres (en résolvant le système $det( \bf{A} - \lambda \bf{I}) = 0$ )
- Pour chaque valeur propre, obtenir un système d'équations linéaires pour chaque vecteur propre et les résoudre


On obtient pour notre jeu de données :

$\lambda_1 = 1.57128949$ et $\lambda_2 = 0.00797714$

$v_1 = \begin{pmatrix}0.89454536 \\ 0.44697717\end{pmatrix}$ et $v_2 = \begin{pmatrix}-0.44697717 \\ 0.89454536\end{pmatrix}$

![plot2DEigensVectors](src\PCA\imgs\plot2DEigensVectors.png)

On remarque bien ici que le vecteur propre $v_1$ indique la direction qui maximise la variance de nos donnée. On a donc bien trouvé ici l'axe "principal" de notre jeu de données. Le deuxième vecteur propre pointe dans la direction de la plus petite variance et est orthogonal au premier vecteur.

>  Remarque : la longueur des vecteurs est exprimée à l'aide de leurs valeurs propre pour illustrer l'importance des différents axes dans la variance de nos données.

### Projection des données

Nous avons maintenant nos composantes (vecteurs propres), et nous les avons "classées" selon leur "importance". Nous allons maintenant éliminer les directions de faible variance moins importantes. En d'autres termes, nous allons projeter les données sur les différentes composantes principales de plus grande variance.

Il est possible par un changement de base d'exprimer nos données selon ces axes principaux :

![projectedData](src\PCA\imgs\EigenBaseData.png)

La matrice de covariance obtenue avec ces données exprimées dans notre nouvelle base donne : 
$$
\approx
\begin{pmatrix}
	1.57128949 & 0 \\
	0 & 0.00797714 \\
\end{pmatrix}
$$
Ce nouveau "système de coordonnées" exprime les données dans des "directions" découplées les unes des autres ($Cov(a_i,a_j) = 0 \quad\forall i \neq j$)

C'est pour cette raison que les vecteurs propres de nos données sont intéressants.

Intuitivement on se doutait bien depuis le début qu'il serait intéressant d'exprimer nos données selon un unique axe, puisque elles s'apparentent grossièrement à une droite.

---

Pour effectuer la projection il va nous suffire de réduire à zéro la dimension de plus petite variance, ce qui reviens à effectué un "changement de base" avec seulement les vecteurs de plus grande variance (dans notre cas, un seul vecteur).

On obtient alors les données suivantes : 

![projectedData](src\PCA\imgs\projectedData.png)

On peut ramener ensuite nos données dans notre base d'origine pour comparer avec nos données d'origine. Cette étape est utilisée uniquement afin de comparer notre projection dans le même "système de coordonnées" que nos données d'origine ou autrement dit dans la même dimension dans le sens où il s'agit ici d'ajouter une dimension "superflue" à nos données ainsi réduite.

![projectedData](src\PCA\imgs\backProjectedData.png)

### Le cas réel en hautes dimensions

L'ACP est généralement utilisée pour éliminer beaucoup plus de dimensions qu'une seule (comme dans notre exemple 2D). Elle est souvent utilisée pour la visualisation des données, mais aussi pour la réduction des caractéristiques, c'est-à-dire pour envoyer moins de données dans votre algorithme d'apprentissage afin d'améliorer ses performances.

Dans notre cas 2D il était évident qu'une seule direction ou dimension était intéressante pour exprimer nos données ; mais dans les cas de grande dimension, comment savoir combien de dimensions omettre ? En d'autres termes, combien de "composants" doit-on conserver lors de l'ACP ?

Il y a plusieurs façons de faire ce choix. Il faudra généralement faire un compromis entre la précision et la vitesse de calcul. On peut par exemple exprimer sur un graphique la variance des données en fonction du nombre de composantes que l'on garde.

Abordons pour finir cette partie sur l'ACP un exemple plus concret illustrant cette méthode de sélection et un cas "réel" d'application de l'ACP.

### Example: MNIST dataset

Prenons un ensemble d'images de 28x28 pixels représentant des chiffres manuscrits et appliquons l'analyse en composantes principales sur ces images considérées comme des vecteurs de 784 composantes (dans un espace de 784 dimensions donc).

![MnistExamples](src\PCA\imgs\MnistExamples.png)

Après calcul des vecteurs et valeurs propres de nos données, traçons le graphe des valeurs propres rangées par ordre décroissant :

![eigensValuesByComponents](src\PCA\imgs\eigensValuesByComponents.png)

On aperçoit ici que les valeurs des valeurs propres décroissent très rapidement. On peut interpréter cela comme une décroissance très rapide de la variances de nos données expliquées par les composantes ; ou autrement dit, "peu" de composantes représentent une partie significative de la variance de nos données. 

> Rappel : plus la valeur propre d'un vecteur propre est grande plus la variance expliquée par ce vecteur est grande

Exprimons maintenant ce même graphique en valeurs cumulées et en normalisant les valeurs propres.

![cumulativeExplainedVariance](src\PCA\imgs\cumulativeExplainedVariance.png)

On obtient alors le  graphe de la variance cumulative expliquée en fonction du nombre de composantes.
Ce graphe permet alors de connaître pour un nombre de composantes donné le pourcentage de variance des données expliqué, ce qui peut s'interpréter comme un pourcentage de précision à représenter les données d'origine avec ce nombre de composantes.
On voit par exemple qu'avec seulement 330 composantes il est possible de représenter à 99% nos données d'origine.

---

Pour conclure avec cet exemple, voici un tableau de différents chiffres projetés avec différents niveaux de variance expliquée (et le nombre de composantes retenues).
Les chiffres sous les images correspondent à trois mesures permettant l'évaluation de la qualité de projection des images ; à savoir de haut en bas : le rapport signa-bruit (PSNR), l'erreur quadratique moyenne (MSE) et la structural similarity (SSIM).

![MnistEvaluation](src\PCA\imgs\MnistEvaluation.png)

## Restricted Boltzmann Machines

Inventées par Geoffrey Hinton, les machines de Boltzmann restreintes sont des réseaux neuronaux peu profonds à deux couches qui constituent les éléments de base des réseaux profonds.

### L'idée

Une RBM est utilisée pour avoir une estimation de la distribution probabiliste d'un jeu de données.

La première couche de la RBM est appelée la couche visible, ou couche d'entrée, et la seconde est la couche cachée.

### Énergie d'activation

On définit l'énergie d'activation d'une machine de Boltzmann restreinte par la formule suivante :
$$
E = -\sum_i b_iv_i - \sum_j c_jh_j - \sum_{i,j} w_{ij}v_ih_j
$$
avec :

- $w_{ij}$ le poids entre le neurone $j$ et le neurone $i$
- $v_i$ l'état du neurone $i$ de la couche visible
- $h_j$ l'état du neurone $j$ de la couche cachée (hidden)
- $b_i$ et $c_j$ les biais des neurones $v_i$ et $h_j$ d'entrée et de sortie

Cette énergie est interprétée comme l'énergie d'un système physique et on peut définir le score d'une configuration énergétique comme l'inverse de cette énergie. Plus l'énergie est faible (stabilité) plus le score est élevé.
$$
\text{Score} = - E
$$

### Probabilité

Considérons des scores donnés pour des configurations possibles de notre système.
$$
[ 0, 1, 2, 5, 8]
$$
Il est intéressant d'exprimer le score de ces configurations en probabilités et le moyen le plus classique est de normaliser ces scores par la somme de tous les scores :

![scoresProba](src\RBM\imgs\scoresProba.png)

Cependant des problèmes peuvent survenir avec des scores négatifs car ils peuvent se compenser et la somme peut alors être nulle.

Un moyen naturel peut être de passer à l'exponentielle puis de normaliser, c'est ce qu'on appelle l'opération de softmax.

![softmax](src\RBM\imgs\softmax.png)

C'est ainsi que l'on peut définir la probabilité d'avoir une certaine configuration entrée-sortie $(v_i, h_j)$ :
$$
P(v_i, h_j) = {e^{-E(v_i, h_j)} \over Z}
$$
où $Z$ est la constante de normalisation.

### Positionnement du problème

L'idée générale est de modifier les poids $w_{ij}$ pour approcher au mieux la distribution de probabilité de nos données.
C'est différent d'un algorithme plus classique comme une régression par exemple, qui estime une valeur continue basée sur de nombreuses entrées.

Imaginons que les données d'entrée et les reconstructions soient des courbes normales de formes différentes, qui ne se chevauchent que partiellement.

En ajustant itérativement les poids en fonction de l'erreur qu'ils produisent ou de leurs scores, une RBM apprend à se rapprocher des données originales en mimant la distribution de probabilité des données d'origine dans les données de la couche cachée. On pourrait dire que les poids en viennent lentement à refléter la structure de l'entrée qui est encodée dans la couche cachée.

Considérons un exemple simple dans lequel une personne dispose de trois accessoires :

Une paire de lunette (notée **L**), un parapluie (noté **P**) et une caméra (notée **C**)

Regardons ce qu'elle décide ou non de prendre lorsqu'elle sort de chez elle en fonction des jours de la semaine :

| jour | Lunettes :eyeglasses: | Parapluie :closed_umbrella: | Caméra :camera:    |
| ---- | --------------------- | --------------------------- | ------------------ |
| 0    | :heavy_check_mark:    | :x:                         | :heavy_check_mark: |
| 1    | :x:                   | :heavy_check_mark:          | :x:                |
| 2    | :heavy_check_mark:    | :x:                         | :heavy_check_mark: |
| 3    | :heavy_check_mark:    | :x:                         | :heavy_check_mark: |
| 4    | :x:                   | :heavy_check_mark:          | :x:                |
| 5    | :x:                   | :heavy_check_mark:          | :x:                |
| 6    | :x:                   | :heavy_check_mark:          | :heavy_check_mark: |
| 7    | :heavy_check_mark:    | :x:                         | :heavy_check_mark: |
| 8    | :heavy_check_mark:    | :x:                         | :heavy_check_mark: |
| 9    | :x:                   | :heavy_check_mark:          | :x:                |

Ces données vont constituer nos données d'entrée de la couche d'input.

Considérons maintenant qu'il existe deux facteurs qui pourraient expliquer ces données : la présence ou non de soleil :high_brightness: (noté **S**) et d'averse :sweat_drops: ​(noté **A**) au cours de la journée. ​

Ces facteurs vont constituer la couche cachée de notre système.

Initialisons maintenant tous nos poids à 0 et traçons la probabilité de chaque configuration d'entrée-sortie.

> Nous ignorerons les biais associés aux entrées et sorties dans notre exemple pour plus de simplicité.

> Une configuration est notée avec une suite de lettres montrant la présence ou non de l'accessoire et d'un évènement météorologique.

![uniformProbabilities](src\RBM\imgs\uniformProbabilities.png)

On constate évidement que toutes les configurations sont équiprobables (car tous nos poids sont nuls et donc l'énergie de chaque configuration est nulle)

On aimerait se rapprocher d'une configuration de probabilités qui représente nos données, c'est-à-dire qui exprime les formations possibles de notre jeu de données.

Dans notre cas voici les configurations qui apparaissent tout au long de la semaine concernant les inputs : 

| jour          | 0                     | 1    | 2                     | 3                     | 4                 | 5                 | 6                         | 7                     | 8                     | 9                 |
| ------------- | --------------------- | ---- | --------------------- | --------------------- | ----------------- | ----------------- | ------------------------- | --------------------- | --------------------- | ----------------- |
| configuration | :eyeglasses: :camera: | :    | :eyeglasses: :camera: | :eyeglasses: :camera: | :closed_umbrella: | :closed_umbrella: | :closed_umbrella::camera: | :eyeglasses: :camera: | :eyeglasses: :camera: | :closed_umbrella: |

En incluant les configurations possibles de notre couche cachée, il est possible de lister ainsi toutes les configurations (entrée et sortie) envisageables:

- :eyeglasses: :camera:
- :eyeglasses: :camera::sweat_drops:
- :eyeglasses: :camera::high_brightness:
-  : ::: 
- :closed_umbrella:
- :closed_umbrella::sweat_drops:
- :closed_umbrella::high_brightness:
-  :closed_umbrella::high_brightness::sweat_drops: 
- :closed_umbrella: :camera:
- :closed_umbrella: :camera::sweat_drops:
- :closed_umbrella: :camera::high_brightness:
- :closed_umbrella: :camera::high_brightness::sweat_drops: 

On aimerait donc trouver les poids de notre algorithme pour que les configurations ou évènements envisageables aient une grande probabilité et les autres une faible probabilité.

Cela donnerait quelque chose comme cela pour notre exemple :

![exempleWantedProbabilities](src\RBM\imgs\exempleWantedProbabilities.png)

> L'évènement :eyeglasses: :camera: (LC) a lieu 5 fois dans la semaine. Il est donc normal que cette configuration soit plus probable que l'évènement :closed_umbrella: (P) (qui a lieu 3 fois) ou que l'évènement :closed_umbrella: :camera: (PC) (qui a lieu une unique fois).

On peut reformuler cela mathématiquement dans le sens où on cherche à maximiser le produit des probabilités des évènements probables de notre jeu de donnée.
$$
arg \; \underset{W}{max}\;\underset{v \in V}{\Pi}P(v)
$$


### Apprentissage

####  *Contrastive Divergence*



## Solution proposée

TODO 

ajout des photo tirées du papier, unfolding multicouche, ..



## Comparaison



## Conclusion

> Learning with examples first is always better than starting with math

résultats obtenus, ..

limitations, erreurs, ouvertures, améliorations, papiers plus récents sur la même problématique



Bien que les RBM soient parfois utilisés, ils sont peu à peu ont dépréciés au profit de réseaux adversaires générateurs (GAN) ou d'auto-codeurs variationnels(VAE) .

## Bibliographie

Vecteur propres : 

[3Blue1Brown]: https://www.youtube.com/watch?v=PFDu9oVAE-g	"Les vecteurs propres et valeurs propres"


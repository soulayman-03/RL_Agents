# Modélisation Mathématique du Système (Pour Article Scientifique)

Cette section décrit formellement le problème d'allocation de tâches (Task Offloading) dans notre système IoT hétérogène sous forme d'un problème d'optimisation.

## 1. Notations et Variables du Système

Soit $\mathcal{N} = \{1, 2, ..., N\}$ l'ensemble des agents (tâches) et $\mathcal{M} = \{1, 2, ..., M\}$ l'ensemble des dispositifs IoT. 
Pour un agent $n \in \mathcal{N}$, sa tâche d'inférence est composée d'un graphe séquentiel de couches (layers) de Réseau de Neurones Profond (DNN) noté $\mathcal{L}_n = \{1, 2, ..., L_n\}$.

Chaque couche $l \in \mathcal{L}_n$ est caractérisée par :
- $C_{n,l}$ : Demande en puissance de calcul (ex: GFLOPS).
- $Mem_{n,l}$ : Demande en mémoire vive (RAM).
- $D_{n,l}$ : Taille des données de sortie à transmettre à la couche suivante.
- $PL_{n,l}$ : Niveau d'exigence en matière de confidentialité (Privacy Level).

Chaque dispositif $m \in \mathcal{M}$ est caractérisé par :
- $F_m$ : Fréquence / Vitesse du processeur (CPU speed).
- $BW_m$ : Bande passante réseau disponible en transmission.
- $C^{mem}_m$ : Capacité maximale de mémoire utilisable.
- $B_m$ : Budget énergétique initial disponible pour l'épisode.
- $P_m^{comp}, P_m^{comm}$ : Puissance nominale consommée lors du calcul et de la communication (en Watts).
- $TR_m \in [0, 1]$ : Score de confiance ou niveau d'accréditation du dispositif.

### Variable de Décision
Le problème d'allocation est modélisé comme une variable de décision booléenne $x_{n,l}^m \in \{0, 1\}$ définie par :
$$
x_{n,l}^m = 
\begin{cases} 
1 & \text{si la couche } l \text{ de l'agent } n \text{ est exécutée sur le dispositif } m \\
0 & \text{sinon}
\end{cases}
$$

---

## 2. Formulation des Modèles de Coût

### 2.1 Modèle de Latence et File d'Attente (Queuing Model)
Le système opère comme un environnement synchrone à intervalles de temps (*Synchronous Time-Slotted System*). À chaque étape temporelle $t$, si un goulot d'étranglement se forme sur un dispositif $m$ ciblé par plusieurs agents (collision synchrone), un modèle de délai d'attente cumulatif (*Queuing Delay*) est utilisé.

Soit une fonction $\mathcal{O}(n)$ définissant l'ordre de traitement des agents (rendu aléatoire à chaque étape temporelle pour garantir l'équité d'accès au canal). Si l'agent $n$ cible le dispositif $m$, son délai d'attente au processeur $W_{comp}^m$ (et au réseau $W_{comm}^m$) est la somme stricte des temps de service de tous les agents l'ayant devancé vers ce même dispositif :

$$ W_{comp}^m(n, t) = \sum_{\{k \in \mathcal{N} \mid x_{k,l}^m=1, \, \mathcal{O}(k) < \mathcal{O}(n)\}} \frac{C_{k,l}}{F_m} $$
$$ W_{comm}^m(n, t) = \sum_{\{k \in \mathcal{N} \mid x_{k,l}^m=1, \, m' \neq m, \, \mathcal{O}(k) < \mathcal{O}(n)\}} \frac{D_{k,l-1}}{BW_{m}} $$

Ainsi, la latence réelle perçue par l'agent $n$ inclut ce temps d'attente subi causé par les agents concurrents :

**Temps de calcul perçu :**
$$ t_{comp}(n, l, m) = \frac{C_{n,l}}{F_m} + W_{comp}^m(n, t) $$

**Temps de communication perçu :**
Si la couche précédente $l-1$ a été exécutée sur un dispositif différent ($m' \neq m$), un transfert est requis à travers la bande passante du dispositif de réception $m$ :
$$ t_{comm}(n, l, m', m) = 
\begin{cases} 
0 & \text{si } m = m' \text{ (traitement local)} \\
\frac{D_{n,l-1}}{BW_{m}} + W_{comm}^{m}(n, t) & \text{si } m \neq m'
\end{cases}
$$

La latence totale de l'épisode pour l'agent $n$ sur la totalité du graphe d'inférence est la somme des retards successifs :
$$ T_n = \sum_{l \in \mathcal{L}_n} \sum_{m \in \mathcal{M}} x_{n,l}^m \left( t_{comp}(n, l, m) + \sum_{m' \in \mathcal{M}} x_{n,l-1}^{m'} t_{comm}(n, l, m', m) \right) $$

*(Note: Pour la toute première couche, $t_{comm}$ modélise la latence d'envoi initial des données vers le dispositif de calcul).*

### 2.2 Modèle de Consommation Énergétique
L'énergie dissipée par chaque dispositif est directement proportionnelle à ses spécifications matérielles. L'énergie coûte lors du traitement et de la réception/transmission :
$$ E_{comp}(n, l, m) = P_m^{comp} \times t_{comp}(n, l, m) $$
$$ E_{comm}(n, l, m', m) = P_m^{comm} \times t_{comm}(n, l, m', m) $$

L'énergie totale consommée par le dispositif $m$ au cours d'un pas de temps ou de l'épisode est :
$$ E_m^{total} = \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \left( E_{comp}(n, l, m) + \sum_{m'} x_{n,l-1}^{m'} E_{comm}(n, l, m', m) \right) $$

L'énergie totale consommée par le réseau pour satisfaire la tâche complète de l'agent $n$ est notée $E_n$.

---

## 3. Formulation de la Fonction Objectif

L'objectif du système d'Apprentissage par Renforcement Multi-Agent (MADDPG) est de trouver la politique conjointe d'allocation optimale minimisant simultanément la latence d'inférence globale et la consommation énergétique du réseau.

$$ \min_{\mathbf{X}} \mathcal{J}(\mathbf{X}) = \sum_{n \in \mathcal{N}} \left( \alpha T_n + \beta E_n \right) $$

*(Dans notre implémentation, $\alpha = 1$ et $\beta = 1$, reflétés par la constante $R = -(T_{total} + E_{cost})$ en guise de récompense d'agent).*

Cette minimisation est rigoureusement sujette aux contraintes dures ("Hard Constraints") suivantes.

---

## 4. Contraintes Mathématiques ("Hard Constraints")

**Contrainte C1 : Allocation Unique**
Chaque couche d'un agent doit être assignée et exécutée par un et un seul dispositif :
$$ \sum_{m \in \mathcal{M}} x_{n,l}^m = 1 \quad \forall n \in \mathcal{N}, \forall l \in \mathcal{L}_n $$

**Contrainte C2 : Capacité de la Mémoire Vive**
À chaque étape de calcul, la somme des capacités mémoires requises par l'ensemble des couches en cours d'exécution sur le dispositif $m$ ne doit pas excéder sa RAM disponible :
$$ \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \times Mem_{n,l} \leq C^{mem}_m \quad \forall m \in \mathcal{M} $$

**Contrainte C3 : Budget Énergétique Strict**
L'énergie totale consommée par un dispositif pour traiter l'intégralité des tâches qu'il héberge ne doit pas excéder le budget énergétique dont il dispose (prévenant ainsi son extinction critique) :
$$ E_m^{total} \leq B_m \quad \forall m \in \mathcal{M} $$

**Contrainte C4 : Confidentialité et Modèle de Confiance**
Le dispositif $m$ ne peut héberger une tâche classée comme confidentielle que si son score de confiance (Trust Score) excède strictement le ratio requis par le profil de confidentialité de la couche en cours d'analyse :
$$  TR_m \geq TR_{min} \times \frac{PL_{n,l}}{PL_{max}} \quad \text{si } x_{n,l}^m = 1 \quad \forall n \in \mathcal{N}, \forall l \in \mathcal{L}_n, \forall m \in \mathcal{M} $$

**Contrainte C5 : Diversité Séquentielle (Optionnel)**
Si configurée, la contrainte de diversité interdit de placer plus de $\kappa$ couches consécutives de la même tâche sur le même nœud, de façon à prévenir l'étranglement ou la monopolisation de ressources singulières :
$$ \sum_{k=l}^{l+\kappa} x_{n,k}^m \leq \kappa \quad \forall n \in \mathcal{N}, \forall m \in \mathcal{M} $$

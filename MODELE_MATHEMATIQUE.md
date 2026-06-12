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

L'énergie dissipée par chaque dispositif dépend de ses spécifications matérielles et de la technique d'ajustement dynamique de la fréquence et de la tension (DVFS). Soit $f_m$ la fréquence effective du processeur, telle que $f_m = F_m \times \text{dvfs\_ratio}$.

La puissance consommée lors du calcul suit un modèle physique cubique basé sur la fréquence d'horloge :
$$ P_m^{comp}(f_m) = \kappa \cdot (f_m)^3 $$
où $\kappa$ représente le facteur de capacitance et d'activité du circuit.

Ainsi, l'énergie consommée lors du traitement et de la réception/transmission est :
$$ E_{comp}(n, l, m) = P_m^{comp}(f_m) \times t_{comp}(n, l, m) $$
$$ E_{comm}(n, l, m', m) = P_m^{comm} \times t_{comm}(n, l, m', m) $$

L'énergie totale consommée par le dispositif $m$ au cours d'un pas de temps ou de l'épisode est :
$$ E_m^{total} = \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \left( E_{comp}(n, l, m) + \sum_{m'} x_{n,l-1}^{m'} E_{comm}(n, l, m', m) \right) $$

L'énergie totale consommée par le réseau pour satisfaire la tâche complète de l'agent $n$ est notée $E_n$.

---


La puissance consommée lors du calcul suit un modèle physique cubique basé sur la fréquence d'horloge :
$$ P_m^{comp}(f_m) = \kappa \cdot (f_m)^3 $$
où $\kappa$ représente le facteur de capacitance et d'activité du circuit.

Ainsi, l'énergie consommée lors du traitement et de la réception/transmission est :
$$ E_{comp}(n, l, m) = P_m^{comp}(f_m) \times t_{comp}(n, l, m) $$
$$ E_{comm}(n, l, m', m) = P_m^{comm} \times t_{comm}(n, l, m', m) $$

L'énergie totale consommée par le dispositif $m$ au cours d'un pas de temps ou de l'épisode est :
$$ E_m^{total} = \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \left( E_{comp}(n, l, m) + \sum_{m'} x_{n,l-1}^{m'} E_{comm}(n, l, m', m) \right) $$

L'énergie totale consommée par le réseau pour satisfaire la tâche complète de l'agent $n$ est notée $E_n$.

---
L'énergie dissipée par chaque dispositif dépend de ses spécifications matérielles et de la technique d'ajustement dynamique de la fréquence et de la tension (DVFS). Soit $f_m$ la fréquence effective du processeur, telle que $f_m = F_m \times \text{dvfs\_ratio}$.

La puissance consommée lors du calcul suit un modèle physique cubique basé sur la fréquence d'horloge :
$$ P_m^{comp}(f_m) = \kappa \cdot (f_m)^3 $$
où $\kappa$ représente le facteur de capacitance et d'activité du circuit.

Ainsi, l'énergie consommée lors du traitement et de la réception/transmission est :
$$ E_{comp}(n, l, m) = P_m^{comp}(f_m) \times t_{comp}(n, l, m) $$
$$ E_{comm}(n, l, m', m) = P_m^{comm} \times t_{comm}(n, l, m', m) $$

L'énergie totale consommée par le dispositif $m$ au cours d'un pas de temps ou de l'épisode est :
$$ E_m^{total} = \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \left( E_{comp}(n, l, m) + \sum_{m'} x_{n,l-1}^{m'} E_{comm}(n, l, m', m) \right) $$

L'énergie totale consommée par le réseau pour satisfaire la tâche complète de l'agent $n$ est notée $E_n$.

---

## 3. Formulation de la Fonction Objectif

L'objectif du système d'Apprentissage par Renforcement Multi-Agent (MADDPG) est de trouver la politique conjointe d'allocation optimale minimisant simultanément la latence d'inférence globale et la consommation énergétique du réseau.

$$ \min_{\mathbf{X}} \mathcal{J}(\mathbf{X}) = \sum_{n \in \mathcal{N}} \left( \alpha T_n + \beta E_n \right) $$

*(Dans notre implémentation, $\alpha$ et $\beta$ sont des paramètres configurables (par ex. $\alpha = 0.4$, $\beta = 0.6$), reflétés par la récompense d'agent $R = -(\alpha T_{total} + \beta E_{cost})$, ce qui permet d'explorer finement le compromis Latence-Énergie).*

Cette minimisation est rigoureusement sujette aux contraintes dures ("Hard Constraints") suivantes.

---

## 4. Contraintes Mathématiques ("Hard Constraints")

**Contrainte C1 : Allocation Unique**
Chaque couche d'un agent doit être assignée et exécutée par un et un seul dispositif :
$$ \sum_{m \in \mathcal{M}} x_{n,l}^m = 1 \quad \forall n \in \mathcal{N}, \forall l \in \mathcal{L}_n $$

**Contrainte C2 : Capacité de la Mémoire Vive**
La somme des capacités mémoires requises par l'ensemble des couches en cours d'exécution sur le dispositif $m$ ne doit pas excéder sa RAM disponible :
$$ \sum_{n \in \mathcal{N}} \sum_{l \in \mathcal{L}_n} x_{n,l}^m \times Mem_{n,l} \leq C^{mem}_m \quad \forall m \in \mathcal{M} $$

**Contrainte C3 : Budget Énergétique Strict**
L'énergie totale consommée par un dispositif pour traiter l'intégralité des tâches qu'il héberge ne doit pas excéder le budget énergétique initial dont il dispose :
$$ E_m^{total} \leq B_m \quad \forall m \in \mathcal{M} $$

**Contrainte C4 : Confidentialité et Modèle de Confiance**
Un dispositif $m$ ne peut exécuter une couche $l$ que s'il respecte deux critères de sécurité cumulatifs :
1. Son niveau d'accréditation (Privacy Clearance) doit être suffisant : $PrivacyClearance_m \geq PL_{n,l}$
2. Son score de confiance (Trust Score) doit satisfaire le seuil requis pour ce niveau de confidentialité :
$$ TrustScore_m \geq TR_{min} \times \frac{PL_{n,l}}{PL_{max}} $$

**Contrainte C5 : Limites d'Admission par Étape Temporelle (Compute & Bandwidth)**
Pour éviter l'engorgement soudain, la charge allouée à un dispositif $m$ lors d'une même étape temporelle $t$ est limitée par sa fréquence et sa bande passante :
$$ \sum_{\text{allocations à } t} C_{n,l} \leq \gamma_{comp} \times F_m \quad (\text{avec } \gamma_{comp} = 10.0) $$
$$ \sum_{\text{allocations à } t, m' \neq m} D_{n,l-1} \leq \gamma_{bw} \times BW_m \quad (\text{avec } \gamma_{bw} = 5.0) $$

**Contrainte C6 : Exposition Maximale (Security Level Exposure)**
Afin de minimiser l'exposition d'un modèle entier à un seul nœud, un dispositif ne peut traiter plus qu'une fraction maximale $S_l$ des couches totales de la tâche de l'agent $n$ :
$$ \sum_{l \in \mathcal{L}_n} x_{n,l}^m \leq \max(1, \lfloor L_n \times S_l \rfloor) \quad \forall n \in \mathcal{N}, \forall m \in \mathcal{M} $$

**Contrainte C7 : Diversité Séquentielle (Optionnelle)**
Si activée, cette contrainte interdit l'exécution de deux couches successives de la même tâche sur le même dispositif, forçant ainsi la distribution du calcul :
$$ x_{n,l}^m + x_{n,l+1}^m \leq 1 \quad \forall n \in \mathcal{N}, \forall l \in \mathcal{L}_n, \forall m \in \mathcal{M} $$

## 📄 Pseudocode de l'Algorithme MADDPG

```text
# Initialise pour chaque agent i = 1,…,N
    Initialise l'acteur μ_{θ_i} et le critique Q_{φ_i} avec des poids aléatoires
    Initialise les réseaux cibles μ_{θ'_i} ← μ_{θ_i} et Q_{φ'_i} ← Q_{φ_i}
    Initialise le replay buffer D

Pour chaque épisode = 1 à M :
    Initialise le processus de bruit N (ex. Ornstein‑Uhlenbeck) pour l'exploration
    Observe l'état initial x = {o_1,…,o_N}
    Pour chaque pas t = 1 à max_steps :
        Pour chaque agent i :
            a_i = μ_{θ_i}(o_i) + N_t   # action avec exploration
        Exécute les actions a = {a_1,…,a_N} dans l'environnement
        Observe les récompenses r = {r_1,…,r_N} et le nouvel état x' = {o'_1,…,o'_N}
        Stocke la transition (x, a, r, x') dans D
        x ← x'
        Pour chaque agent i :
            Échantillonne un mini‑lot S depuis D
            y_i = r_i + γ Q_{φ'_i}(x', a'_1,…,a'_N) |_{a'_k = μ_{θ'_k}(o'_k)}
            Met à jour le critique en minimisant
                L(φ_i) = (1/|S|) Σ (y_i - Q_{φ_i}(x, a_1,…,a_N))^2
            Met à jour l'acteur par le gradient de politique
                ∇_{θ_i} J ≈ (1/|S|) Σ ∇_{θ_i} μ_{θ_i}(o_i) ∇_{a_i} Q_{φ_i}(x, a_1,…,a_i,…,a_N)
        Pour chaque agent i :
            Mise à jour douce des réseaux cibles
                θ'_i ← τ θ_i + (1-τ) θ'_i
                φ'_i ← τ φ_i + (1-τ) φ'_i
```

Ce pseudocode reprend les étapes clés décrites dans les ressources standards : initialisation des acteurs/critics, boucle d'épisodes, collecte d'expériences, mise à jour des réseaux critiques et acteurs, et mise à jour des cibles.

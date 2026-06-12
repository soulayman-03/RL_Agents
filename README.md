# 🤖 Optimisation du Computation Offloading dans les Réseaux MEC avec MARL

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge)

**PFE — Université Moulay Slimane, FST | Département Informatique | 2024–2025**

Réalisé par **Ait Ahmed Oulhaj Soulaimane** · Encadré par **Pr. Ait Omar Driss**

</div>

---

## 📋 Table des Matières

- [Contexte et Problématique](#-contexte-et-problématique)
- [Architecture du Système](#-architecture-du-système)
- [Modélisation](#-modélisation)
- [Approche MARL](#-approche-marl)
- [Structure du Projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Références](#-références)

---

## 🎯 Contexte et Problématique

Avec l'essor de l'IoT et des applications critiques (véhicules autonomes, réalité augmentée, surveillance intelligente), les dispositifs embarqués doivent exécuter des modèles de **Deep Learning** sous des contraintes strictes de **latence**, d'**énergie** et de **sécurité**.

Les architectures Cloud traditionnelles introduisent des délais de transmission élevés et une forte dépendance réseau. Le paradigme **Edge Computing** rapproche le calcul des sources de données, mais la gestion optimale des ressources dans des environnements hétérogènes et dynamiques reste un problème ouvert.

### Défis principaux

| Défi | Description |
|------|-------------|
| ⚡ Ressources limitées | Faible CPU, mémoire et batterie sur les dispositifs IoT |
| ⏱️ Latence | Contraintes temps-réel pour les applications critiques |
| 🔒 Sécurité & Confidentialité | Risque d'exposition des données sur des nœuds non fiables |
| 🌐 Hétérogénéité | Capacités variables entre dispositifs IoT, Edge et Cloud |
| 📈 Scalabilité | Coordination complexe entre agents multiples |

---

## 🏗️ Architecture du Système

Le système modélise un environnement de **calcul distribué à trois niveaux** :

```
┌─────────────────────────────────────────────┐
│                  CLOUD                       │
│         (Haute puissance de calcul)          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              EDGE NODES                      │
│   (Cloudlets, Stations de base, Micro-DC)    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            EDGE DEVICES (IoT)                │
│   Agents générateurs de tâches + Nœuds de   │
│          calcul contraints                   │
└─────────────────────────────────────────────┘
```

Chaque dispositif `m ∈ M` est caractérisé par :

| Paramètre | Symbole | Description |
|-----------|---------|-------------|
| Fréquence CPU | `F_m` | Puissance de traitement |
| Bande passante | `BW_m` | Débit réseau disponible |
| Capacité mémoire | `C_m^mem` | RAM disponible |
| Budget énergétique | `B_m` | Énergie restante (batterie) |
| Score de confiance | `sl_m` | Niveau de sécurité du nœud |

---

## 📐 Modélisation

### Modélisation des tâches

Chaque agent `n ∈ N` génère une tâche d'inférence modélisée comme un pipeline séquentiel de couches CNN :

```
Lₙ = { 1, 2, ..., L }
```

Chaque couche `l` est caractérisée par :
- **Coût de calcul** `c_{n,l}` (FLOPS) — calculé selon le type de couche :
  - Couche convolutionnelle : `c_{n,l} = n_{l-1} · S_l · n_l · o_l`
  - Couche fully-connected : `c_{n,l} = n*_{l-1} · n*_l`
- **Besoin mémoire** `m_{n,l} = W_{n,l} · b`
- **Taille de sortie** `D_{n,l}` (coût de communication en cas d'offloading)
- **Niveau de confidentialité** `σ_{n,l}`

### Modèle de Latence

La latence totale d'une tâche intègre les délais de calcul, de communication et de file d'attente :

```
Tₙ = Σ_{l,m} A^m_{n,l} · [ t_comp(n,l,m) + Σ_{m'} A^m'_{n,l-1} · t_comm(n,l,m',m) ]
```

Avec le délai d'attente (file d'attente multi-agent) :
```
W^m_comp(n,t) = Σ_{n'≠n, l'} A^m_{n',l'} · C_{n',l'} / f_m
```

### Modèle Énergétique avec DVFS

Le **Dynamic Voltage and Frequency Scaling (DVFS)** permet d'ajuster dynamiquement la fréquence effective :

```
f_eff = f_m · ρ,   ρ ∈ (0, 1]
```

La puissance de calcul suit un modèle cubique (circuits CMOS) :
```
P_comp = κ · f³
```

Ce modèle introduit un **compromis fondamental** :
> Diviser la fréquence par 2 → latence ×2, mais puissance ÷8 → énergie ÷4

### Problème d'Optimisation

```
min_a  Σ_{n∈N} ( α·Tₙ + β·Eₙ )
```

Sous les contraintes :

| Contrainte | Description |
|-----------|-------------|
| Affectation unique | Chaque couche exécutée sur un seul dispositif |
| Mémoire | `Σ A^m_{n,l} · m_{n,l} ≤ C_m^mem` |
| Capacité de calcul | `Σ A^m_{n,l} · c_{n,l} ≤ C_m` |
| Énergie | `E_m^total ≤ B_m` |
| Confidentialité | Score de confiance du nœud suffisant |
| Sécurité (exposition) | Max `⌊L_n · S⌋` couches par nœud (anti white-box) |

> Ce problème est de type **MINLP (NP-difficile)** — résolu via MARL.

---

## 🧠 Approche MARL

### Pourquoi MARL ?

Les méthodes classiques d'optimisation sont inadaptées aux environnements dynamiques et temps-réel. Le **Multi-Agent Reinforcement Learning** permet :

- Une **prise de décision distribuée** : chaque IoT agent prend ses propres décisions
- Une **adaptabilité** aux variations de ressources et de charge réseau
- Une gestion simultanée de **multiples contraintes** (latence, énergie, sécurité)

### Paradigme CTDE

Le projet adopte le paradigme **Centralized Training with Decentralized Execution (CTDE)** :

```
Phase d'entraînement (centralisée)
  └── Accès à l'état global du système
  └── Partage de paramètres entre agents
  └── Experience Buffer commun

Phase d'exécution (décentralisée)
  └── Chaque agent agit avec ses observations locales
  └── Pas de communication inter-agents requise
```

### Formulation MDP

| Composant | Description |
|-----------|-------------|
| **État** `s` | Ressources disponibles, charge CPU/réseau, énergie, files d'attente |
| **Action** `a` | Allocation des couches + ratio DVFS `ρ` |
| **Récompense** `r` | `-( α·Tₙ + β·Eₙ )` sous respect des contraintes |
| **Politique** `π` | Apprise via Deep RL (DQN / Actor-Critic) |

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/soulayman-03/RL_Agents.git
cd RL_Agents

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**
```
torch>=2.0
numpy
gymnasium
matplotlib
pandas
```

---

## 🚀 Utilisation

### Entraînement

```bash
python experiments/train.py --agents 5 --episodes 1000 --alpha 0.5 --beta 0.5
```

### Évaluation

```bash
python experiments/evaluate.py --model results/best_model.pth
```

### Paramètres clés

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--agents` | Nombre d'agents IoT `N` | `5` |
| `--devices` | Nombre de dispositifs `M` | `10` |
| `--alpha` | Poids latence dans la récompense | `0.5` |
| `--beta` | Poids énergie dans la récompense | `0.5` |
| `--dvfs` | Activer DVFS | `True` |
| `--security` | Paramètre sécurité `S` | `0.5` |

---

## 📊 Résultats Expérimentaux et Analyse

Cette section présente les résultats détaillés de la phase d'évaluation à travers trois tests clés du paramètre d'activité capacitive $\kappa$ ainsi que l'impact de la contrainte de **Diversité Séquentielle** dans l'environnement DVFS équilibré.

### 📈 Tableau Récapitulatif Global

Voici les performances comparées des différents modèles après 5000 épisodes d'entraînement dans des conditions identiques (7 agents, 5 terminaux hétérogènes) :

| Modèle / Configuration | Paramètre $\kappa$ | Diversité Séquentielle | Taux de Succès | Épisodes en Échec (Stalls) | Étapes Totales (Env Steps) | Observations Clés |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Test 1 : DVFS Physique Initial** | $10^{-4}$ (fixe) | Désactivée | **82.60%** | 870 | 93 456 | Consommation excessive à haute fréquence; les agents forcent le DVFS au minimum au détriment de la latence. |
| **Test 2 : DVFS Équilibré** | $8 \times 10^{-6}$ (fixe) | Désactivée | **98.06%** | 97 | 94 912 | Calibrage optimal ($P_{max} = 1.0\text{ W}$); excellent compromis latence/énergie. |
| **Test 3 : DVFS Alignement Dynamique**| Dynamique $\kappa_m$ | Désactivée | **98.46%** | 77 | 94 892 | Alignement adaptatif par terminal; meilleure utilisation des nœuds rapides sans pénalité énergétique disproportionnée. |
| **Test 2 + Diversité Séquentielle** | $8 \times 10^{-6}$ (fixe) | **Activée** | **98.08%** | 96 | 94 856 | Routage forcé sur différents nœuds; surcoût de communication surmonté par l'apprentissage. |
| **Test 1.6e-5 sans Diversité** | $1.6 \times 10^{-5}$ (fixe) | Désactivée | **98.62%** | 69 | 94 896 | Puissance maximale de 2.0 W; apprentissage robuste et stable. |
| **Test 1.6e-5 avec Diversité** | $1.6 \times 10^{-5}$ (fixe) | **Activée** | **98.86%** | 57 | 94 904 | Survie énergétique maximale et routage optimal sous contraintes strictes. |

---

### 1️⃣ Test 1 : Modèle Physique Initial ($\kappa = 10^{-4}$)
*   **Dossier des résultats associés :** [avec_queue_attend](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00)
*   **Analyse :** Avec un coefficient $\kappa = 10^{-4}$, la puissance dynamique maximale du CPU atteint $12.5\text{ W}$ à $f = 50.0$, soit **12.5 fois** la consommation de la baseline linéaire. Cette disproportion biaise la récompense multi-objectif en faveur de l'économie d'énergie. Les agents apprennent à réduire systématiquement la fréquence au niveau minimal ($0.5$), augmentant drastiquement la latence de traitement. De plus, les batteries des terminaux s'épuisent rapidement lors des premières étapes de l'apprentissage, conduisant à un taux d'échec élevé (**17.4%** d'échecs, soit 870 épisodes inaboutis).

````carousel
![Récompense Cumulative Moyenne - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/plots/avg_cumulative_rewards.png)
<!-- slide -->
![Stratégie d'Exécution - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/plots/execution_strategy.png)
<!-- slide -->
![Taux de Succès vs Pénalités - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/analysis_plots/success_vs_penalties.png)
<!-- slide -->
![Consommation Énergétique Totale - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/analysis_plots/total_energy_consumption.png)
<!-- slide -->
![Latence Globale - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/analysis_plots/global_latency.png)
<!-- slide -->
![Évolution des files d'attente - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/analysis_plots/queuing_delay_evolution.png)
<!-- slide -->
![Consommation Énergétique par Terminal - Test 1](./MultiAgentMADDPG_DVFS_Physics_Env/results/avec_queue_attend/sl_1p00/analysis_plots/device_energy_evolution.png)
````

---

### 2️⃣ Test 2 : Calibrage Équilibré ($\kappa = 8 \times 10^{-6}$)
*   **Dossier des résultats associés :** [8e6_false](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00)
*   **Analyse :** Ce calibrage est conçu pour que la consommation à fréquence maximale ($f = 50.0$) soit exactement de $1.0\text{ W}$, s'alignant sur la baseline. Les agents exploitent activement la dépendance cubique du modèle DVFS : réduire la fréquence à $37.5$ ($0.75$) permet d'économiser **57.8%** de puissance, et descendre à $25.0$ ($0.5$) permet d'économiser **87.5%** de puissance. L'apprentissage est beaucoup plus stable, atteignant un taux de succès de **98.06%**. Les agents apprennent à faire des compromis fins en ajustant la fréquence à $0.75$ ou $1.0$ pour les couches de calcul intensives nécessitant une réponse rapide, et à $0.5$ pour les couches légères.

````carousel
![Récompense Cumulative Moyenne - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/plots/avg_cumulative_rewards.png)
<!-- slide -->
![Stratégie d'Exécution - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/plots/execution_strategy.png)
<!-- slide -->
![Convergence des Récompenses - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/figure_6_convergence_recompenses.png)
<!-- slide -->
![Taux de Succès vs Pénalités - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/success_vs_penalties.png)
<!-- slide -->
![Consommation Énergétique Totale - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/total_energy_consumption.png)
<!-- slide -->
![Latence Globale - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/global_latency.png)
<!-- slide -->
![Évolution des files d'attente - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/queuing_delay_evolution.png)
<!-- slide -->
![Consommation Énergétique par Terminal - Test 2](./MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00/analysis_plots/device_energy_evolution.png)
````

---

### 3️⃣ Test 3 : Alignement Dynamique par Équipement ($\kappa$ dynamique)
*   **Dossier des résultats associés :** [dyn](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00)
*   **Analyse :** Ce test résout le problème d'hétérogénéité matérielle. Un coefficient $\kappa_m$ est calculé dynamiquement pour chaque terminal $m$ afin d'aligner sa puissance à fréquence maximale ($\rho = 1.0$) sur le modèle baseline correspondant : $\kappa_m = P_{Baseline}(m) / f_m^3$. Les terminaux rapides ne subissent plus de pénalités de puissance rédhibitoires. Les agents profitent de cette équité pour mieux répartir les couches sur l'ensemble du réseau (load balancing), atteignant un taux de succès maximal de **98.46%** (seulement 77 échecs sur 5000 épisodes). La consommation d'énergie totale diminue de façon fluide tout au long de la convergence, tandis que la latence globale reste parfaitement maîtrisée.

````carousel
![Récompense Cumulative Moyenne - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/plots/avg_cumulative_rewards.png)
<!-- slide -->
![Stratégie d'Exécution - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/plots/execution_strategy.png)
<!-- slide -->
![Convergence des Récompenses - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/figure_6_convergence_recompenses.png)
<!-- slide -->
![Taux de Succès vs Pénalités - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/success_vs_penalties.png)
<!-- slide -->
![Consommation Énergétique Totale - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/total_energy_consumption.png)
<!-- slide -->
![Latence Globale - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/global_latency.png)
<!-- slide -->
![Évolution des files d'attente - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/queuing_delay_evolution.png)
<!-- slide -->
![Consommation Énergétique par Terminal - Test 3](./MultiAgentMADDPG_DVFS_BalancedEnv/results/dyn/sl_1p00/analysis_plots/device_energy_evolution.png)
````

---

### 🛡️ Impact de la Contrainte de Diversité Séquentielle

La contrainte de **Diversité Séquentielle** (`sequential_diversity = True`) interdit l'affectation de deux couches consécutives d'un même modèle au même terminal. Elle vise à réduire le risque de White-box attack en fragmentant l'exécution.

*   **Dossiers des résultats associés :** [sequential_8e6](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00) (comparé à [8e6_false](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/8e6_false/sl_1p00)) et [1.6e-5_sequ](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00) (comparé à [1.6e_false](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e_false/sl_1p00)).
*   **Analyse :** 
    1.  **Délai de communication accru :** En forçant le transfert des données intermédiaires entre terminaux différents pour chaque couche successive, la latence totale augmente légèrement en raison du goulot d'étranglement de la bande passante.
    2.  **Sollicitation énergétique partagée :** L'énergie de communication réseau augmente, mais les agents compensent en augmentant le taux d'utilisation du DVFS sur les terminaux de calcul, permettant de rester dans les limites de batterie.
    3.  **Apprentissage robuste :** Malgré ces fortes contraintes de routage, les agents MARL maintiennent un taux de succès de **98.08%** (pour $\kappa = 8 \times 10^{-6}$) et de **98.86%** (pour $\kappa = 1.6 \times 10^{-5}$). Cela démontre l'aptitude de MADDPG à découvrir des stratégies d'offloading coopératives complexes là où des heuristiques de routage classiques échoueraient face à l'explosion combinatoire.

#### 📊 Visualisation des Résultats sous Contraintes de Diversité Séquentielle ($\kappa = 8 \times 10^{-6}$)

````carousel
![Récompense Cumulative Moyenne - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/plots/avg_cumulative_rewards.png)
<!-- slide -->
![Stratégie d'Exécution - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/plots/execution_strategy.png)
<!-- slide -->
![Convergence des Récompenses - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/figure_6_convergence_recompenses.png)
<!-- slide -->
![Taux de Succès vs Pénalités - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/success_vs_penalties.png)
<!-- slide -->
![Consommation Énergétique Totale - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/total_energy_consumption.png)
<!-- slide -->
![Latence Globale - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/global_latency.png)
<!-- slide -->
![Évolution des files d'attente - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/queuing_delay_evolution.png)
<!-- slide -->
![Consommation Énergétique par Terminal - Diversité 8e6](./MultiAgentMADDPG_DVFS_BalancedEnv/results/sequential_8e6/sl_1p00/analysis_plots/device_energy_evolution.png)
````

#### 📊 Visualisation des Résultats sous Contraintes de Diversité Séquentielle ($\kappa = 1.6 \times 10^{-5}$)

````carousel
![Récompense Cumulative Moyenne - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/plots/avg_cumulative_rewards.png)
<!-- slide -->
![Stratégie d'Exécution - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/plots/execution_strategy.png)
<!-- slide -->
![Convergence des Récompenses - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/figure_6_convergence_recompenses.png)
<!-- slide -->
![Taux de Succès vs Pénalités - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/success_vs_penalties.png)
<!-- slide -->
![Consommation Énergétique Totale - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/total_energy_consumption.png)
<!-- slide -->
![Latence Globale - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/global_latency.png)
<!-- slide -->
![Évolution des files d'attente - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/queuing_delay_evolution.png)
<!-- slide -->
![Consommation Énergétique par Terminal - Diversité 1.6e-5](./MultiAgentMADDPG_DVFS_BalancedEnv/results/1.6e-5_sequ/sl_1p00/analysis_plots/device_energy_evolution.png)
````

---

---

## ⚙️ Command-line Options

Below is the full list of command‑line arguments supported by the training script in the `MultiAgentMADDPG_DevicePowerEnv` environment.

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes EPISODES` | Number of training episodes | `1000` |
| `--seed SEED` | Random seed for reproducibility | `42` |
| `--sl [SL ...]` | Security level(s) | – |
| `--models MODELS [MODELS ...]` | Paths to model files | – |
| `--log-every LOG_EVERY` | Logging frequency | – |
| `--log-trace` | Enable trace logging | `False` |
| `--trace-max-steps TRACE_MAX_STEPS` | Max steps for trace | – |
| `--queue-per-device` | Enable per‑device queuing model | `False` |
| `--privacy-max-level PRIVACY_MAX_LEVEL` | Max privacy level | – |
| `--privacy-profile PRIVACY_PROFILE` | Privacy profile name | – |
| `--trust-min-for-max-privacy TRUST_MIN_FOR_MAX_PRIVACY` | Trust threshold for max privacy | – |
| `--trust-score-min TRUST_SCORE_MIN` | Minimum trust score | – |
| `--trust-score-max TRUST_SCORE_MAX` | Maximum trust score | – |
| `--energy-min ENERGY_MIN` | Minimum energy budget | – |
| `--energy-max ENERGY_MAX` | Maximum energy budget | – |
| `--base-power-comp BASE_POWER_COMP` | Baseline compute power (W) | – |
| `--base-power-comm BASE_POWER_COMM` | Baseline communication power (W) | – |
| `--base-cpu-speed BASE_CPU_SPEED` | Baseline CPU speed for normalization | – |
| `--base-bandwidth BASE_BANDWIDTH` | Baseline bandwidth for normalization | – |
| `--eps-decay EPS_DECAY` | Epsilon decay rate | – |
| `--eps-min EPS_MIN` | Minimum epsilon | – |
| `--alpha ALPHA` | Weight for latency in reward | `0.5` |
| `--beta BETA` | Weight for energy in reward | `0.5` |

You can see the full help by running:

```bash
python train.py -h
```

---

## 📚 Références

1. Ahmed, A. — *A Survey on Mobile Edge Computing*, IEEE, 2016
2. Albrecht, S. — *Multi-Agent Reinforcement Learning*, 2024
3. Baccour, E. et al. — *Multi-agent RL for privacy-aware distributed CNN*, JNCA, 2024
4. Baccour, E. — *RL-PDNN*, IEEE Access, 2021
5. Hady, M.A. et al. — *MARL for resources allocation: a survey*, AI Review, 2025
6. Kang, Y. et al. — *Neurosurgeon*, ACM SIGOPS, 2017
7. Li, Y. et al. — *Learning-based computation offloading for MEC*, IEEE, 2018
8. Lin, L. et al. — *Computation Offloading Toward Edge Computing*, IEEE, 2019
9. Mi, X. et al. — *Multi-Agent RL for Dynamic Task Offloading*, Sensors, 2024
10. Mnih, V. et al. — *Human-level control through deep RL*, Nature, 2015
11. Sutton, R.S. & Barto, A.G. — *Reinforcement Learning: An Introduction*, MIT Press, 2018
12. Zhang, K. et al. — *MARL: A Selective Overview*, IEEE Transactions, 2021

---

<div align="center">

**Université Moulay Slimane — FST Béni Mellal**  
Département Informatique · Année 2025–2026

</div>

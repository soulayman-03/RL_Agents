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

```
RL_Agents/
├── agents/                  # Définition des agents MARL
│   ├── dqn_agent.py         # Agent Deep Q-Network
│   ├── actor_critic.py      # Agent Actor-Critic
│   └── base_agent.py        # Classe de base
│
├── environment/             # Environnement de simulation
│   ├── edge_env.py          # Environnement IoT/Edge/Cloud
│   ├── task_model.py        # Modélisation des tâches CNN
│   └── dvfs.py              # Module DVFS
│
├── models/                  # Réseaux de neurones
│   ├── q_network.py
│   └── policy_network.py
│
├── utils/                   # Outils et métriques
│   ├── replay_memory.py     # Experience Replay
│   ├── metrics.py           # Latence, énergie, etc.
│   └── config.py            # Hyperparamètres
│
├── experiments/             # Scripts d'expérimentation
│   ├── train.py             # Entraînement
│   └── evaluate.py          # Évaluation et comparaison
│
├── results/                 # Résultats et figures
├── requirements.txt
└── README.md
```

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

## 📊 Résultats

> *Section en cours — les résultats expérimentaux seront ajoutés après la phase d'évaluation.*

Les expériences compareront l'approche MARL proposée avec les baselines suivantes :
- Exécution locale pure
- Offloading total vers le Cloud
- Allocation aléatoire
- DQN mono-agent

**Métriques évaluées :** Latence moyenne · Consommation énergétique · Taux de violation des contraintes · Convergence

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
Département Informatique · Année 2024–2025

</div>

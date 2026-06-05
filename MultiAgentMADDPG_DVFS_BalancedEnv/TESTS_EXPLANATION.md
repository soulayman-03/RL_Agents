# Explication des Tests en Cours (Paramètres Expérimentaux)

Ce document décrit les différentes configurations expérimentales en cours de test dans l'environnement `MultiAgentMADDPG_DVFS_BalancedEnv` et l'impact de chaque paramètre sur le comportement du système multi-agents.

## 1. Test du Modèle d'Énergie Physique (`capacitance_activity_factor` ou $\kappa$)

Nous testons deux échelles différentes pour le paramètre $\kappa$ (facteur de capacité et d'activité) qui régit la formule de consommation énergétique dynamique du CPU : $P_{comp} = \kappa \cdot f^3$.

### Test avec $\kappa = 8 \times 10^{-6}$ (`8e-6`)
- **Signification :** C'est le paramétrage "Normal / Baseline-aligned". 
- **Conséquence :** À la fréquence maximale ($f = 50.0$), la consommation électrique est exactement de **1.0 Watt** ($8 \times 10^{-6} \times 50^3$). 
- **Objectif :** Cela permet une comparaison directe et équitable avec la méthode Baseline. Toute utilisation du DVFS (réduction de $f$) permet un gain énergétique net mesurable par rapport à ce 1.0 W de référence.

### Test avec $\kappa = 1.6 \times 10^{-8}$ (`1.6e-8`)
- **Signification :** C'est un paramétrage pour une architecture de processeur **haute efficacité énergétique** (Ultra-Low Power) ou pour tester le comportement de l'agent face à des coûts énergétiques beaucoup plus faibles. *(Note: Si vos fréquences testées sont de l'ordre de $f=500$, cela ramènerait la puissance maximale à 2.0 W).*
- **Conséquence :** Si on reste sur $f = 50.0$, l'énergie consommée devient très faible ($1.6 \times 10^{-8} \times 50^3 = 0.002$ Watt). 
- **Objectif :** Vérifier si les agents MADDPG décident toujours d'utiliser le DVFS lorsque la pénalité énergétique (et donc le gain potentiel du DVFS) est drastiquement réduite ou modifiée. L'agent doit s'adapter à une nouvelle échelle de valeurs dans sa fonction de récompense.

---

## 2. Test de la Diversité Séquentielle (`sequential_diversity`)

Ce paramètre, défini dans `integrated_system/resource_manager.py`, représente une contrainte de sécurité et de répartition des charges pour le traitement des sous-tâches (ou couches d'un modèle d'IA).

### Test avec `sequential_diversity = False`
- **Comportement :** Le système permet à un même périphérique (Device) de traiter plusieurs couches successives d'une même tâche.
- **Impact sur la Latence :** Souvent plus rapide (latence minimisée) car on évite les délais de communication réseau (transmission) entre deux couches consécutives si elles sont exécutées sur la même machine.
- **Impact sur la Sécurité :** L'exposition des données est plus élevée. Si un appareil est compromis, il a accès à une séquence continue de données, ce qui augmente le risque pour la confidentialité.

### Test avec `sequential_diversity = True`
- **Comportement :** Contrainte stricte : **deux couches successives d'une tâche ne doivent jamais être traitées par le même périphérique**. L'ordonnanceur est forcé de distribuer la charge.
- **Impact sur la Latence :** Augmente intrinsèquement la latence globale en raison du surcoût de communication obligatoire entre les périphériques pour transférer les résultats intermédiaires d'une couche à l'autre.
- **Impact sur la Sécurité :** Haute sécurité et confidentialité renforcée. Le fractionnement forcé garantit qu'aucun nœud ne détient une portion trop importante du traitement consécutif.
- **Défi pour l'Agent RL :** Apprendre à optimiser la latence globale et l'énergie via le DVFS, tout en subissant des pénalités de communication obligatoires. L'agent est forcé de devenir plus intelligent dans le placement de chaque couche.

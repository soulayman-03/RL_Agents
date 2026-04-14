# Documentation du Projet : Optimisation du Placement de Tâches IoT par Apprentissage Renforcé Multi-Agent (MADDPG)

Ce document décrit l'architecture, la fonction objectif, les contraintes et les scénarios de test du système de gestion des ressources pour l'offloading (déchargement) de modèles de Deep Learning dans un environnement IoT multi-agent.

## 1. Architecture du Système

Le projet utilise l'algorithme **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**, une extension du DDPG pour les environnements multi-agents.

### Composants principaux :
- **Agents (Acteurs)** : Chaque agent représente une tâche ou un utilisateur IoT tentant de placer les couches de son modèle (ex: ResNet, VGG) sur les dispositifs disponibles. Ils prennent des décisions basées sur leurs observations locales.
- **Critique Centralisé** : Pendant la phase d'entraînement, un critique centralisé observe les états et les actions de **tous** les agents pour évaluer la qualité des décisions prises globalement. Cela aide à stabiliser l'apprentissage dans un environnement non stationnaire.
- **Environnement (Multi-Agent IoT)** : Un simulateur personnalisé qui modère les ressources (CPU, Mémoire, Bande passante), gère l'énergie des dispositifs et applique les règles de sécurité/privacité.
- **Réseau de neurones** : 
    - **Actor** : Réseau de neurones qui produit les probabilités d'actions (choix du dispositif).
    - **Critic** : Réseau de neurones qui estime la valeur Q (récompense future attendue).

## 2. Espace d'Observation et d'Action

Pour que les agents puissent prendre des décisions intelligentes, ils doivent recevoir des informations précises sur l'état du système.

### A. L'Espace d'Observation (Ce que l'agent voit)
Chaque agent reçoit un vecteur d'état normalisé comprenant :
1.  **Progression de la tâche** : Un ratio (0 à 1) indiquant l'avancement dans les couches du modèle.
2.  **Caractéristiques de la couche actuelle** :
    *   Demande de calcul (FLOPS).
    *   Demande de mémoire (RAM).
    *   Taille des données de sortie (pour la transmission).
    *   Niveau de confidentialité requis.
3.  **État des dispositifs (pour chaque dispositif)** :
    *   Vitesse CPU et bande passante théoriques.
    *   Mémoire disponible actuelle.
    *   Niveau d'accréditation de sécurité (Privacy Clearance).
    *   Charge de travail actuelle (Compute/BW load pour l'étape).
    *   **Ratio d'énergie** : Énergie restante par rapport au budget initial.
    *   **Score de Trust** : Fiabilité actuelle du dispositif.

### B. L'Espace d'Action (Ce que l'agent fait)
L'action est **discrète**. À chaque étape, l'agent doit choisir l'ID du dispositif (entre 0 et $M-1$) sur lequel placer la couche actuelle.
- **Masquage d'actions** : Si un dispositif ne respecte pas les contraintes (énergie épuisée ou sécurité insuffisante), cette action est masquée et l'agent ne peut pas la choisir.

## 3. Détails de l'Algorithme MADDPG

MADDPG utilise le paradigme **CTDE (Centralized Training, Decentralized Execution)** :

*   **Entraînement Centralisé** : Les réseaux "Critics" ont accès aux observations et actions de tous les agents. Cela permet de résoudre le problème de non-stationnarité dans les systèmes multi-agents (les changements d'un agent affectent les autres).
*   **Exécution Décentralisée** : Une fois entraînés, les agents n'utilisent que leur réseau "Actor" et leurs propres observations locales pour prendre des décisions en temps réel sur le terrain.
*   **Gumbel-Softmax** : Comme les actions sont discrètes (choix du device), nous utilisons la relaxation Gumbel-Softmax pour permettre la rétropropagation du gradient à travers des choix discrets.

## 4. Fonction Objectif et Récompenses

L'objectif global du système est de minimiser à la fois la **latence totale** (temps d'exécution) et la **consommation d'énergie**.

### Formule de la Récompense (Reward) :
La récompense d'un agent pour une action réussie est définie par :
$$ R = -(T_{total} + E_{total}) $$
Où :
- $T_{total} = T_{comp} + T_{comm}$ (Temps de calcul + Temps de communication).
- $E_{total} = E_{comp} + E_{comm}$ (Énergie consommée pour le calcul + Énergie pour la transmission).

### Récompense d'Équipe (Coopération) :
Pour encourager la collaboration, la récompense finale d'un agent est un mélange entre sa performance individuelle et la moyenne du groupe :
$$ R_{final} = 0.7 \times R_{individuelle} + 0.3 \times \overline{R}_{groupe} $$

### Pénalités :
Si un agent échoue à respecter une contrainte (ex: batterie épuisée, mémoire insuffisante), il reçoit une forte pénalité :
- **Reward d'échec** : $-500.0$

## 5. Contraintes du Système

Le système impose plusieurs contraintes strictes ("Hard Constraints") pour simuler un environnement réel :

### A. Contraintes de Ressources (ResourceManager)
- **Calcul (Compute)** : Le dispositif doit avoir une capacité CPU suffisante pour traiter la charge de travail de la couche.
- **Mémoire** : Le dispositif doit avoir assez de mémoire vive (RAM) disponible pour charger les paramètres de la couche.
- **Bande Passante (Bandwidth)** : La transmission des données entre les dispositifs est limitée par la bande passante disponible.

### B. Énergie (Modèle de Puissance)
- Chaque dispositif possède un **budget batterie** (Energy Budget) pour l'épisode.
- **Consommation** : $E = P_i \times T$ (La puissance dépend des caractéristiques matérielles du dispositif).
- Si l'énergie restante est inférieure au coût estimé de l'action, l'allocation est refusée.

### C. Sécurité et Confidentialité
- **Niveaux de Confidentialité (Privacy Levels)** : Chaque couche d'un modèle a un niveau de confidentialité requis (ex: de 0 à 3).
- **Autorisation (Privacy Clearance)** : Chaque dispositif a un niveau d'accréditation.
- **Trust (Confiance)** : Chaque dispositif a un "score de confiance" (Trust Score). Un agent ne peut placer une tâche sensible que sur un dispositif dont le trust score est supérieur au seuil requis pour ce niveau de confidentialité.

### D. Diversité Séquentielle (Sequential Diversity)
- Une règle qui peut limiter l'utilisation répétitive des mêmes dispositifs pour des couches consécutives afin d'éviter la congestion ou d'augmenter la redondance.

### E. Modélisation des files d'attente (Queuing Delay)
- Pour éviter que les agents ne saturent de manière irréaliste les nœuds les plus performants, le système peut appliquer un modèle de file d'attente (`queue_per_device`).
- Lorsqu'il est actif, si plusieurs agents allouent des calculs ou des transferts réseau au même dispositif simultanément pendant la même étape, un temps d'attente est ajouté à la latence totale.
- Cela force les agents à internaliser le coût de la congestion dans leur récompense et favorise un équilibrage de charge (Load Balancing) naturel sur l'ensemble du réseau IoT.

## 6. Scénarios de Test

Les performances du modèle sont évaluées selon les scénarios suivants définis dans `train.py` :

### Configuration par défaut :
- **Agents** : 10 agents.
- **Dispositifs** : 15 dispositifs hétérogènes.
- **Modèles de DL** : Mix de ResNet18, VGG11, DeepCNN, etc.
- **Épisodes** : 5000 épisodes d'entraînement.

### Paramétres de Test :
1. **Heterogénéité du réseau** : Tests avec différents ratios de puissance de calcul et de bande passante (Power/BW baseline).
2. **Niveaux de Risque (SL - Security Level)** : Variation de la fraction d'exposition maximale autorisée pour tester la robustesse aux contraintes de sécurité.
3. **Profils de Confidentialité** :
    - `linear_front_loaded` : Les premières couches sont plus sensibles.
    - `random` : Sensibilité aléatoire des couches.
    - `first_layer_max` : Seule la première couche est ultra-sensible.
4. **Contraintes d'Énergie** : Plages de budget batterie variables (ex: 5000 à 12000 unités) pour tester la survie des agents en situation de pénurie d'énergie.

## 7. Visualisation des Résultats

Le système génère automatiquement des métriques pour valider le travail :
- **Courbes de récompense** (Cumulée et moyenne).
- **Pertes Actor/Critic** (Convergence de l'IA).
- **Taux de succès par agent**.
- **Répartition des choix de dispositifs** (Execution Strategy).
- **Flux d'exécution** (Visualisation du cheminement des tâches).

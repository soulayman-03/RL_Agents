# Environnement de Référence : Baseline (MultiAgentMADDPG_DevicePowerEnv)

Ce dossier contient l'environnement de référence (Baseline) de votre système IoT Multi-Agent. Il sert de point de comparaison fondamental pour évaluer les performances et les gains apportés par des mécanismes d'optimisation plus avancés, comme le DVFS.

## 1. Modèle Physique d'Énergie (Modèle Linéaire)
Contrairement au modèle DVFS (qui utilise une physique cubique), cet environnement utilise un modèle de consommation énergétique **linéaire et proportionnel** aux capacités matérielles du serveur.
La puissance de calcul ($P_{comp}$) d'un appareil est définie par :
$$ P_{comp} = P_{base} \times \left( \frac{cpu\_speed}{base\_cpu\_speed} \right) $$
- Avec $P_{base} = 1.0$ Watt et $base\_cpu\_speed = 50.0$.
- **Résultat :** Un processeur ayant une capacité maximale de 50.0 consommera fixement 1.0 Watt lorsqu'il calcule. Ce modèle ne permet pas à l'agent de réduire la fréquence du processeur pour économiser de l'énergie : les serveurs tournent toujours à leur vitesse physique maximale.

## 2. Rôle de l'Agent RL (Allocation pure)
Dans cet environnement, l'espace d'action de l'agent se limite strictement au **placement des tâches (Task Offloading)**. 
- L'agent doit décider vers quel appareil (device_id) envoyer chaque couche (layer) du réseau de neurones.
- Son seul moyen de réduire la consommation d'énergie est de choisir des appareils avec un meilleur ratio d'efficacité énergétique ou de limiter les transferts réseau inutiles, mais il **ne peut pas** moduler la vitesse matérielle.

## 3. Pondération de la Récompense ($\alpha$ et $\beta$)
Afin d'assurer une comparaison 100% équitable avec le modèle DVFS, cet environnement a été mis à jour pour respecter la même formulation mathématique multi-objectif :
$$ \min_{\mathbf{a}} \sum_{n \in \mathcal{N}} \left( \alpha T_n + \beta E_n \right) $$
- **Modification :** Les paramètres `--alpha` et `--beta` sont disponibles dans `train.py` pour pondérer l'importance de la latence ($T_n$) par rapport à l'énergie ($E_n$).
- **Recommandation :** Lors de vos tests comparatifs, veillez à lancer la Baseline avec **exactement les mêmes valeurs** de $\alpha$ et $\beta$ que l'environnement DVFS (par exemple `--alpha 0.4 --beta 0.6`) pour que la fonction de coût évaluée par l'agent soit mathématiquement identique.

## 4. Structure du Dossier de Résultats (`results/`)
Chaque fois qu'un entraînement est lancé, un dossier de résultats est généré automatiquement avec une structure très précise pour faciliter l'analyse scientifique :

**Arborescence type :**
`results/models_..._e0.5k_a0.4_b0.6/sl_1p00/`
- **Le nom du dossier (Scénario) :** Inclut les paramètres de sécurité (ex: `p3`), d'énergie (ex: `e0.5k`), et les pondérations de la récompense (`a0.4_b0.6`).
- **`sl_1p00` :** Indique le niveau de la contrainte d'exposition (Security Level).

**Contenu du dossier de l'expérience :**
1. **`plots/` :** Dossier contenant tous les graphiques générés à la fin de l'entraînement (courbes de convergence, taux de succès, latence par couche, exécution spatiale).
2. **`run_config.json`, `model_summary.json`, `device_summary.json` :** Fichiers de configuration sauvegardant les conditions initiales exactes de l'expérience (permettant la reproductibilité).
3. **`train_log.jsonl` & `episode_impact.jsonl` :** Fichiers journaux détaillés (Step-by-Step) contenant la trace de toutes les actions, consommations énergétiques, et latences.
4. **Fichiers `.npy` :** Fichiers de données Numpy brutes (récompenses, loss de l'actor/critic) utilisés par vos scripts de comparaison comme `compare_all_metrics.py`.
5. **`summary.json` :** Résumé global de l'entraînement avec le taux de succès final de chaque agent.

Les métriques obtenues dans ce dossier Baseline représentent le "pire scénario énergétique" (les CPU tournent toujours à 100%), servant à démontrer l'utilité absolue du DVFS lors de la comparaison des résultats.

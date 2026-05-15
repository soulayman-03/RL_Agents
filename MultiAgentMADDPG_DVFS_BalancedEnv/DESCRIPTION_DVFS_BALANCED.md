# Environnement DVFS Équilibré (MultiAgentMADDPG_DVFS_BalancedEnv)

Ce dossier est une évolution de l'environnement DVFS physique initial. Il intègre plusieurs modifications mathématiques et logicielles majeures visant à permettre une comparaison juste et paramétrable entre l'approche Baseline et l'approche DVFS.

## 1. Calibrage du modèle physique d'énergie (`kappa`)
Dans le modèle physique original, la puissance de calcul du CPU était calculée par la formule :
$$ P_{comp} = \kappa \cdot f^3 $$
Avec un $\kappa = 10^{-4}$, cela engendrait une consommation de base 12,5 fois supérieure à la Baseline (qui consommait 1.0 W pour un CPU à vitesse 50.0).
- **Modification :** La constante $\kappa$ (`capacitance_activity_factor` dans `environment.py`) a été ajustée à `8e-6`.
- **Résultat :** À fréquence maximale ($f = 50.0$), la consommation est désormais exactement de **1.0 Watt** ($8 \times 10^{-6} \cdot 50^3 = 1.0$), s'alignant parfaitement sur la Baseline. Toute réduction de fréquence via le DVFS se traduit désormais par un véritable gain net par rapport à la Baseline (ex: 0.125 W à 50% de fréquence).

## 2. Pondération de la fonction de récompense ($\alpha$ et $\beta$)
Pour forcer l'agent à trouver un véritable équilibre entre la Latence et l'Énergie, la fonction de récompense a été modifiée pour respecter précisément la formulation mathématique multi-objectif :
$$ \min_{\mathbf{a}} \sum_{n \in \mathcal{N}} \left( \alpha T_n + \beta E_n \right) $$
- **Modification :** Ajout des paramètres `alpha` et `beta` dans l'initialisation de l'environnement et l'équation des récompenses (`rewards[aid] = -(self.alpha * total_latency + self.beta * cost)`).
- **Motivation :** L'énergie chutant au cube ($f^3$) tandis que la latence augmente plus lentement ($1/f$), une pénalité brute asymétrique favorisait le maintien du CPU à 100%. Ces paramètres permettent de valoriser davantage les gains énergétiques pour inciter l'agent à utiliser le DVFS (recommandation : $\alpha=0.4, \beta=0.6$).

## 3. Paramétrage dynamique dans l'entraînement (`train.py`)
- **Modification :** Ajout des arguments `--alpha` et `--beta` dans le parseur d'arguments (`argparse`) du script de lancement de l'entraînement.
- **Modification :** Inclusion de ces valeurs dans la génération automatique du nom du dossier de résultats (`_a{alpha}_b{beta}`).
- **Résultat :** Il est désormais possible de tester en ligne de commande de multiples pondérations, chaque test générant son propre sous-dossier de résultats sans écraser les exécutions précédentes.

## 4. Structure du Dossier de Résultats (`results/`)
Chaque fois qu'un entraînement est lancé, un dossier de résultats est généré automatiquement avec une structure très précise pour faciliter l'analyse scientifique :

**Arborescence type :**
`results/models_..._e0.5k_a0.4_b0.6/sl_1p00/`
- **Le nom du dossier (Scénario) :** Inclut les paramètres de sécurité (ex: `p3`), d'énergie (ex: `e0.5k`), et les pondérations de la récompense (`a0.4_b0.6`).
- **`sl_1p00` :** Indique le niveau de la contrainte d'exposition (Security Level).

**Contenu du dossier de l'expérience :**
1. **`plots/` :** Dossier contenant tous les graphiques générés à la fin de l'entraînement (courbes de convergence, taux de succès, latence par couche, exécution spatiale).
2. **`run_config.json`, `model_summary.json`, `device_summary.json` :** Fichiers de configuration sauvegardant les conditions initiales exactes de l'expérience (permettant la reproductibilité).
3. **`train_log.jsonl` & `episode_impact.jsonl` :** Fichiers journaux détaillés (Step-by-Step) contenant la trace de toutes les actions, pondérations DVFS choisies, consommations énergétiques, et latences.
4. **Fichiers `.npy` :** Fichiers de données Numpy brutes (récompenses, loss de l'actor/critic) utilisés par vos scripts de comparaison comme `compare_all_metrics.py`.
5. **`summary.json` :** Résumé global de l'entraînement avec le taux de succès final de chaque agent.

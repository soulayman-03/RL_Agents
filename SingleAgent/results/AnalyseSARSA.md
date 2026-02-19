# Analyse SARSA (Deep SARSA)

SARSA est **on-policy** : la mise à jour utilise `(s, a, r, s’, a’)` où `a’` est l’action réellement choisie par la politique (contrairement à DQN qui utilise typiquement `max_a Q(s’, a)`).

## 1) Reward (training)
Fichier :
- `SingleAgent/models/sarsa_train_history.npy`

Reward = **− latence totale** (compute + communication).
- reward moins négatif ⇒ latence plus faible ⇒ meilleure stratégie.

## 2) Stalls (training)
Fichier :
- `SingleAgent/models/sarsa_stall_history.npy`

Stall = pénalité **−500** quand la policy choisit une action invalide.
- stalls ≈ 0 ⇒ contraintes respectées et placement faisable.

## 3) Métriques ressources (training)
Fichiers :
- `SingleAgent/models/sarsa_cpu.npy`
- `SingleAgent/models/sarsa_mem.npy`
- `SingleAgent/models/sarsa_net.npy`

Interprétation :
- `net` élevé ⇒ changements de device fréquents ⇒ communication coûteuse,
- `cpu/mem` permettent de voir si la stratégie charge beaucoup certains devices.

## 4) Plots training
Dossier :
- `SingleAgent/results/resultSARSA/train/`

Exemples :
- `sarsa_training_history.png` : reward + moving average,
- `execution_strategy.png` : trace d’un épisode réussi,
- `training_metrics.png` : reward + CPU + mem + net.

## 5) Logs structurés
Fichier :
- `SingleAgent/results/resultSARSA/train/train_log.jsonl`

Chaque épisode inclut `devices`, `trace` (`t_comp`, `t_comm`) + métriques `cpu/mem/net`.

## 6) Évaluation et robustesse
Évaluation :
- `SingleAgent/results/resultSARSA/eval/`

Robustesse :
- `SingleAgent/results/resultSARSA/robustness/robustness_results.json`
- `SingleAgent/results/resultSARSA/robustness/robustness_comparison.png`


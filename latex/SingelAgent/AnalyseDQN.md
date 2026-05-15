# Analyse DQN

Le DQN apprend une fonction **Q(s, a)** et choisit l’action (device) qui maximise la valeur estimée, avec exploration via **epsilon-greedy**.

## 1) Reward (training)
Fichiers :
- `SingleAgent/results/resultDQN/train/dqn_training_history.png`
- `SingleAgent/models/single_train_history.npy`

Ici, reward = **− latence totale** (compute + communication). Donc :
- reward moins négatif / qui remonte ⇒ latence plus faible ⇒ meilleure stratégie.

## 2) Stalls (training)
Fichiers :
- `SingleAgent/results/resultDQN/train/dqn_stalls_history.png`
- `SingleAgent/models/single_stall_history.npy`

Un *stall* correspond à une pénalité **−500** (action invalide / contrainte violée / allocation impossible).
- stalls proches de 0 ⇒ contraintes respectées et exécution faisable.

## 3) Epsilon (exploration)
Fichier :
- `SingleAgent/results/resultDQN/train/dqn_epsilon.png`

Le DQN explore au début (epsilon élevé), puis exploite (epsilon ↓).
- si le reward s’améliore pendant que epsilon baisse, c’est attendu.

## 4) Logs détaillés par épisode
Fichier :
- `SingleAgent/results/resultDQN/train/train_log.jsonl`

Chaque épisode inclut :
- `devices` : liste des devices choisis,
- `trace` : (layer, device, `t_comp`, `t_comm`).

## 5) Évaluation (multi-seeds)
Dossier :
- `SingleAgent/results/resultDQN/eval/`

Exemples :
- `layer_latency_avg_seed_*.png` : latence moyenne par layer (computation vs communication).

## 6) Robustesse (scénarios)
Fichiers :
- `SingleAgent/results/resultDQN/robustness/robustness_results.json`
- `SingleAgent/results/resultDQN/robustness/robustness_comparison.png`

Lecture :
- reward plus négatif en *Low BW* ⇒ communication domine (réseau = bottleneck),
- reward plus négatif en *Slow CPU* ⇒ compute domine (device ralenti).


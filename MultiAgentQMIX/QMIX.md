# MultiAgentQMIX (QMIX / CTDE)

## C’est quoi QMIX ?
QMIX est un algorithme **MARL CTDE** :
- **Training centralisé** : on apprend une valeur d’équipe `Q_tot`.
- **Exécution décentralisée** : chaque agent choisit son action avec son observation locale.

L’idée clé : au lieu de sommer les Q des agents (VDN), QMIX apprend une fonction de mélange non-linéaire :

`Q_tot = f_mix(Q_1, Q_2, ..., Q_N, s)`

où `s` est un **état global**.  
La contrainte importante est la **monotonicité** :

`∂Q_tot / ∂Q_i ≥ 0`

Ainsi, maximiser chaque `Q_i` localement reste cohérent avec maximiser `Q_tot`.

## Comment c’est implémenté ici ?

Package : `MultiAgentQMIX/`

- `agent.py` : réseau `Q_i(o_i, a)` + epsilon-greedy + action masking
- `mixer.py` : `QMixer` (hypernetworks conditionnées par `state`)
- `manager.py` : replay buffer joint + entraînement TD sur `Q_tot`
- `environment.py` : ré-export de `MultiAgent.environment.MultiAgentIoTEnv`
- `train.py` / `evaluate.py` : scripts d’entraînement et d’évaluation

### État global utilisé (`state`)
On n’a pas un `state` explicite dans l’environnement, donc on construit :
- `state = concat( obs_0, obs_1, ..., obs_{N-1} )`

Les observations contiennent déjà l’état des devices (ressources), donc ce `state` capture le contexte global.

## Commandes

- Entraîner : `python -m MultiAgentQMIX.train`
- Évaluer : `python -m MultiAgentQMIX.evaluate`

## Fichiers générés

Modèles :
- `MultiAgentQMIX/models/qmix_agent_*.pt`
- `MultiAgentQMIX/models/qmix_mixer.pt`

Résultats :
- `MultiAgentQMIX/results/training_trends.png`
- `MultiAgentQMIX/results/training_agent_rewards.png`
- `MultiAgentQMIX/results/marl_eval_report.json`
- `MultiAgentQMIX/results/evaluation_summary.png`
- `MultiAgentQMIX/results/marl_eval_summary.png`
- `MultiAgentQMIX/results/execution_flow.png`


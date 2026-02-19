# MultiAgentVDN (VDN / CTDE)

`MultiAgentVDN` implémente un entraînement multi-agent **CTDE** (*Centralized Training, Decentralized Execution*) avec **VDN** (*Value Decomposition Networks*) pour décider sur quel **device** exécuter les layers de **3 modèles DNN** différents, tout en partageant les ressources (CPU/RAM/BW/privacy).

## 1) Idée VDN (cœur de l’algo)

- Chaque agent `i` apprend un Q-network local : `Q_i(o_i, a_i)` (comme un DQN).
- En CTDE, on construit une valeur d’équipe :
  - `Q_total = Σ_i Q_i` (somme simple = VDN).
- L’entraînement minimise une **TD-loss** sur `Q_total` (replay buffer joint), mais l’exécution reste **décentralisée** :
  - chaque agent choisit son action depuis son observation,
  - avec un masque d’actions valides.

Fichier clé : `MultiAgentVDN/manager.py` (calcul de `q_total`, `target_total`, loss, updates).

## 2) Environnement (problème simulé)

`MultiAgentVDN/environment.py` ré-exporte `MultiAgent.environment.MultiAgentIoTEnv` (donc le vrai env est dans `MultiAgent/environment.py`).

Paramètres training (dans `MultiAgentVDN/train.py`) :
- `NUM_AGENTS = 3`, `NUM_DEVICES = 5`
- `MODEL_TYPES = ["simplecnn", "deepcnn", "miniresnet"]`
- `shuffle_allocation_order=True` (ordre d’allocation peut varier → plus robuste)

### Actions
Chaque agent choisit un `device_id` (0..4) à chaque étape (allocation d’un layer).  
Le script utilise `env.get_valid_actions()` pour obtenir les actions valides **par agent** (respect des contraintes).

### Récompense
`rewards` est un dict `{agent_id: reward}`.  
Le manager convertit en reward d’équipe :
- `reward_team = mean(rewards_i)` (normalisation stable). Voir `VDNManager.remember()`.

### Terminaison “team”
Si un agent échoue une allocation (`info success=False`), l’épisode est terminé **pour toute l’équipe** (ressources partagées).

## 3) Agents (réseau + exploration)

Fichier : `MultiAgentVDN/agent.py`
- `VDNAgent` = DQN classique (`policy_net` + `target_net`)
- Exploration : `EpsilonSchedule` (epsilon-greedy)
- Masque actions valides : Q-values des actions invalides masquées à `-inf`

## 4) Replay Buffer joint (transition multi-agent)

Fichier : `MultiAgentVDN/replay_buffer.py`

Stocke une transition conjointe :
- `obs`: `(N, obs_dim)`
- `actions`: `(N,)`
- `reward_team`: scalaire
- `next_obs`: `(N, obs_dim)`
- `dones`: `(N,)` (done par agent au prochain état)
- `done_team`: bool (terminal équipe, ex: fail)
- `active`: `(N,)` (agent actif au moment de la transition)
- `next_action_mask`: `(N, action_dim)` (actions valides au prochain état)

Le champ `active` sert à ne pas compter un agent déjà terminé dans `Q_total`.

## 5) Entraînement (train.py)

Fichier : `MultiAgentVDN/train.py`

Boucle épisodes (5000). À chaque step :
1. `valid_actions = env.get_valid_actions()`
2. `actions = manager.get_actions(obs, valid_actions)`
3. `next_obs, rewards, next_done, truncated, infos = env.step(actions)`
4. `manager.remember(...)` (transition jointe + masque next valid actions)
5. `loss = manager.train()` (quand le replay buffer est assez rempli)

Logs toutes les 50 épisodes : moyenne team, loss, epsilon, steps, taille replay + mapping détaillé par agent.


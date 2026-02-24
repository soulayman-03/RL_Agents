# Analyse de la contrainte `S_l` (Security Level) et comparaison avec "2 couches successives"

Dans ce projet, on a **deux contraintes différentes** qui touchent la façon de répartir les couches d’un modèle sur les devices.

## 1) Contrainte "2 couches successives" (Sequential Diversity)
**Idée :** la couche `i` et la couche `i-1` **ne doivent pas** être exécutées sur le même device.

- **Forme :** `device_i != device_{i-1}`
- **Nature :** contrainte **locale** (elle regarde seulement deux couches adjacentes).
- **Effet :** force l’alternance immédiate, mais **n’empêche pas** qu’un device voie une grande partie du modèle au total.

Exemple (valide avec cette contrainte) :
- `D2, D3, D2, D3, D2, D3, ...`

Ici, `D2` peut quand même traiter ~50% des couches (voire plus selon le pattern), donc l’exposition globale n’est pas contrôlée.

## 2) Contrainte `S_l` (Max Exposure Fraction)
**Idée :** limiter la proportion maximale du modèle (nombre de couches) qu’un **seul** device peut voir/traiter pendant l’exécution d’un épisode.

- **Forme :** pour un modèle de `L` couches et un `S_l` dans `(0, 1]` :
  - `max_layers_per_device = floor(L * S_l)`
  - on impose : `#couches assignées à (device d) <= max_layers_per_device`
- **Nature :** contrainte **globale** (sur tout l’épisode / tout le modèle).
- **Effet :** contrôle réellement l’**exposition globale** du modèle à un device, même si l’agent alterne entre devices.

Exemples pour `cnn15` (`L=15`) :
- `S_l = 0.50` → `floor(15*0.50)=7` couches max par device
- `S_l = 0.33` → `floor(15*0.33)=4` couches max par device
- `S_l = 0.20` → `floor(15*0.20)=3` couches max par device

## 3) Différence clé (en une phrase)
- **Sequential diversity** empêche seulement la répétition **consécutive** d’un device.
- **`S_l`** limite la **fraction totale** du modèle exposée à un device (même avec alternance).

## 4) Impact attendu de `S_l` sur les résultats (Reward / Stalls)
Dans ce projet, la reward est basée sur la latence :
- `reward = -(t_comp + t_comm)`
- un échec d’allocation / violation de contrainte provoque un **stall** avec une grosse pénalité (ex: `-500`).

Quand tu diminues `S_l` :
1) **Plus d’actions deviennent invalides** (un device atteint son quota d’exposition).
2) L’agent est forcé de **répartir davantage** les couches → souvent plus de changements de device.
3) Ça augmente généralement `t_comm` (communication) et parfois `t_comp` (si on doit utiliser un device moins bon).
4) Si `S_l` est trop strict (ou combiné aux autres contraintes), tu peux avoir plus de **stalls** (l’agent n’arrive plus à trouver une allocation faisable).

Lecture pratique :
- Si la reward moyenne devient **plus négative** quand `S_l` baisse → coût performance dû à la sécurité.
- Si les **stalls** augmentent fortement quand `S_l` baisse → la contrainte rend l’ordonnancement difficile (voire impossible).

## 5) Où voir les résultats
### A) Runs individuels (DQN)
Chaque run écrit dans un dossier du type :
- `SingleAgent/results/resultDQN/DQN/<model>/sl_*/seed_*/train/`

Plots typiques :
- `dqn_training_history.png` (reward)
- `dqn_stalls_history.png` (stalls)
- `dqn_epsilon.png` (exploration)
- `execution_strategy.png` (devices choisis par couche sur un épisode “réussi”)

Fichiers utiles :
- `train/train_log.jsonl` : trace détaillée par épisode
- `run_config.json` : paramètres du run
- `summary.json` : métriques agrégées (avg last 100, etc.)

### B) Comparaison multi-`S_l`
Script :
- `SingleAgent/compare_sl.py`

Exemple (cnn15) :
- `python SingleAgent/compare_sl.py --algorithm DQN --model cnn15 --seed 42 --sl 0.5 0.33 0.2`

Sorties :
- `SingleAgent/results/resultDQN/DQN/cnn15/comparisons/seed_42/compare_metrics.png`
- `SingleAgent/results/resultDQN/DQN/cnn15/comparisons/seed_42/compare_reward_curves.png`
- `.../comparison_sl_*.json` (résumé chiffré)


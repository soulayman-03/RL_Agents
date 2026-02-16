# Analyse AC (Actor-Critic)

AC (Actor-Critic) dans ce projet apprend :
- une **politique** (choix du device),
- une **valeur** (estimation du return / de la qualité d’un état).

## 1) Reward
Fichiers :
- `SingleAgent/results/resultAC/train/ac_training_history.png`
- `SingleAgent/models/single_ac_train_history.npy`

Dans cet environnement, le reward = **− latence totale** (computation + communication). Donc :
- reward **moins négatif** / qui remonte vers `0` ⇒ **latence plus faible** ⇒ meilleur placement.

## 2) Stalls (violations)
Fichiers :
- `SingleAgent/results/resultAC/train/ac_stalls_history.png`
- `SingleAgent/models/single_ac_stall_history.npy`

Un *stall* correspond à une pénalité **−500** quand l’allocation viole une contrainte (mémoire / compute / bandwidth / privacy).
- stalls ≈ `0` ⇒ stratégie faisable et contraintes respectées.

## 3) Policy loss
Fichiers :
- `SingleAgent/results/resultAC/train/ac_policy_loss.png`
- `SingleAgent/models/single_ac_policy_loss.npy`

Ce n’est pas un “score” direct : il peut être positif/négatif et varier. À regarder surtout :
- stabilité (pas d’explosion),
- amélioration du reward en parallèle.

## 4) Value loss
Fichiers :
- `SingleAgent/results/resultAC/train/ac_value_loss.png`
- `SingleAgent/models/single_ac_value_loss.npy`

Mesure l’erreur entre `V(s)` et les retours observés.
- valeur loss qui diminue / se stabilise ⇒ critic plus précis ⇒ apprentissage plus stable.

## 5) Entropy
Fichiers :
- `SingleAgent/results/resultAC/train/ac_entropy.png`
- `SingleAgent/models/single_ac_entropy.npy`

Mesure l’exploration :
- entropie élevée ⇒ politique plus “random”,
- entropie qui baisse ⇒ politique plus déterministe (exploitation).

## 6) Stratégie d’exécution (trace)
Fichier :
- `SingleAgent/results/resultAC/train/execution_strategy.png`

Trace d’un épisode réussi : device choisi par layer, avec `C` (compute) et `T` (transmission).
- beaucoup de changements de device ⇒ risque d’augmenter `T` (communication),
- regroupement quand le réseau est “cher” ⇒ comportement cohérent.


# MultiAgentMADDPG_Energy (EnergyHard)

Ce dossier est une variante isolée de `MultiAgentMADDPG/` pour tester une **contrainte hard d’énergie** (batterie) par device.

## Idée

- Chaque device `d` a un budget d’énergie `E_d` au début de chaque épisode.
- Allouer une couche sur un device consomme :
  - `E_cost = alpha_comp * computation_demand + alpha_comm * transmission_data_size`
- Si `E_remaining(device) < E_cost` alors l’allocation **échoue** avec `reason="energy"` et l’agent est terminé (pénalité `-500`), ce qui permet de mesurer l’impact sur la convergence/récompense.

## Exécution

Exemple (3 agents, mêmes modèles, différents `S_l`) :

```bash
python -m MultiAgentMADDPG_Energy.train --models cnn15 --sl 1.0 0.5 0.33 --seed 42 --episodes 2000 --energy-min 500 --energy-max 1200
```

Résultats :
- Logs: `MultiAgentMADDPG_Energy/results/.../train_log.jsonl`
- Plots: `MultiAgentMADDPG_Energy/results/.../plots/*.png`
- Poids: `MultiAgentMADDPG_Energy/models/.../maddpg_actor_agent_*.pt`


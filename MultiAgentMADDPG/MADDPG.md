# MADDPG (CTDE) - Multi-Agent Actor-Critic

## Idee (CTDE)
- **Execution decentralisee** : chaque agent a un **actor** pi_i(o_i) qui voit seulement son observation locale.
- **Entrainement centralise** : chaque agent a un **critic** Q_i(x, a_1..a_n) qui voit l'etat global (concat des obs) et les actions de tous les agents.

## Etapes d'entrainement (resume)
1. **Collecte** : executer les actors (avec exploration) -> actions, puis `env.step`.
2. **Replay buffer** : stocker (obs, state, actions, rewards, next_obs, next_state, done, masks).
3. **Mise a jour critic** (pour chaque agent i) :
   - a'_j = pi'_j(o'_j) (actors cibles)
   - y_i = r_i + gamma * (1 - done_i) * Q'_i(x', a'_1..a'_n)
   - minimiser (Q_i(x, a_1..a_n) - y_i)^2
4. **Mise a jour actor** (pour chaque agent i) :
   - maximiser Q_i(x, a_1..pi_i(o_i)..a_n)
5. **Cibles (soft update)** :
   - theta' <- (1-tau) * theta' + tau * theta

## Notes pour ce projet (actions discretes)
L'environnement IoT choisit un **device** (action discrete). Pour garder un entrainement de type DDPG,
l'actor produit des **logits** et on utilise **Gumbel-Softmax** (straight-through) pendant l'entrainement.

## Fichiers
- `MultiAgentMADDPG/manager.py` : CTDE (replay + updates actor/critic).
- `MultiAgentMADDPG/agent.py` : actor/critic + target networks + epsilon.
- `MultiAgentMADDPG/replay_buffer.py` : buffer joint multi-agent.
- `MultiAgentMADDPG/train.py` : entrainement.
- `MultiAgentMADDPG/evaluate.py` : evaluation greedy.


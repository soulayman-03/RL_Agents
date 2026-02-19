# Comparaison des Algorithmes de Reinforcement Learning

Le passage de **DQN** vers d'autres types d'algorithmes (comme PPO ou des méthodes multi-agents comme VDN/QMIX) modifierait plusieurs aspects fondamentaux du projet.

## 1. Actor-Critic (ex: PPO, A2C)
Passer à une architecture Actor-Critic apporterait :
- **Stabilité** : PPO est généralement plus stable que DQN. DQN peut diverger brusquement, tandis que PPO limite les changements radicaux de politique via son "clipping".
- **Exploration** : Contrairement à l'epsilon-greedy de DQN (choix aléatoire), PPO explore via une distribution de probabilité, ce qui permet un apprentissage plus fin et nuancé.

## 2. CTDE (VDN, QMIX)
Actuellement, le projet utilise du **IQL (Independent Q-Learning)**. Chaque agent apprend de manière autonome. Passer à VDN ou QMIX changerait la dynamique :
- **Coopération Innée** : Le réseau apprendrait à maximiser la valeur Q-totale de l'équipe plutôt que de simples récompenses individuelles ou sociales.
- **Attribution du Crédit** : Ces algorithmes permettent d'identifier précisément quel agent est responsable d'un succès ou d'un échec, facilitant la résolution de conflits complexes.

## 3. Comparaison Technique

| Aspect | DQN (Actuel) | Actor-Critic / QMIX |
| :--- | :--- | :--- |
| **Complexité** | Simple, facile à débugger. | Architecture lourde (Mixer, Critic). |
| **Actions** | Très efficace pour les actions discrètes. | Meilleur pour les espaces continus (PPO). |
| **Architecture** | `State -> Q-Net -> Actions` | `State -> Actor -> Action` + `State -> Critic -> Value` |

### Schéma de Transition
- **DQN** : state → Q-network → Q-values → argmax action
- **Actor–Critic** : state → Actor → π(a|s) → action / state → Critic → V(s)

import sys
import os
import random

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MultiAgent.environment import MultiAgentIoTEnv

def test_async_execution():
    print("=== Test de l'Asynchronisme Multi-Agent ===")
    print("Agent 0: SimpleCNN (5 couches)")
    print("Agent 1: CNN7 (7 couches)")
    print("Agent 2: CNN10 (10 couches)")
    print("-" * 50)

    # 1. Initialisation de l'environnement hétérogène
    num_agents = 3
    num_devices = 5
    model_types = ["simplecnn", "cnn7", "cnn10"]
    env = MultiAgentIoTEnv(num_agents=num_agents, num_devices=num_devices, model_types=model_types, seed=42)

    obs, _ = env.reset()
    done = {i: False for i in range(num_agents)}
    step_count = 0

    while not all(done.values()):
        step_count += 1
        print(f"\n--- CYCLE (STEP) {step_count} ---")
        
        # Choix d'actions aléatoires pour la démo
        # (Dans un vrai système, c'est l'IA qui choisit)
        actions = {}
        valid_actions_dict = env.get_valid_actions()
        
        for i in range(num_agents):
            if not done[i]:
                # On choisit un device au hasard parmi les valides
                v_actions = valid_actions_dict[i]
                if v_actions:
                    actions[i] = random.choice(v_actions)
                else:
                    # Si aucun device n'est valide (cas rare de saturation totale)
                    actions[i] = 0 
        
        # 2. Exécution du step
        next_obs, rewards, next_done, truncated, infos = env.step(actions)
        
        # 3. Affichage du statut de chaque agent
        for i in range(num_agents):
            status = "EN TRAVAIL" if not done[i] else "FINI (Repos)"
            progress = env.agent_progress[i]
            total = len(env.tasks[i])
            
            if not done[i]:
                device = actions[i]
                print(f"Agent {i} ({model_types[i]}): {status} | Couche {progress}/{total} sur Device {device}")
            else:
                print(f"Agent {i} ({model_types[i]}): {status} | Tâche terminée")
        
        done = next_done
        
        if step_count > 15: # Sécurité
            break

    print("\n" + "="*50)
    print(f"Simulation terminée en {step_count} cycles.")
    print("Observation : L'épisode continue tant que CNN10 n'a pas fini.")
    print("="*50)

if __name__ == "__main__":
    test_async_execution()

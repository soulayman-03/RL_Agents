import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration esthétique
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.labelsize": 12
})

def smooth(scalars, weight=0.95):
    """Exponential smoothing for trends."""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def read_jsonl(path: str):
    data = []
    if not os.path.exists(path):
        print(f"[!] Fichier introuvable: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except: pass
    return data

def main():
    # Définition des dossiers
    path_base = r"c:\Users\soulaimane\Desktop\PFE\RL\MultiAgentMADDPG_DevicePowerEnv\results\models_1hugcnn_1cnn15_1miniresnet_1resnet18_1vgg11_1deepcnn_1lenet_p3_linFL_t0.7_e0.5k\sl_1p00"
    path_dvfs = r"c:\Users\soulaimane\Desktop\PFE\RL\MultiAgentMADDPG_DVFS_Physics_Env\results\avec\sl_1p00"
    
    out_dir = r"c:\Users\soulaimane\Desktop\PFE\RL\comparison_all_metrics"
    os.makedirs(out_dir, exist_ok=True)
    
    print("[*] Chargement des données Baseline (Sans DVFS)...")
    log_base = read_jsonl(os.path.join(path_base, "train_log.jsonl"))
    impact_base = read_jsonl(os.path.join(path_base, "episode_impact.jsonl"))
    
    print("[*] Chargement des données DVFS Physique...")
    log_dvfs = read_jsonl(os.path.join(path_dvfs, "train_log.jsonl"))
    impact_dvfs = read_jsonl(os.path.join(path_dvfs, "episode_impact.jsonl"))
    
    if not log_base or not log_dvfs:
        print("[!] Impossible de charger les données train_log.jsonl")
        return

    ep_base = [r.get("episode") for r in log_base]
    ep_dvfs = [r.get("episode") for r in log_dvfs]

    print("[*] Génération des graphiques comparatifs...")

    # 1. Total Energy Consumption
    plt.figure(figsize=(10, 6))
    en_base = smooth([r.get("energy_spent_total", 0.0) for r in log_base], 0.98)
    en_dvfs = smooth([r.get("energy_spent_total", 0.0) for r in log_dvfs], 0.98)
    plt.plot(ep_base, en_base, color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
    plt.plot(ep_dvfs, en_dvfs, color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
    plt.title("Comparaison : Consommation Énergétique Totale", fontsize=14)
    plt.xlabel("Épisodes")
    plt.ylabel("Énergie totale consommée (Joules)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_total_energy.png"), dpi=300)
    plt.close()

    # 2. Global Latency
    plt.figure(figsize=(10, 6))
    lat_base = smooth([sum(r.get("per_agent_latency_sum", {}).values()) for r in log_base], 0.98)
    lat_dvfs = smooth([sum(r.get("per_agent_latency_sum", {}).values()) for r in log_dvfs], 0.98)
    plt.plot(ep_base, lat_base, color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
    plt.plot(ep_dvfs, lat_dvfs, color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
    plt.title("Comparaison : Latence Globale du Système", fontsize=14)
    plt.xlabel("Épisodes")
    plt.ylabel("Latence Totale (secondes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_global_latency.png"), dpi=300)
    plt.close()

    # 3. Success Rate
    plt.figure(figsize=(10, 6))
    succ_base = smooth([1 if not r.get("ep_failed", False) else 0 for r in log_base], 0.95)
    succ_dvfs = smooth([1 if not r.get("ep_failed", False) else 0 for r in log_dvfs], 0.95)
    plt.plot(ep_base, succ_base, color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
    plt.plot(ep_dvfs, succ_dvfs, color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
    plt.title("Comparaison : Taux de Succès (Budget 200-500 J)", fontsize=14)
    plt.xlabel("Épisodes")
    plt.ylabel("Taux de Succès")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_success_rate.png"), dpi=300)
    plt.close()

    # 4. Global Queuing Delay (if impact data exists)
    if impact_base and impact_dvfs:
        plt.figure(figsize=(10, 6))
        
        ep_imp_base = [r.get("episode") for r in impact_base]
        wait_base = []
        for row in impact_base:
            w = sum([alloc.get("t_comp_wait", 0.0) + alloc.get("t_comm_wait", 0.0) for alloc in row.get("allocations", [])])
            wait_base.append(w)
            
        ep_imp_dvfs = [r.get("episode") for r in impact_dvfs]
        wait_dvfs = []
        for row in impact_dvfs:
            w = sum([alloc.get("t_comp_wait", 0.0) + alloc.get("t_comm_wait", 0.0) for alloc in row.get("allocations", [])])
            wait_dvfs.append(w)

        plt.plot(ep_imp_base, smooth(wait_base, 0.98), color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
        plt.plot(ep_imp_dvfs, smooth(wait_dvfs, 0.98), color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
        plt.title("Comparaison : Délai d'Attente Total (Queuing Delay)", fontsize=14)
        plt.xlabel("Épisodes")
        plt.ylabel("Délai d'attente cumulé (sec)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comp_queuing_delay.png"), dpi=300)
        plt.close()

    # 5. Device Survival (Average Battery %)
    plt.figure(figsize=(10, 6))
    
    def get_avg_survival(log_data):
        avg_surv = []
        for row in log_data:
            init = row.get("device_energy_init", {})
            rem = row.get("device_energy_remaining", {})
            pcts = []
            for d in init:
                i_val = float(init[d])
                r_val = float(rem.get(d, 0.0))
                pcts.append((r_val / i_val * 100.0) if i_val > 0 else 100.0)
            avg_surv.append(np.mean(pcts) if pcts else 100.0)
        return avg_surv

    surv_base = smooth(get_avg_survival(log_base), 0.98)
    surv_dvfs = smooth(get_avg_survival(log_dvfs), 0.98)
    
    plt.plot(ep_base, surv_base, color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
    plt.plot(ep_dvfs, surv_dvfs, color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.title("Comparaison : Survie Énergétique Moyenne (Batterie Restante)", fontsize=14)
    plt.xlabel("Épisodes")
    plt.ylabel("Batterie Restante Moyenne (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_battery_survival.png"), dpi=300)
    plt.close()

    # 6. Rewards Comparison
    plt.figure(figsize=(10, 6))
    reward_base = [row.get("team_reward_sum", 0.0) for row in log_base]
    reward_dvfs = [row.get("team_reward_sum", 0.0) for row in log_dvfs]
    
    plt.plot(ep_base, smooth(reward_base, 0.98), color="#EE5253", linewidth=2.5, label="Baseline (Sans DVFS)")
    plt.plot(ep_dvfs, smooth(reward_dvfs, 0.98), color="#1dd1a1", linewidth=2.5, label="DVFS Physique (v³)")
    plt.title("Comparaison : Récompenses d'Équipe (Team Reward)", fontsize=14)
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense Cumulée")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comp_rewards.png"), dpi=300)
    plt.close()

    print(f"\n[+] Succès ! Les graphiques de comparaison ont été générés dans :\n{out_dir}")

if __name__ == "__main__":
    main()

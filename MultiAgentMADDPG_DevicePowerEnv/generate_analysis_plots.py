import os
import json
import argparse
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "Arial"],
    "figure.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.labelweight": "medium"
})

def smooth(scalars, weight=0.9):
    """Exponential smoothing for trends."""
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def _get_latest_results_dir(base_dir: str) -> str:
    # Look for the most recently modified episode_impact.jsonl
    search_pattern = os.path.join(base_dir, "results", "**", "episode_impact.jsonl")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise ValueError("No 'episode_impact.jsonl' found in 'results/' directory.")
    latest_file = max(files, key=os.path.getmtime)
    return os.path.dirname(latest_file)

def plot_queuing_delay(impact_data, out_path):
    """Generates the evolution of Queuing Delay (Total waiting time per episode)."""
    episodes = []
    total_waits = []
    
    for row in impact_data:
        ep = row.get("episode", 0)
        waits = [alloc.get("t_comp_wait", 0.0) + alloc.get("t_comm_wait", 0.0) for alloc in row.get("allocations", [])]
        total_wait = sum(waits)
        episodes.append(ep)
        total_waits.append(total_wait)
        
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_waits, alpha=0.2, color="#FF9F43", label="Raw Queuing Delay")
    plt.plot(episodes, smooth(total_waits, 0.98), color="#EE5253", linewidth=2, label="Lissée (Trend)")
    plt.title("Évolution du Processus de Délai d'Attente (Waiting Process)", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Délai d'attente cumulé (sec)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Queuing Delay chart saved to {out_path}")

def plot_device_queuing_delay(impact_data, out_path):
    """Generates a line plot showing the queuing delay evolution per device."""
    episodes = [row.get("episode", 0) for row in impact_data]
    
    # Identify unique devices
    devices = set()
    for row in impact_data:
        for alloc in row.get("allocations", []):
            d_id = alloc.get("device_before", {}).get("device_id")
            if d_id is not None:
                devices.add(int(d_id))
                
    if not devices:
        return
        
    ordered_devices = sorted(list(devices))
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(ordered_devices))
    
    for i, d_id in enumerate(ordered_devices):
        waits_per_ep = []
        for row in impact_data:
            ep_wait = 0.0
            for alloc in row.get("allocations", []):
                ad_id = alloc.get("device_before", {}).get("device_id")
                if ad_id is not None and int(ad_id) == d_id:
                    ep_wait += alloc.get("t_comp_wait", 0.0) + alloc.get("t_comm_wait", 0.0)
            waits_per_ep.append(ep_wait)
            
        plt.plot(episodes, smooth(waits_per_ep, 0.98), color=colors[i], linewidth=2, label=f"Terminal {d_id}")
        
    plt.title("Évolution du Délai d'Attente (Queuing Delay) par Terminal", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Délai d'attente cumulé (sec)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Device Queuing Delay chart saved to {out_path}")

def plot_total_energy_consumption(log_data, out_path):
    """Generates the evolution of Total Energy Consumption per episode."""
    episodes = [row.get("episode", 0) for row in log_data]
    energy = [row.get("energy_spent_total", 0.0) for row in log_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, energy, alpha=0.2, color="#48dbfb", label="Raw Energy Consumption")
    plt.plot(episodes, smooth(energy, 0.98), color="#1dd1a1", linewidth=2.5, label="Trend (Lissage)")
    plt.title("Consommation Énergétique Totale du Système", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Énergie totale consommée (Joules)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Total Energy Consumption chart saved to {out_path}")

def plot_global_latency(log_data, out_path):
    """Generates the evolution of Global Latency (Sum of all agent latencies)."""
    episodes = [row.get("episode", 0) for row in log_data]
    
    # Global latency is the sum of per-agent latency sums
    global_latency = []
    for row in log_data:
        per_agent_lat = row.get("per_agent_latency_sum", {})
        global_lat = sum(per_agent_lat.values()) if isinstance(per_agent_lat, dict) else 0.0
        global_latency.append(global_lat)
        
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, global_latency, alpha=0.2, color="#5f27cd", label="Raw Global Latency")
    plt.plot(episodes, smooth(global_latency, 0.98), color="#341f97", linewidth=2.5, label="Trend (Lissage)")
    plt.title("Évolution de la Latence Globale du Système", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Latence Totale (secondes)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Global Latency chart saved to {out_path}")

def plot_success_vs_penalties(log_data, out_path):
    """Generates a chart showing Success Rate vs Penalty/Failure reasons."""
    episodes = [row.get("episode", 0) for row in log_data]
    successes = [1 if not row.get("ep_failed", False) else 0 for row in log_data]
    sr_smoothed = smooth(successes, 0.95)
    
    # Failure reasons aggregation
    reasons_list = []
    all_reasons = set()
    for row in log_data:
        fr = row.get("fail_reasons", {})
        reasons_list.append(fr)
        all_reasons.update(fr.keys())
    
    ordered_reasons = sorted(list(all_reasons))
    reason_series = {r: [] for r in ordered_reasons}
    for fr in reasons_list:
        for r in ordered_reasons:
            reason_series[r].append(fr.get(r, 0))
            
    # Smoothing reason series
    smoothed_reasons = {r: smooth(reason_series[r], 0.98) for r in ordered_reasons}
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Success Rate
    ax1.plot(episodes, sr_smoothed, color="#1dd1a1", linewidth=3, label="Taux de Succès (Lissé)")
    ax1.set_xlabel("Épisodes", fontsize=12)
    ax1.set_ylabel("Taux de Succès", fontsize=12, color="#1dd1a1")
    ax1.tick_params(axis='y', labelcolor="#1dd1a1")
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot Failure Reasons (Penalties) on secondary axis
    ax2 = ax1.twinx()
    colors = sns.color_palette("rocket", len(ordered_reasons))
    bottom = np.zeros(len(episodes))
    
    for i, r in enumerate(ordered_reasons):
        ax2.fill_between(episodes, bottom, bottom + np.array(smoothed_reasons[r]), label=f"Penalité: {r}", color=colors[i], alpha=0.6)
        bottom += np.array(smoothed_reasons[r])
        
    ax2.set_ylabel("Fréquence des Échecs / Pénalités", fontsize=12, color="#EE5253")
    ax2.tick_params(axis='y', labelcolor="#EE5253")
    
    plt.title("Performance Globale : Taux de Succès vs Pénalités des Contraintes", fontsize=14)
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Success vs Penalties chart saved to {out_path}")

def plot_device_energy_evolution(log_data, out_path):
    """Generates a line plot showing the evolution of energy consumption per device."""
    episodes = [row.get("episode", 0) for row in log_data]
    
    # Extract devices from the first row that has energy data
    devices = []
    for row in log_data:
        init = row.get("device_energy_init", {})
        if init:
            devices = sorted(list(init.keys()), key=lambda x: int(x))
            break
            
    if not devices:
        return

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(devices))
    
    for i, d_id in enumerate(devices):
        consumptions = []
        for row in log_data:
            init = float(row.get("device_energy_init", {}).get(d_id, 0.0))
            rem = float(row.get("device_energy_remaining", {}).get(d_id, 0.0))
            consumptions.append(max(0.0, init - rem))
        
        plt.plot(episodes, smooth(consumptions, 0.98), color=colors[i], linewidth=2, label=f"Terminal {d_id}")
        
    plt.title("Évolution de la Consommation Énergétique par Terminal", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Énergie consommée (Joules)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Device Energy Evolution chart saved to {out_path}")

def plot_device_survival_percentage(log_data, out_path):
    """Generates a line plot showing the survival percentage (100% -> X%) of each device's battery over episodes."""
    episodes = [row.get("episode", 0) for row in log_data]
    
    # Extract devices from the first row that has energy data
    devices = []
    for row in log_data:
        init = row.get("device_energy_init", {})
        if init:
            devices = sorted(list(init.keys()), key=lambda x: int(x))
            break
            
    if not devices:
        return

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(devices))
    
    for i, d_id in enumerate(devices):
        percentages = []
        for row in log_data:
            init = float(row.get("device_energy_init", {}).get(d_id, 0.0))
            rem = float(row.get("device_energy_remaining", {}).get(d_id, 0.0))
            if init > 0:
                pct = (rem / init) * 100.0
            else:
                pct = 100.0
            percentages.append(pct)
        
        plt.plot(episodes, smooth(percentages, 0.98), color=colors[i], linewidth=2.5, label=f"Device {d_id}")
        
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.title("Taux de Survie Énergétique par Terminal (Évolution des Batteries)", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Batterie Restante (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Device Battery Trend chart saved to {out_path}")

def plot_global_comp_vs_comm_latency(impact_data, out_path):
    """Generates a line plot showing global computation latency vs communication latency."""
    episodes = []
    comp_latencies = []
    comm_latencies = []
    
    for row in impact_data:
        ep = row.get("episode", 0)
        episodes.append(ep)
        
        # Sum comp and comm latency for all allocations in this episode
        comp_sum = sum(float(alloc.get("t_comp", 0.0)) for alloc in row.get("allocations", []))
        comm_sum = sum(float(alloc.get("t_comm", 0.0)) for alloc in row.get("allocations", []))
        
        comp_latencies.append(comp_sum)
        comm_latencies.append(comm_sum)
        
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, comp_latencies, alpha=0.15, color="#1dd1a1", label="Raw Computation Latency")
    plt.plot(episodes, smooth(comp_latencies, 0.98), color="#10ac84", linewidth=2.5, label="Computation Latency (Smoothed)")
    
    plt.plot(episodes, comm_latencies, alpha=0.15, color="#ff9f43", label="Raw Communication Latency")
    plt.plot(episodes, smooth(comm_latencies, 0.98), color="#ee5253", linewidth=2.5, label="Communication Latency (Smoothed)")
    
    plt.title("Évolution Globale de la Latence : Calcul vs Communication", fontsize=14)
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Latence Totale (secondes)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Global Computation vs Communication Latency chart saved to {out_path}")

def plot_agent_comp_vs_comm_latency(impact_data, out_path):
    """Generates a bar plot comparing computation and communication latency per agent (last 100 episodes)."""
    # Identify unique agents and their model names
    agents = set()
    agent_models = {}
    for row in impact_data:
        for alloc in row.get("allocations", []):
            if alloc.get("agent") is not None:
                a_id = int(alloc.get("agent"))
                agents.add(a_id)
                if alloc.get("model"):
                    agent_models[a_id] = alloc.get("model")
                    
    if not agents:
        return
        
    ordered_agents = sorted(list(agents))
    
    # Analyze the last 100 episodes
    last_100_rows = impact_data[-100:] if len(impact_data) >= 100 else impact_data
    
    agent_comp_sum = {a: 0.0 for a in ordered_agents}
    agent_comm_sum = {a: 0.0 for a in ordered_agents}
    agent_counts = {a: 0 for a in ordered_agents}
    
    for row in last_100_rows:
        for alloc in row.get("allocations", []):
            a_id = alloc.get("agent")
            if a_id is not None:
                a_id = int(a_id)
                agent_comp_sum[a_id] += float(alloc.get("t_comp", 0.0))
                agent_comm_sum[a_id] += float(alloc.get("t_comm", 0.0))
        for a_id in ordered_agents:
            agent_counts[a_id] += 1
            
    # Compute averages
    avg_comp = [agent_comp_sum[a] / max(1, agent_counts[a]) for a in ordered_agents]
    avg_comm = [agent_comm_sum[a] / max(1, agent_counts[a]) for a in ordered_agents]
    
    labels = [f"Agent {a}\n({agent_models.get(a, 'Unknown')})" for a in ordered_agents]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, avg_comp, width, label='Calcul (Computation)', color='#10ac84')
    rects2 = ax.bar(x + width/2, avg_comm, width, label='Communication', color='#ff9f43')
    
    ax.set_ylabel('Latence moyenne par épisode (secondes)', fontsize=12)
    ax.set_title('Répartition Latence de Calcul vs Communication par Agent (Moyenne 100 derniers épisodes)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Agent Computation vs Communication Latency chart saved to {out_path}")

# Additional plots from the DVFS BalancedEnv version

def plot_convergence_recompenses(log_data, out_path):
    """Figure 6 : Convergence des récompenses cumulées"""
    episodes = [row.get("episode", 0) for row in log_data]
    models = log_data[0].get("models", []) if log_data else []
    num_agents = len(models)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", max(1, num_agents))
    
    for a in range(num_agents):
        rewards = [row.get("per_agent_reward_sum", {}).get(str(a), 0.0) for row in log_data]
        smoothed_rewards = smooth(rewards, 0.98)
        model_name = models[a] if a < len(models) else f"Agent {a}"
        plt.plot(episodes, smoothed_rewards, color=colors[a], linewidth=2, label=f"Agent {a} ({model_name})")
        
    plt.title("Figure 6 : Convergence des récompenses cumulées", fontsize=14)
    plt.xlabel("Épisodes d'entraînement", fontsize=12)
    plt.ylabel("Récompense cumulée moyenne", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Figure 6 saved to {out_path}")

def plot_respect_contraintes(log_data, out_path):
    """Figure 7 : Capacité à respecter les contraintes"""
    episodes = [row.get("episode", 0) for row in log_data]
    
    global_success = [1.0 if not row.get("ep_failed", False) else 0.0 for row in log_data]
    smoothed_success = smooth(global_success, 0.95)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_success, color="#1dd1a1", linewidth=3, label="Taux de respect")
    plt.title("Figure 7 : Capacité à respecter les contraintes", fontsize=14)
    plt.xlabel("Épisodes d'entraînement", fontsize=12)
    plt.ylabel("Taux de respect des contraintes", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Figure 7 saved to {out_path}")

def plot_convergence_latence(log_data, out_path):
    """Figure 8 : Convergence des pénalités de latence"""
    episodes = [row.get("episode", 0) for row in log_data]
    models = log_data[0].get("models", []) if log_data else []
    num_agents = len(models)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", max(1, num_agents))
    
    for a in range(num_agents):
        latencies = [row.get("per_agent_latency_sum", {}).get(str(a), 0.0) for row in log_data]
        smoothed_latencies = smooth(latencies, 0.98)
        model_name = models[a] if a < len(models) else f"Agent {a}"
        plt.plot(episodes, smoothed_latencies, color=colors[a], linewidth=2, label=f"Agent {a} ({model_name})")
        
    plt.title("Figure 8 : Convergence des pénalités de latence", fontsize=14)
    plt.xlabel("Épisodes d'entraînement", fontsize=12)
    plt.ylabel("Pénalité de latence / Temps (sec)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Figure 8 saved to {out_path}")

def plot_donnees_partagees(impact_data, out_path):
    """Figure 9 : Convergence des données partagées"""
    episodes = [row.get("episode", 0) for row in impact_data]
    
    all_agents = set()
    agent_models = {}
    for row in impact_data:
        for alloc in row.get("allocations", []):
            if alloc.get("agent") is not None:
                a = int(alloc.get("agent"))
                all_agents.add(a)
                if alloc.get("model"):
                    agent_models[a] = alloc.get("model")
                
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("tab10", max(1, len(all_agents)))
    
    for i, a in enumerate(sorted(list(all_agents))):
        data_per_ep = []
        for row in impact_data:
            ep_sum = 0.0
            for alloc in row.get("allocations", []):
                if alloc.get("agent") is not None and int(alloc.get("agent")) == a:
                    ep_sum += float(alloc.get("trans_data", 0.0))
            data_per_ep.append(ep_sum)
            
        smoothed_data = smooth(data_per_ep, 0.98)
        model_name = agent_models.get(a, f"Agent {a}")
        plt.plot(episodes, smoothed_data, color=colors[i % len(colors)], linewidth=2, label=f"Agent {a} ({model_name})")
        
    plt.title("Figure 9 : Convergence des données partagées", fontsize=14)
    plt.xlabel("Épisodes d'entraînement", fontsize=12)
    plt.ylabel("Quantité de données partagées", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Figure 9 saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Génère des graphiques d'analyse pour MADDPG")
    parser.add_argument("--results-dir", type=str, default=None, help="Chemin vers le dossier de résultats")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.results_dir:
        res_dir = args.results_dir
    else:
        try:
            res_dir = _get_latest_results_dir(project_dir)
        except Exception as e:
            print(f"[!] Erreur : {e}")
            sys.exit(1)
            
    print(f"[*] Analyse des résultats dans : {res_dir}")
    train_log_file = os.path.join(res_dir, "train_log.jsonl")
    impact_file = os.path.join(res_dir, "episode_impact.jsonl")
    
    log_data = []
    if os.path.exists(train_log_file):
        with open(train_log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        log_data.append(json.loads(line))
                    except: pass
                    
    impact_data = []
    if os.path.exists(impact_file):
        with open(impact_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        impact_data.append(json.loads(line))
                    except: pass
                    
    if not log_data and not impact_data:
        print("[!] Aucune donnée trouvée.")
        sys.exit(1)
        
    plots_dir = os.path.join(res_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"[*] Génération des graphiques...")
    
    if impact_data:
        plot_queuing_delay(impact_data, os.path.join(plots_dir, "queuing_delay_evolution.png"))
        plot_device_queuing_delay(impact_data, os.path.join(plots_dir, "device_queuing_delay.png"))
        plot_donnees_partagees(impact_data, os.path.join(plots_dir, "figure_9_donnees_partagees.png"))
        plot_global_comp_vs_comm_latency(impact_data, os.path.join(plots_dir, "global_comp_vs_comm_latency.png"))
        plot_agent_comp_vs_comm_latency(impact_data, os.path.join(plots_dir, "agent_comp_vs_comm_latency.png"))
        
    if log_data:
        plot_total_energy_consumption(log_data, os.path.join(plots_dir, "total_energy_consumption.png"))
        plot_device_energy_evolution(log_data, os.path.join(plots_dir, "device_energy_evolution.png"))
        plot_device_survival_percentage(log_data, os.path.join(plots_dir, "device_battery_trends.png"))
        plot_global_latency(log_data, os.path.join(plots_dir, "global_latency.png"))
        plot_success_vs_penalties(log_data, os.path.join(plots_dir, "success_vs_penalties.png"))
        plot_convergence_recompenses(log_data, os.path.join(plots_dir, "figure_6_convergence_recompenses.png"))
        plot_respect_contraintes(log_data, os.path.join(plots_dir, "figure_7_respect_contraintes.png"))
        plot_convergence_latence(log_data, os.path.join(plots_dir, "figure_8_convergence_latence.png"))
    
    print(f"\n[+] Succès ! Les graphiques ont été sauvegardés dans : {plots_dir}")

if __name__ == "__main__":
    main()

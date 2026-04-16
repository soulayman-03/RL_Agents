import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(scalars, weight=0.9):
    """Lissage exponentiel pour les courbes d'évolution."""
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
    # Trouve le dossier contenant 'train_log.jsonl' modifié le plus récemment
    search_pattern = os.path.join(base_dir, "results", "**", "episode_impact.jsonl")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise ValueError("Aucun fichier 'episode_impact.jsonl' trouvé dans le dossier 'results/'.")
    latest_file = max(files, key=os.path.getmtime)
    return os.path.dirname(latest_file)

def plot_queuing_delay(impact_data, out_path):
    """Génère la courbe d'évolution du Queuing Delay (Délai d'attente total par épisode)."""
    episodes = []
    total_waits = []
    
    for row in impact_data:
        ep = row.get("episode", 0)
        waits = [alloc.get("t_comp_wait", 0.0) + alloc.get("t_comm_wait", 0.0) for alloc in row.get("allocations", [])]
        total_wait = sum(waits)
        episodes.append(ep)
        total_waits.append(total_wait)
        
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_waits, alpha=0.3, color="orange", label="Total Queuing Delay (Brut)")
    plt.plot(episodes, smooth(total_waits, 0.95), color="red", linewidth=2.5, label="Tendance Lissée")
    plt.title("Apprentissage du Load Balancing : Évolution du Queuing Delay", fontsize=14, fontweight="bold")
    plt.xlabel("Épisodes", fontsize=12)
    plt.ylabel("Délai d'attente cumulé (secondes)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[1/3] Queuing Delay chart saved to {out_path}")

def plot_energy_fairness(impact_data, out_path):
    """Génère un graphique à barres du pourcentage d'énergie restante par Device en fin d'épisode."""
    recent_data = impact_data[-min(100, len(impact_data)):]
    device_energy_pct = {}  # {device_id: [pct, pct, ...]}
    
    for row in recent_data:
        init_energy = row.get("device_energy_init", {})
        final_energy = row.get("device_energy_final", {})
        for d_id, init_val in init_energy.items():
            if init_val > 0:
                final_val = float(final_energy.get(d_id, 0.0))
                pct = (final_val / float(init_val)) * 100.0
                if d_id not in device_energy_pct:
                    device_energy_pct[d_id] = []
                device_energy_pct[d_id].append(pct)
                
    if not device_energy_pct:
        print("Aucune donnée d'énergie dans les épisodes récents.")
        return
        
    devices = sorted(list(device_energy_pct.keys()), key=lambda x: int(x))
    avg_pct = [np.mean(device_energy_pct[d]) for d in devices]
    std_pct = [np.std(device_energy_pct[d]) for d in devices]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(devices, avg_pct, yerr=std_pct, capsize=5, color="mediumseagreen", alpha=0.8, edgecolor="black")
    plt.axhline(y=100.0, color='grey', linestyle=':', label='100% (Batterie Pleine)')
    plt.axhline(y=0.0, color='red', linestyle='-', label='0% (Batterie Vide)')
    plt.ylim(0, 105)
    
    # Text on bars
    for i, b in enumerate(bars):
        height = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., height - 10,
                 f'{height:.1f}%',
                 ha='center', va='bottom', color='white', fontweight='bold')

    plt.title("Distribution de Survie Énergétique \n(100 derniers épisodes)", fontsize=14, fontweight="bold")
    plt.xlabel("Terminal Numéro (Device ID)", fontsize=12)
    plt.ylabel("Énergie Restante en fin de tâche (%)", fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[3/4] Energy Fairness (100 derniers) chart saved to {out_path}")

def plot_energy_fairness_last_episode(impact_data, out_path):
    """Génère un graphique à barres du pourcentage d'énergie restante par Device pour le tout dernier épisode."""
    if not impact_data:
        return
        
    last_row = impact_data[-1]
    
    device_energy_pct = {}
    init_energy = last_row.get("device_energy_init", {})
    final_energy = last_row.get("device_energy_final", {})
    
    for d_id, init_val in init_energy.items():
        if init_val > 0:
            final_val = float(final_energy.get(d_id, 0.0))
            pct = (final_val / float(init_val)) * 100.0
            device_energy_pct[d_id] = pct
            
    if not device_energy_pct:
        print("Aucune donnée d'énergie dans le dernier épisode.")
        return
        
    devices = sorted(list(device_energy_pct.keys()), key=lambda x: int(x))
    pct_values = [device_energy_pct[d] for d in devices]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(devices, pct_values, color="royalblue", alpha=0.8, edgecolor="black")
    plt.axhline(y=100.0, color='grey', linestyle=':', label='100% (Batterie Pleine)')
    plt.axhline(y=0.0, color='red', linestyle='-', label='0% (Batterie Vide)')
    plt.ylim(0, 105)
    
    # Text on bars
    for i, b in enumerate(bars):
        height = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., height - 10,
                 f'{height:.1f}%',
                 ha='center', va='bottom', color='white', fontweight='bold')

    ep_num = last_row.get("episode", "?")
    plt.title(f"Survie Énergétique au Dernier Épisode (Ep: {ep_num})", fontsize=14, fontweight="bold")
    plt.xlabel("Terminal Numéro (Device ID)", fontsize=12)
    plt.ylabel("Énergie Restante (%)", fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[4/4] Energy Fairness (Last Episode) chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Génère des graphiques d'analyse pour MADDPG")
    parser.add_argument("--results-dir", type=str, default=None, help="Chemin explicite vers le dossier contenant episode_impact.jsonl")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.results_dir:
        res_dir = args.results_dir
    else:
        try:
            res_dir = _get_latest_results_dir(project_dir)
        except Exception as e:
            print(e)
            sys.exit(1)
            
    print(f"[*] Lecture des données depuis : {res_dir}")
    impact_file = os.path.join(res_dir, "episode_impact.jsonl")
    
    data = []
    with open(impact_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not data:
        print("[!] Le fichier JSONL est vide ou n'a pas pu être lu.")
        sys.exit(1)
        
    plots_dir = os.path.join(res_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"[*] Génération des {len(data)} épisodes lus...")
    plot_queuing_delay(data, os.path.join(plots_dir, "queuing_delay_evolution.png"))
    plot_energy_fairness(data, os.path.join(plots_dir, "energy_fairness_100_last.png"))
    plot_energy_fairness_last_episode(data, os.path.join(plots_dir, "energy_fairness_last_episode.png"))
    
    print(f"\n[+] Succès ! Les graphiques ont été sauvegardés dans : {plots_dir}")

if __name__ == "__main__":
    main()

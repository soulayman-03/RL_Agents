import os
import glob
import subprocess
import sys

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Trouver tous les environnements qui ont un generate_analysis_plots.py
    env_dirs = [
        "MultiAgentMADDPG_DVFS_Physics_Env",
        "MultiAgentMADDPG_DevicePowerEnv"
    ]
    
    for env_dir in env_dirs:
        env_path = os.path.join(project_root, env_dir)
        plot_script = os.path.join(env_path, "generate_analysis_plots.py")
        
        if not os.path.exists(plot_script):
            continue
            
        print(f"\n{'='*50}\nRecherche de résultats dans : {env_dir}\n{'='*50}")
        
        # Chercher tous les sous-dossiers contenant train_log.jsonl
        search_pattern = os.path.join(env_path, "results", "**", "train_log.jsonl")
        log_files = glob.glob(search_pattern, recursive=True)
        
        if not log_files:
            print("Aucun fichier train_log.jsonl trouvé.")
            continue
            
        for log_file in log_files:
            result_dir = os.path.dirname(log_file)
            print(f"\n[*] Traitement du dossier : {os.path.relpath(result_dir, project_root)}")
            
            try:
                # Appeler generate_analysis_plots.py avec le bon dossier
                subprocess.run(
                    [sys.executable, plot_script, "--results-dir", result_dir],
                    cwd=env_path,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[!] Erreur lors de la génération des graphiques pour {result_dir} : {e}")

if __name__ == "__main__":
    main()

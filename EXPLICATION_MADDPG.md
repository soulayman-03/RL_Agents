# Explication Détaillée du Modèle MADDPG Appliqué au Projet

Ce document présente une explication théorique et technique de l'algorithme **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) tel qu'il est implémenté et utilisé dans ce projet pour l'optimisation conjointe du placement des couches de réseaux de neurones (DNN) et du réglage DVFS dans un environnement IoT multi-agents.

---

## 1. Introduction à MADDPG et Philosophie CTDE

L'algorithme **MADDPG** est une extension du modèle DDPG (un algorithme d'apprentissage par renforcement pour espaces d'actions continus) adapté aux environnements multi-agents. Il repose sur le paradigme **CTDE** (*Centralized Training with Decentralized Execution* - Entraînement Centralisé et Exécution Décentralisée) :

*   **Entraînement Centralisé (Centralized Training) :** Durant la phase d'apprentissage, l'algorithme dispose d'une vue globale de l'environnement. Le **Critique** de chaque agent a accès aux observations de *tous* les agents ainsi qu'aux actions de *tous* les agents. Cela permet de lever le problème de **non-stationnarité** inhérent aux systèmes multi-agents (où l'environnement change constamment du point de vue d'un agent individuel car les autres agents mettent à jour leurs politiques simultanément).
*   **Exécution Décentralisée (Decentralized Execution) :** Lors du déploiement ou de l'évaluation (la prise de décision en temps réel), seul l'**Acteur** de chaque agent est utilisé. Chaque agent choisit son action en se basant uniquement sur son **observation locale** ($o_i$), sans avoir besoin de connaître les observations ou les décisions des autres agents à cet instant.

---

## 2. Architecture des Réseaux et Fonctionnement

Pour chaque agent $i \in \{1, \dots, N\}$, l'architecture MADDPG se compose de 4 réseaux de neurones distincts :

1.  **L'Acteur local ($\mu_i$ avec paramètres $\theta_i$) :**
    *   **Entrée :** L'observation locale $o_i$ de l'agent.
    *   **Sortie :** Les logits associés aux actions discrètes de taille $M \times |DVFS|$ (nombre d'objets IoT $\times$ nombre de fréquences CPU DVFS).
2.  **L'Acteur Cible (Target Actor $\mu'_i$ avec paramètres $\theta'_i$) :**
    *   Copie retardée de l'acteur local, utilisée pour stabiliser le calcul des cibles temporelles (Target Q-values).
3.  **Le Critique centralisé ($Q_i$ avec paramètres $\phi_i$) :**
    *   **Entrée :** L'état global $s$ (concaténation de toutes les observations $[o_1, o_2, \dots, o_N]$) et l'action conjointe $a = [a_1, a_2, \dots, a_N]$ représentée sous forme de vecteurs *one-hot*.
    *   **Sortie :** Une valeur scalaire $Q_i(s, a_1, \dots, a_N)$ représentant la qualité de l'action conjointe sous cet état.
4.  **Le Critique Cible (Target Critic $Q'_i$ avec paramètres $\phi'_i$) :**
    *   Copie retardée du critique centralisé, utilisée pour stabiliser l'apprentissage.

---

## 3. Spécificités Techniques de l'Implémentation du Projet

### A. Gumbel-Softmax avec Masquage (Masked Gumbel-Softmax)
Puisque MADDPG repose à l'origine sur des gradients de politique déterministes continus, l'adaptation à un espace d'actions **discrètes** nécessite une approximation différentiable pour permettre la rétropropagation du gradient du Critique vers l'Acteur.
*   **Gumbel-Softmax** est utilisé pour générer une approximation continue et différentiable d'une distribution catégorielle dure :
    $$\tilde{a}_i = \text{Softmax}\left( \frac{\log(\pi_i(o_i)) + G_i}{\tau_{Gumbel}} \right)$$
    où $G_i$ est un bruit de Gumbel standard et $\tau_{Gumbel}$ est la température.
*   **Masquage des actions invalides :** Pour respecter les contraintes dures du système (comme la mémoire vive maximale des objets IoT, la batterie résiduelle ou la contrainte de confiance/sécurité), les actions invalides sont éliminées. Dans le code, les logits des actions interdites sont forcés à $-\infty$ (ou $-10^9$) avant de passer par le Gumbel-Softmax, garantissant que la probabilité de sélection de ces actions soit rigoureusement nulle :
    $$\text{Logits}_{\text{effective}} = \text{where}(\text{Mask}_{\text{valid}}, \text{Logits}, -1e9)$$

### B. Masque d'Activité (`active_i`)
Étant donné que les modèles DNN des agents ont des nombres de couches différents (ex. LeNet a moins de couches que ResNet18), certains agents terminent leur processus d'allocation de couches plus tôt que d'autres. Le manager applique un masque `active` lors de l'apprentissage :
*   Si un agent $i$ a déjà terminé ou échoué lors de cette étape de l'épisode, sa perte de critique et sa perte d'acteur ne contribuent pas à la mise à jour des gradients, évitant ainsi d'entraîner le réseau sur des états factices ou inactifs.

### C. Récompense Multi-Objectif et Coopérative
*   **Objectif local (success) :** Minimiser le temps de traitement $T_{\text{total}} = t_{\text{comp}} + t_{\text{comm}}$ et le coût énergétique de l'allocation $E_{\text{total}} = E_{\text{comp}} + E_{\text{comm}}$.
    $$R_i^{\text{raw}} = -(\alpha \cdot T_{\text{total}} + \beta \cdot E_{\text{total}})$$
*   **Coopération :** Pour encourager la collaboration dans le partage des ressources physiques (mémoire, bande passante), la récompense finale d'un agent mélange son score individuel avec la moyenne du groupe :
    $$R_i = 0.7 \cdot R_i^{\text{raw}} + 0.3 \cdot \left( \frac{1}{N} \sum_{j=1}^{N} R_j^{\text{raw}} \right)$$
*   **Pénalité d'échec :** Si une contrainte dure est violée (mémoire saturée, énergie épuisée sur un appareil, ou manque de confiance requis pour la confidentialité), l'agent reçoit une forte pénalité de $-500.0$ et l'épisode s'arrête (pour l'agent en question).

---

## 4. Pseudocode de l'algorithme MADDPG du Projet

Le pseudocode suivant détaille le processus d'entraînement centralisé de bout en bout implémenté dans [manager.py](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/manager.py) et [train.py](file:///c:/Users/soulaimane/Desktop/PFE/RL/MultiAgentMADDPG_DVFS_BalancedEnv/train.py) :

```python
# =========================================================================
# PSEUDOCODE : ENTRAÎNEMENT MADDPG AVEC CONTRAINTES ET DVFS
# =========================================================================

Initialiser le Replay Buffer D de capacité maximale
Pour chaque agent i de 1 à N:
    Initialiser le réseau Acteur local μ_i(o_i; θ_i) avec des poids aléatoires
    Initialiser le réseau Critique centralisé Q_i(s, a_1...a_N; φ_i) avec des poids aléatoires
    Initialiser les réseaux Cibles μ'_i avec poids θ'_i <- θ_i
    Initialiser les réseaux Cibles Q'_i avec poids φ'_i <- φ_i
    Initialiser le calendrier d'exploration Epsilon (ε)

Pour chaque épisode = 1 jusqu'à MAX_EPISODES:
    Réinitialiser l'environnement IoT (budgets d'énergie, charges des périphériques, modèles des agents)
    Obtenir l'observation initiale o_i pour chaque agent i
    Construire l'état global s = [o_1, o_2, ..., o_N]
    done = {i: Faux pour tout i}
    active_before = {i: Vrai pour tout i}

    Tant que non tous(done):
        # 1. Sélection décentralisée des actions (Exécution)
        Obtenir le masque des actions valides valid_actions_dict pour chaque agent
        Pour chaque agent i:
            Si done[i] est Vrai:
                a_i = action_nulle (ou valeur de remplissage)
            Sinon:
                Avec probabilité ε:
                    a_i = Choisir_Alea(valid_actions_dict[i])
                Sinon:
                    Calculer logits = μ_i(o_i)
                    Appliquer le masque sur logits ( logits[invalide] = -inf )
                    a_i = argmax(logits)
        
        Exécuter l'action conjointe a = [a_1, ..., a_N] dans l'environnement IoT
        Calculer pour chaque agent :
            - Temps de calcul (t_comp) et de communication (t_comm) avec DVFS
            - Consommation d'énergie (E_total) et mise à jour de la batterie restante du device choisi
            - Vérification des contraintes (Mémoire, Confidentialité/Trust)
        
        Obtenir les récompenses r_i, les observations suivantes o'_i, et les états done'[i]
        Construire le nouvel état global s' = [o'_1, o'_2, ..., o'_N]
        Définir active[i] = non done[i] (actif au début de l'étape)

        # Enregistrer la transition dans le Replay Buffer
        D.ajouter(s, s', o, o', a, r, done', active, masks_actuels, masks_suivants)

        Mettre à jour: s <- s', o_i <- o'_i, done <- done'

        # 2. Étape d'apprentissage Centralisé (si taille de D >= batch_size)
        Si taille(D) >= BATCH_SIZE:
            Échantillonner un lot aléatoire de transitions depuis D:
                (S, S', O, O', A, R, Done', Active, M, M')

            # 2.1 Calcul des actions futures cibles avec Gumbel-Softmax
            Pour chaque agent j:
                logits_suivants = μ'_j(O'_j)
                A'_j = Masked-Gumbel-Softmax(logits_suivants, M'_j, tau_gumbel, hard=Vrai)
            A_joint_cible = Concaténer(A'_1, A'_2, ..., A'_N)

            # 2.2 Mise à jour des Critiques Centralisés
            Pour chaque agent i:
                Si l'agent i n'est pas actif dans ce batch (somme(Active[:, i]) <= 0):
                    Continuer au prochain agent
                
                Calculer la cible temporelle Y_i pour chaque échantillon du batch :
                    Q_next = Q'_i(S', A_joint_cible)
                    Y_i = R_i + γ * (1 - Done'_i) * Q_next
                
                Obtenir la prédiction actuelle : Q_actuel = Q_i(S, A_one_hot)
                Calculer la perte du Critique (MSE pondérée par le masque Active) :
                    Perte_Critique = somme( (Q_actuel - Y_i)^2 * Active[:, i] ) / (somme(Active[:, i]) + 1e-6)
                
                Optimiser le Critique:
                    Minimiser Perte_Critique -> Mettre à jour les paramètres φ_i (Adam)

            # 2.3 Calcul des actions actuelles différentiables pour la mise à jour de l'acteur
            Pour chaque agent j:
                logits_actuels = μ_j(O_j)
                A_actuelle_diff_j = Masked-Gumbel-Softmax(logits_actuels, M_j, tau_gumbel, hard=Vrai)

            # 2.4 Mise à jour des Acteurs Décentralisés via le Gradient de Politique
            Pour chaque agent i:
                Si l'agent i n'est pas actif dans ce batch:
                    Continuer au prochain agent

                # Construire l'action conjointe où seule l'action de i est différentiable
                # (les autres actions des agents j != i sont détachées du graphe de calcul)
                A_joint_optim = [
                    A_actuelle_diff_j si j == i sinon Detach(A_actuelle_diff_j)
                    pour j dans 1..N
                ]
                A_joint_optim = Concaténer(A_joint_optim)
                
                Calculer la valeur Q projetée: Q_pi = Q_i(S, A_joint_optim)
                Calculer la perte de l'Acteur (Maximiser Q_pi) :
                    Perte_Acteur = somme( -Q_pi * Active[:, i] ) / (somme(Active[:, i]) + 1e-6)
                
                Optimiser l'Acteur:
                    Minimiser Perte_Acteur -> Mettre à jour les paramètres θ_i (Adam)

            # 2.5 Mise à jour douce des réseaux cibles (Soft Update)
            Pour chaque agent i:
                θ'_i <- (1 - τ) * θ'_i + τ * θ_i
                φ'_i <- (1 - τ) * φ'_i + τ * φ_i
                Décroître l'exploration ε de l'agent i (ε <- ε * ε_decay)
```

---

## 5. Formules Mathématiques de Mise à Jour

### Perte du Critique (Value Loss)
Le Critique centralisé $Q_i$ de l'agent $i$ est mis à jour en minimisant la perte d'erreur quadratique moyenne (MSE) par rapport à la valeur cible Bellman $y_i$ sur le batch échantillonné $\mathcal{B}$ :

$$\mathcal{L}(\phi_i) = \frac{1}{\sum_{b \in \mathcal{B}} \text{active}_i^{(b)}} \sum_{b \in \mathcal{B}} \text{active}_i^{(b)} \cdot \left[ Q_i\left(s^{(b)}, a_1^{(b)}, \dots, a_N^{(b)}\right) - y_i^{(b)} \right]^2$$

Où la valeur cible temporelle (TD-target) $y_i$ est calculée par :

$$y_i^{(b)} = r_i^{(b)} + \gamma \left(1 - d_i^{(b)}\right) Q'_i\left(s'^{(b)}, a'_1, \dots, a'_N\right)\Big|_{a'_j = \mu'_j\left(o'_j^{(b)}\right)}$$

*   $d_i^{(b)}$ est l'indicateur binaire indiquant si l'agent $i$ a terminé sa tâche.
*   $\text{active}_i^{(b)}$ indique si l'agent $i$ effectuait toujours des allocations de couches à ce pas de temps (évite de comptabiliser les pas de remplissage post-échec ou post-succès).

### Perte de l'Acteur (Policy Loss)
L'Acteur décentralisé $\mu_i$ est entraîné à maximiser l'estimation du Critique centralisé en ajustant ses poids $\theta_i$ à l'aide de la rétropropagation à travers les actions :

$$\mathcal{L}(\theta_i) = -\frac{1}{\sum_{b \in \mathcal{B}} \text{active}_i^{(b)}} \sum_{b \in \mathcal{B}} \text{active}_i^{(b)} \cdot Q_i\left(s^{(b)}, a^{(b)}_1, \dots, \tilde{a}_i^{(b)}, \dots, a^{(b)}_N\right)$$

Où $\tilde{a}_i^{(b)}$ est l'action différentiable échantillonnée par l'acteur de l'agent $i$ via le Gumbel-Softmax masqué :

$$\tilde{a}_i^{(b)} = \text{Masked-Gumbel-Softmax}\left(\mu_i\left(o_i^{(b)}; \theta_i\right), \text{mask}_i^{(b)}\right)$$

Les actions des autres agents $j \neq i$ sont figées (détachées du graphe de calcul de gradients, soit $a^{(b)}_j = \text{detach}(\dots)$) afin que l'optimisation ne mette à jour que la politique de l'agent $i$.

### Mise à jour Douce (Soft Update)
Pour garantir une convergence stable et éviter les divergences brutales de l'apprentissage par renforcement, les poids des réseaux cibles ($\theta'_i$ et $\phi'_i$) suivent lentement les réseaux principaux ($\theta_i$ et $\phi_i$) avec un facteur de lissage $\tau \ll 1$ :

$$\theta'_i \leftarrow (1-\tau)\theta'_i + \tau \theta_i$$
$$\phi'_i \leftarrow (1-\tau)\phi'_i + \tau \phi_i$$
Dans ce projet, $\tau = 0.01$ par défaut.

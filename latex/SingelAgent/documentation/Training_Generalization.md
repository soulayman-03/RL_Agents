# Explication de l'Apprentissage et de la Généralisation

Pourquoi les agents atteignent-ils 100% de succès sur de nouvelles configurations ?

## 1. Perception par Caractéristiques (Features)
Dans `MultiAgentIoTEnv`, l'agent ne voit pas seulement des index d'appareils, mais leurs caractéristiques réelles :
- **CPU Speed** (Puissance brute)
- **Memory Free** (Capacité mémoire)
- **Bandwidth** (Vitesse réseau)
- **Privacy Clearance** (Sécurité)

**Résultat** : Le réseau apprend une corrélation logique : *"Si CPU est haut et Mémoire suffisante -> Récompense élevée"*.

## 2. La Rotation des Seeds (L'Examen Final)
C'est le mécanisme qui empêche la mémorisation brute.
- Si l'agent apprenait par cœur que *"Device 3 est toujours le meilleur"*, il échouerait dès que l'environnement change.
- En changeant le **Seed**, le Device 3 peut devenir lent et le Device 0 rapide. L'agent est alors **forcé** d'analyser les caractéristiques (CPU/BW) pour s'adapter.

## 3. Preuve de Généralisation
Le succès sur 5 seeds différents (42, 55, 66, 77, 88) prouve que l'agent a acquis une **intelligence de décision**.
- Il scanne le réseau de manière dynamique.
- Il compare les spécifications aux besoins de sa couche.
- Il applique la règle apprise : *"Si Ressources > Besoins -> Allocation validée"*.

Cette capacité est essentielle pour le déploiement dans des réseaux IoT réels et changeants.

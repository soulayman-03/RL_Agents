# Analyse du Paradigme CTDE ( Apprentissage Centralisé, Exécution Décentralisée)

## État Actuel du Projet
Bien que le cadre proposé suive les principes généraux de l'apprentissage centralisé avec exécution décentralisée (**CTDE**), ce paradigme n'est pas pleinement exploité dans notre implémentation actuelle.

### Observations Globales
Dans notre configuration, tous les agents ont accès à la même **observation globale** de l'environnement. Cela inclut :
- L'ensemble des caractéristiques des dispositifs (CPU, Mémoire, Bande passante).
- Les contraintes du système (Confidentialité, Dépendances).

### Conclusion Technique
Par conséquent, les acteurs et le critique travaillent sur des informations identiques. Aucune décentralisation stricte des observations n'est imposée pour le moment.
> [!NOTE]
> Le processus d'apprentissage se rapproche davantage d'une formulation **acteur-critique centralisée** que d'un système multi-agents (MAS) entièrement décentralisé.

## Perspectives d'Évolution
L'adoption d'une structure compatible CTDE offre une base flexible pour de futures extensions :
1. **Observations Partielles** : Les agents pourraient être limités à ne voir que leur voisinage immédiat.
2. **Véritable Exécution Décentralisée** : Permettre une prise de décision autonome sans partage d'état global en temps réel.
3. **VDN/QMIX** : Intégration de réseaux de mixage pour mieux gérer le crédit global de l'équipe.

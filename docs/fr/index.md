# VisualMetrics

**Voyez la théorie. Changez les hypothèses. Comprenez le modèle.**

Un laboratoire visuel interactif pour la statistique, l'inférence statistique,
l'économétrie, l'inférence causale, l'apprentissage automatique et l'IA
moderne - en anglais, arabe et français.

- **47 laboratoires** répartis sur 15 domaines scientifiques
- **13 démonstrations**, chacune indiquant ce
  qu'elle n'établit pas
- **3 langues**, avec un vrai support de l'écriture de droite à gauche
- **189 objectifs pédagogiques** couvrant chaque laboratoire

## Ce qui le distingue

La plupart des outils pédagogiques vous montrent une image. Celui-ci vous dit
aussi de quelle nature elle est.

Chaque figure et chaque animation déclare son **type de preuve**. Une
démonstration géométrique et une étude de Monte-Carlo ne sont pas des
affirmations de même nature : elles ne portent donc jamais le même badge. Une
simulation est étiquetée comme telle, jamais comme une démonstration, et cette
règle est imposée par le code et par les tests.

Chaque animation porte une **couche d'explication** : ce que vous voyez, ce qui
a changé, pourquoi, comment le lire et ce qu'il faut en conclure - à côté de la
figure, et non dans une légende en dessous.

Chaque démonstration indique **ce qu'elle n'établit pas**, faute de quoi elle ne
se construit pas.

Le catalogue **ne se surestime jamais** : 16 concepts y figurent
sans être réalisés ; ils sont affichés comme prévus et refusent de s'ouvrir en
expliquant pourquoi.

## Commencer

- [Installation](getting-started/installation.md)
- [L'interface graphique](getting-started/gui.md)
- [L'API Python](getting-started/api.md)
- [Types de preuve](principles/evidence.md)

## Un premier aperçu

```python
import visualmetrics as vm

vm.configure(language="fr")
result = vm.lab("inference.power", effect_size=0.4, n=60)

result.evidence            # le type de preuve, déclaré et non sous-entendu
result.metric_dict()       # les chiffres
print(result.code)         # le code Python qui reproduit cette exécution
```

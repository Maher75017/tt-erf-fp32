# tt-erf-fp32 (en français)

La fonction `erf` en FP32 pour l'unité de calcul vectoriel (SFPU) des puces Tenstorrent Blackhole et Wormhole : moins d'instructions et une erreur maximale plus faible que le noyau actuel de tt-metal, vérifiée sur **toutes** les entrées FP32 positives avec un modèle bit à bit de l'arithmétique du matériel.

Contexte : l'issue tt-metal [#58282](https://github.com/tenstorrent/tt-metal/issues/58282) (précision de `ttnn.erf` en FP32) et l'issue [#61160](https://github.com/tenstorrent/tt-metal/issues/61160) (un éventuel niveau « PRECISE »).
Auteur : Maher El Hmissi ([Maher75017](https://github.com/Maher75017)), novembre 2026. English: [README.md](README.md).

## Résultats

Évaluation exhaustive des 2 139 095 041 entrées FP32 positives (de 0 à +inf ; `erf` est impaire), identique sur les modèles Blackhole et Wormhole.

| Noyau | Instructions SFPU par élément (estimation) | Erreur maximale | Entrées au-dessus de 1 ULP | Erreur moyenne |
| --- | --- | --- | --- | --- |
| Actuel de tt-metal (rationnelle n16/d16 sur [-10, 10], sortie bornée) | 64 | 8,000 ULP | 8,97 % | 0,341 ULP |
| **Proposé par défaut** (`R5/6+corr` : x·N(x²)/D(x²) plus une étape de correction) | **54** | **6,011 ULP** | 8,75 % | 0,335 ULP |
| Variante précise (`2M c=1.0 A6/B14` : deux polynômes, coupure à \|x\| = 1) | 80 | **1,000 ULP** | **0 %** | 0,374 ULP |

Les autres variantes (2,2 ULP pour 73 instructions, 1,6 ULP pour 77, etc.) sont dans `results/erf_tt_noyau.txt`, avec tous leurs coefficients en hexadécimal.

## Les limites, dites d'emblée

- **Le nombre d'instructions est une estimation** (instructions SFPU par élément, chargements de constantes compris), pas des cycles mesurés. Les cycles par tuile seront mesurés sur ttsim pour la contribution à tt-metal.
- Le gain de la version proposée porte surtout sur le pire cas ; son principal intérêt est d'avoir **environ 15 % d'instructions en moins**.
- La variante précise coûte environ 25 % d'instructions en plus. Vu la contrainte de performance exprimée dans #58282, elle n'est pas proposée par défaut.
- Mon propre critère de réussite, fixé avant la recherche (dans `erf/erf_tt.py`), était plus exigeant : au plus 1 ULP sans surcoût, ou au plus 2 ULP sans surcoût, ou au plus 1 ULP pour au plus 1,2 fois le coût. **Il n'a pas été atteint.** Les résultats ci-dessus sont ceux que la recherche a trouvés, présentés tels quels.

## Comment c'est vérifié

1. **Modèle bit à bit de l'arithmétique.** `model/sfpu_model.h` reprend le modèle de référence de Tenstorrent pour SFPMAD (Blackhole et Wormhole, `model/fma_ref.c` tiré de tt-isa-documentation, non modifié) et la description de SFPARECIP. `model/verif_modele.c` compare ce portage à la référence sur 20 millions de triplets d'opérandes tirés au hasard, dont la moitié avec des exposants proches pour provoquer des annulations : **0 écart** sur Blackhole comme sur Wormhole. Contrôle : comparer le portage Wormhole à la référence Blackhole donne 118 031 écarts, ce qui prouve que le test détecte bien les différences.
2. **Mesure de l'erreur.** Erreur en ULP par rapport à `erf` calculée en float64 sur l'entrée exacte, la taille de l'ULP étant prise dans la binade de la valeur exacte. C'est un peu plus strict que la mesure utilisée dans #58282.
3. **Contrôle positif.** Le modèle du noyau actuel reproduit, tranche par tranche, les chiffres mesurés sur ttsim et publiés dans #58282 :

| Tranche de \|x\| | Erreur moyenne, ce modèle | Erreur moyenne, #58282 (ttsim) | Au-dessus de 1 ULP, ce modèle | Au-dessus de 1 ULP, #58282 |
| --- | --- | --- | --- | --- |
| [0,5 ; 1,0] | 0,653 | 0,647 | 22,2 % | 22,5 % |
| [1,0 ; 1,5] | 0,946 | 0,943 | 40,7 % | 40,8 % |
| [1,5 ; 2,0] | 0,972 | 0,973 | 41,6 % | 41,2 % |
| [2,0 ; 2,5] | 1,065 | 1,073 | 45,7 % | 45,8 % |
| [2,5 ; 3,0] | 1,135 | 1,122 | 48,5 % | 49,0 % |
| [3,0 ; 4,0] | 1,146 | 1,127 | 47,7 % | 46,8 % |

4. **Contrôle négatif.** Décaler un seul coefficient de la variante précise de 256 ULP fait monter son erreur maximale à 64,9 ULP.
5. **Saturation.** `erf(x)` s'arrondit à 1,0f dès x = 3,9192059 ; les entrées sont bornées à 3,921875, valeur exacte en bf16.

## Preuve de date

L'empreinte SHA-256 de `results/erf_tt_noyau.txt` est
`7f33cf4e3fcfecf4593419aa21735197aa21e6b0edf4b0c9a7ed905188fbac01`.
La même empreinte a été envoyée à Tenstorrent le 19 novembre 2026, avec la proposition pour #58282. Le fichier est publié sans aucune modification.

## Reproduire

Les commandes sont dans [README.md](README.md) : environ une seconde pour la vérification du modèle, et 47 minutes sur une RTX 5090 pour la chaîne complète.

## Prochaine étape

Une contribution (« pull request ») à tt-metal, qui fera référence à #58282 : la modification du noyau pour Blackhole et Wormhole, des tests, les entrées du budget de précision, et des cycles par tuile mesurés sur ttsim plutôt qu'estimés.

Voir aussi [exact-exp32](https://github.com/Maher75017/exact-exp32) : des exponentielles correctement arrondies, vérifiées sur les 2^32 entrées, avec la même méthode.

## Licence

Licence Apache 2.0 (voir `LICENSE`), comme tt-metal et tt-isa-documentation. `model/fma_ref.c` est © Tenstorrent AI ULC et repris sans modification ; `model/sfpu_model.h` en est un portage et en conserve la mention.

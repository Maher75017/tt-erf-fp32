# tt-erf-fp32 (en français)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24485626-blue.svg)](https://doi.org/10.5281/zenodo.24485626)

La fonction `erf` en FP32 pour l'unité de calcul vectoriel (SFPU) des puces Tenstorrent Blackhole et Wormhole : un noyau à **1 ULP au plus sur toutes les entrées**, et une variante plus rapide dont l'erreur maximale est plus faible que celle du noyau actuel de tt-metal. Les deux sont vérifiés sur **toutes** les entrées FP32 positives avec un modèle bit à bit de l'arithmétique du matériel.

Contexte : l'issue tt-metal [#58282](https://github.com/tenstorrent/tt-metal/issues/58282) (précision de `ttnn.erf` en FP32) et l'issue [#61160](https://github.com/tenstorrent/tt-metal/issues/61160) (un éventuel niveau « PRECISE »).
Auteur : Maher El Hmissi ([Maher75017](https://github.com/Maher75017)), novembre 2026. English: [README.md](README.md).

## Résultats

Évaluation exhaustive des 2 139 095 041 entrées FP32 positives (de 0 à +inf ; `erf` est impaire), sur des modèles bit à bit des deux puces. Le calcul principal exécute ces modèles en CUDA sur une carte NVIDIA RTX 5090 ; le contrôle du calcul de 1/x sur Wormhole (plus bas) tourne sur processeur.

| Noyau | Instructions SFPU par élément (estimation) | Erreur maximale, Blackhole | Erreur maximale, Wormhole | Entrées au-dessus de 1 ULP (BH / WH) | Erreur moyenne |
| --- | --- | --- | --- | --- | --- |
| Actuel de tt-metal (rationnelle n16/d16 sur [-10, 10], sortie bornée) | 64 | 8,000 ULP | 7,000 ULP | 8,97 % / 9,05 % | 0,341 ULP |
| Variante rapide (`R5/6+corr` : x·N(x²)/D(x²) plus une étape de correction) | 54 | **6,011 ULP** | **6,011 ULP** | 8,75 % / 8,75 % | 0,335 ULP |
| **Variante précise, proposée à tt-metal** (`2M c=1.0 A6/B14` : deux polynômes, coupure à \|x\| = 1) | 80 | **1,000 ULP** | **1,000 ULP** | **0 % / 0 %** | 0,374 ULP |

Les autres variantes (2,2 ULP pour 73 instructions, 1,6 ULP pour 77, etc.) sont dans `results/erf_tt_noyau.txt`, avec tous leurs coefficients en hexadécimal.

## Correction (22 novembre 2026)

La première version de cette page disait que les chiffres Wormhole étaient identiques à ceux du Blackhole. C'était faux pour les noyaux qui divisent. Le calcul du 19 novembre modélisait bien l'opération SFPMAD du Wormhole, mais calculait 1/x avec la méthode du Blackhole sur les deux puces. Or, dans tt-metal, `sfpu_reciprocal` sur Wormhole suit un autre algorithme (graine quadratique, deux pas de Newton, mise à l'échelle par une puissance de deux). Avec ce calcul modélisé instruction par instruction (`model/wh_recip.c`, résultats dans `results/wormhole_vraie_reciproque.txt`) :

- le noyau actuel sur Wormhole est à **7,000 ULP** au maximum, avec 9,05 % des entrées au-dessus de 1 ULP (et non 8,000 ULP et 8,97 %) ;
- la variante rapide ne change pas sur Wormhole (6,011 ULP, 8,75 %) : son étape de correction absorbe l'écart entre les deux calculs de 1/x ;
- la variante précise ne fait pas de division : ses chiffres étaient déjà justes pour les deux puces.

Sur un échantillon de 214 489 entrées, le pire cas du noyau actuel paraît être 4,5 ULP ; l'évaluation exhaustive trouve 8,0 ULP sur Blackhole (en x = 5,8636, où `erf` s'arrondit à 1,0f). Un échantillon ne trouve pas les pires cas.

## Les limites, dites d'emblée

- **Le nombre d'instructions du tableau est une estimation.** Comptes exacts, sur les noyaux compilés avec SFPI 7.96.0 (instructions SFPU par tour de boucle de 32 éléments) : actuel 68 sur Blackhole / 76 sur Wormhole ; variante rapide 58 / 69 ; variante précise 81 / 85. Ce sont des nombres d'instructions, pas des cycles.
- La variante rapide baisse le pire cas, mais environ 18 % de [0, 4] reste au-dessus de 1 ULP, comme avec le noyau actuel. La variante précise n'a plus aucune entrée au-dessus de 1 ULP, pour 19 % (Blackhole) / 12 % (Wormhole) d'instructions en plus.
- La discussion de #58282 demandait moins d'erreur sans dégrader les performances. La variante rapide respecte strictement cette condition ; la variante précise échange des instructions contre de la précision. La contribution à tt-metal ([#66809](https://github.com/tenstorrent/tt-metal/pull/66809)) propose la variante précise, et la rapide en alternative. Mesurée avec le banc ULP de tt-llk sur ttsim (`INP_FLOAT32` défini) : pire cas 1 ULP sur les deux puces, en mode exact comme approché.
- Mon propre critère de réussite, fixé avant la recherche (dans `erf/erf_tt.py`), était plus exigeant : au plus 1 ULP sans surcoût, ou au plus 2 ULP sans surcoût, ou au plus 1 ULP pour au plus 1,2 fois le coût. **Il n'a pas été atteint.** Les résultats ci-dessus sont ceux que la recherche a trouvés, présentés tels quels.

## Comment c'est vérifié

1. **Modèle bit à bit de l'arithmétique.** `model/sfpu_model.h` reprend le modèle de référence de Tenstorrent pour SFPMAD (Blackhole et Wormhole, `model/fma_ref.c` tiré de tt-isa-documentation, non modifié) et la description de SFPARECIP. `model/verif_modele.c` compare ce portage à la référence sur 20 millions de triplets d'opérandes tirés au hasard, dont la moitié avec des exposants proches pour provoquer des annulations : **0 écart** sur Blackhole comme sur Wormhole. Contrôle : comparer le portage Wormhole à la référence Blackhole donne 118 031 écarts, ce qui prouve que le test détecte bien les différences.
2. **Calcul de 1/x sur Wormhole.** `model/wh_recip.c` suit `sfpu_reciprocal` de tt-metal pour Wormhole instruction par instruction (`ckernel_sfpu_recip.h`, commit a695405). Contrôle (`model/ctrl_recip.c`) : sur une entrée sur 7 entre 1 et 2^25, le 1/x modélisé est à moins de 0,892 ULP du vrai 1/x ; avec une seule constante faussée, l'écart monte à environ 10^8 ULP.
3. **Mesure de l'erreur.** Erreur en ULP par rapport à `erf` calculée en float64 sur l'entrée exacte, la taille de l'ULP étant prise dans la binade de la valeur exacte. C'est un peu plus strict que la mesure utilisée dans #58282.
4. **Contrôle positif.** Le modèle du noyau actuel reproduit, tranche par tranche, les chiffres mesurés sur ttsim et publiés dans #58282 :

| Tranche de \|x\| | Erreur moyenne, ce modèle | Erreur moyenne, #58282 (ttsim) | Au-dessus de 1 ULP, ce modèle | Au-dessus de 1 ULP, #58282 |
| --- | --- | --- | --- | --- |
| [0,5 ; 1,0] | 0,653 | 0,647 | 22,2 % | 22,5 % |
| [1,0 ; 1,5] | 0,946 | 0,943 | 40,7 % | 40,8 % |
| [1,5 ; 2,0] | 0,972 | 0,973 | 41,6 % | 41,2 % |
| [2,0 ; 2,5] | 1,065 | 1,073 | 45,7 % | 45,8 % |
| [2,5 ; 3,0] | 1,135 | 1,122 | 48,5 % | 49,0 % |
| [3,0 ; 4,0] | 1,146 | 1,127 | 47,7 % | 46,8 % |

5. **Contrôle négatif.** Décaler un seul coefficient de la variante précise de 256 ULP fait monter son erreur maximale à 64,9 ULP.
6. **Saturation.** `erf(x)` s'arrondit à 1,0f dès x = 3,9192059 ; les entrées sont bornées à 3,921875, valeur exacte en bf16.

## Travaux antérieurs

- **CUDA (NVIDIA).** Norbert Juffa, qui a travaillé chez NVIDIA jusqu'en 2014, a publié sur le forum développeurs de NVIDIA une `erff` CUDA optimisée pour la précision ([2022, code mis à jour jusqu'en 2025](https://forums.developer.nvidia.com/t/accuracy-optimized-implementation-of-erff-without-performance-impact/218114) ; [une version antérieure pour CUDA 7.5](https://forums.developer.nvidia.com/t/optimized-version-of-single-precision-error-function-erff/40977)). Comme la variante précise d'ici, elle coupe le domaine à \|x\| = 1, avec erf(a) ≈ a + a·P(a²) en dessous : une forme classique. Au-dessus de 1, elle appelle l'exponentielle `exp2` matérielle du GPU ; la variante précise utilise à la place un seul polynôme en \|x\| − 2,46875, sans exponentielle ni division. L'erreur maximale annoncée est de 1,04 ULP, contre 1,14 ULP pour l'`erff` de CUDA 11. Les coefficients d'ici viennent de l'ajustement propre à ce dépôt (`erf/erf_tt.py`).
- **Processeur.** La bibliothèque LLVM libc a une `erff` correctement arrondie depuis 2023 (32 sous-intervalles, polynôme impair de degré 15, calcul en double précision). Le SFPU n'a pas de double précision : cette recette ne s'y transpose pas directement.
- **Ce que ce dépôt apporte**, ce n'est pas la forme de l'approximation, mais son adaptation au SFPU : des modèles bit à bit de l'arithmétique Blackhole et Wormhole, et une vérification de toutes les entrées FP32 sur les deux puces.

## Preuve de date

L'empreinte SHA-256 de `results/erf_tt_noyau.txt` est
`7f33cf4e3fcfecf4593419aa21735197aa21e6b0edf4b0c9a7ed905188fbac01`.
La même empreinte a été envoyée à Tenstorrent le 19 novembre 2026, avec la proposition pour #58282. Le fichier est publié sans aucune modification.

## Reproduire

Les commandes sont dans [README.md](README.md) : environ une seconde pour la vérification du modèle, et 47 minutes sur une RTX 5090 pour la chaîne complète. Le contrôle Wormhole (étape 3) prend environ 25 minutes sur 2 cœurs.

## Prochaine étape

Contribution [tenstorrent/tt-metal#66809](https://github.com/tenstorrent/tt-metal/pull/66809) : la variante précise pour Blackhole et Wormhole (#58282), avec la variante rapide en alternative. À noter : les tests Float32 de tt-llk compilent aujourd'hui la version bf16 d'`erf` ([#56492](https://github.com/tenstorrent/tt-metal/issues/56492)) ; les mesures de la contribution définiront donc `INP_FLOAT32`.

Voir aussi [exact-exp32](https://github.com/Maher75017/exact-exp32) : des exponentielles correctement arrondies, vérifiées sur les 2^32 entrées, avec la même méthode.

## Licence

Licence Apache 2.0 (voir `LICENSE`), comme tt-metal et tt-isa-documentation. `model/fma_ref.c` est © Tenstorrent AI ULC et repris sans modification ; `model/sfpu_model.h` en est un portage et en conserve la mention.

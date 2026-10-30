# Good and bad Turkish FLEURS TEST examples

These are selected examples from checkpoint **400,000** in the original source runtime. Full Turkish TEST WER is **18.21%** (2,341/12,856 words, 743 utterances). **103/743** utterances have zero normalized word errors; **43/743** have utterance WER of at least 50%. Selection illustrates behavior and is not a representative random sample. No new inference was run for this selection.

The scorer ignores punctuation/case using Turkish-aware lowercase, but does not equate spelled-out numbers with digits. Scores below are individual utterance WER. Audio is available from the original FLEURS dataset; no audio is redistributed in this code repository.

## Good: ordinary explanatory speech

TEST row `64`, sentence ID `1867`; **0.00% WER** (0/13).

**Reference:** Bitkiler insanların soluduğu oksijeni üretir ve insanların çıkardığı (yani soluk verdiği) karbondioksiti alır.

**Model:** Bitkiler insanların soluduğu oksijeni üretir ve insanların çıkardığı yani soluk verdiği karbondioksiti alır.

## Good: scientific explanation

TEST row `91`, sentence ID `1766`; **0.00% WER** (0/15).

**Reference:** Bazı atomların kararsız çekirdekleri vardır, yani küçük bir müdahaleyle ya da hiç müdahale olmadan parçalanabilirler.

**Model:** Bazı atomların kararsız çekirdekleri vardır. Yani küçük bir müdahaleyle ya da hiç müdahale olmadan parçalanabilirler.

## Bad: foreign proper names

TEST row `1`, sentence ID `1660`; **35.71% WER** (5/14).

**Reference:** Romantizm; Goethe, Fichte ve Schlegel gibi yazarlardan geçen geniş bir kültürel determinizm özelliği taşıyordu.

**Model:** Romantizm, göte, pihte, ve şile gel gibi yazarlardan geten geniş bir kültürel determinizm özelliği taşıyordu.

## Bad: technical vocabulary

TEST row `2`, sentence ID `1692`; **45.45% WER** (5/11).

**Reference:** Japonya'nın nükleer ajansına göre, tesiste radyoaktif sezyum ve iyodin tespit edildi.

**Model:** Japonya'nın nükleer ajansına göre tesis de radioaktiv sezum ve iodin tespit edildi.

## Bad: dense list of sports

TEST row `212`, sentence ID `1832`; **73.68% WER** (14/19).

**Reference:** Futbol, ​​basketbol, ​​voleybol, sutopu, eskrim, ragbi, bisiklete binme, buz hokeyi, patenli hokey ve F1 motorlu araç yarışları popüler sporlardandır.

**Model:** Futbol, basketbol, baleybol su topu, eks grim rug bir bisikletebime, buz okey, batanli okey ve efir motorlu ara çalışları popüler sporlardandır

## Bad: number formatting plus recognition errors

TEST row `260`, sentence ID `1883`; **82.35% WER** (14/17).

**Reference:** Köprünün altındaki dikey açıklık 15 metredir. İnşaat 2011 yılı Ağustos ayında tamamlandı, Mart 2017'ye kadar trafiğe açılmadı.

**Model:** Köprünün altındaki dike açıklık on beş metridir. İnşaat iki bin on bir yılı Avuslu sayında tamamlandı, Mart iki bin on yediye kadar trafiğe açılmadı.

In the number example, the score includes digit-versus-word mismatches as well as actual word mistakes; it should not be interpreted as the same percentage of meaning lost.

Text/data attribution: Google and FLEURS authors/contributors, [google/fleurs](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), CC-BY-4.0. The selected source text is reproduced with attribution; hypotheses are model outputs.

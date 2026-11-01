# A terv, az eredmények előtt rögzítve

A szerző kísérleti naplójának két bejegyzése, szó szerint. A 23:35-ös időbélyeget kézzel írtam: a kinyerés kb. 23:33-kor indult, így a bejegyzés korábban készült, mint amit mond. Az eredmények először 23:47-kor készültek el. Angolul: [PLAN.md](PLAN.md).

### 2026-11-01 23:35 — Update 1 és Update 3 a teljes korpuszon: elemzési terv (az eredmény előtt)

- **A felhasználó kérése:** a J: korpusz pontozása végzett, most az Update 1 és az Update 3 összehasonlítása az 5,01 millió frame-en.
- **Anyag:** 5 010 602 frame, mindkét modellnek megvan minden pontszáma és cellatérképe a `master.db`-ben (csak olvasom). Update 1 = a v1.0 Base
  (`9840160f…`), Update 3 = a Small (`6cd68d83…`). Az Update 1 Small-nak (v1.1) nincs korpuszpontszáma.
- **Korlátok, előre:**
  - Az architektúra és a recept egybeesik (Base + recept 1 ~ Small + recept 3), így a különbség nem bontható szét.
  - Frame-típus nincs: a forrásvideók (`I:\delete`) most nem érhetők el, a frame-eknél nincs tárolt típus. A kodekfüggés korpuszon nem mérhető.
  - Mindkét modell ugyanabból az indexből tanult (`round3_index.csv`), így a csoportok: a 6 kimaradt film, a betanított frame-ek, a többi.
  - Egyik modell sem etalon, a mozgásmérés mozgást mér, nem blurt, a Qwen-kód csak durva ellenőrzés.
- **Először a különbségek, nem a győztes:**
  1. *Küszöb:* éles arány a saját r90 / r95 / r98-nál, csoportonként és filmenként.
  2. *Magabiztosság:* a logitok eloszlása, a saját r98 és r90 közé eső frame-ek aránya.
  3. *Egyezés:* Spearman globálisan, filmen és snitten belül (legalább 8 frame); kétszer kettes tábla és κ a saját küszöbökön, és egyenlő éles aránynál
     filmenként (az Update 3 küszöbe filmenként úgy áll, hogy annyit tartson meg, mint az Update 1).
  4. *Erős ellentmondás:* az egyik a saját r98-a alatt, a másik a saját r90-e fölött.
  5. *Térkép:* cellánként, minden frame-en: a két térkép frame-en belüli Spearmanja, piros arány (logit ≥ 0), a piros cellák átfedése (IoU), M = 0 aránya, a
     legnagyobb piros folt. A saját `r`-ek (Update 1 kb. 0,86, Update 3 kb. 0,70) felcserélve: a különbség a térképből vagy a kiolvasásból jön-e.
- **Tanúk, egyenlő éles aránynál:**
  - a. a megtartott frame-ek mért mozgása, a logit és a mozgás snitten belüli Spearmanja;
  - b. a csak az egyik által megtartott frame-ek mozgásrangja a snitten belül;
  - c. időbeli stabilitás: döntésváltás a snitt szomszédos frame-jei közt, lag-1 autokorreláció;
  - d. a két kiadású filmek minden párosítható frame-jén a logitok Spearmanja, az r95-egyezés és a κ;
  - e. a cellaszintű mozgástanú (R4T-1 9.6) a tárolt térképekből, minden frame-en, ahol van mérés;
  - f. durva ellenőrzés: a Qwen-kódok a megtartottak közt, AUC a kimaradt filmeken.
- **Összesítés:** a különbségekre (Update 3 − Update 1) bootstrap a filmekre (2000, seed 20261030); a 6 kimaradt filmre a filmenkénti értékek és a snittekre vett bootstrap.
- **Olvasat, előre:** tanúnként jelentem az irányt és az intervallumot, egy összesített győztest nem hirdetek. Ha egy tanú csak a mozgásban vagy csak a
  betanított frame-eken mutat különbséget, azt így írom le.

### 2026-11-01 23:37 — hozzáadva a korpusz-összehasonlításhoz: a döntési sávok (az eredmény előtt)

- **A felhasználó javaslata:** a küszöbök közti sávok legyenek kategóriák: r98 alatt, r98–r95, r95–r90 („éles, kevésbé éles, legkevésbé éles”). A behozott
  extrém közelit és az elmosódott közelit a legkevésbé éles sávban kell keresni. Az `sss`, `ss`, `s` név ütközne a Qwen-kódokkal, ezért itt **A** (r98 alatt),
  **B** (r98–r95), **C** (r95–r90), **D** (r90-től).
- **Mérés mindkét modellre:** a sávok aránya csoportonként és filmenként; a sávok egymás közti táblája (Update 1 × Update 3); sávonként a nullás snittek aránya, a
  Qwen-kódok, a mért mozgás, a piros arány és a csúcs; a snittek legjobb sávja; a C-sáv frame-jei két részre bontva: a snittben van jobb sáv, vagy a C a snitt legjobbja.
- **Korlát:** nézetcímke nincs. A nullás snitt csak durva jelző a közelire, az extrém közelit és a közelit nem választja szét.

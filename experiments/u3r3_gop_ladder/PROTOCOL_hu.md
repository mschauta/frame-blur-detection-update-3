# U3-R3: folytonos, GOP-ban kódolt fotólétra a round3 receptjével (ConvNeXt-Small)

## Protokoll (rögzítve 2026-10-29, a futás indítása előtt)

**Kérdés** (felhasználó, 2026-10-29): mi történik, ha a round3 receptjében a fotók folytonos 0–30 px-es létrát
kapnak, és a képeket mindhárom kodekben, GOP-os klipként kódoljuk („gyakorlatilag az Update 1-et finomítanánk”)?
1. Csökken-e a kodekfüggés (R4K-2)?
2. Rangsorol-e a modell az éles sávban (0–3 px)?
3. Megmarad-e a frame-pontosság?
4. Milyen cellatérképet ad?

**Döntések** (felhasználó, 2026-10-29):
- Csak ez a futás (a javasolt 2×2-es elrendezés 3. cellája).
- Mindhárom kodek, hogy a FrameBlurStudióban minden videón tesztelhető legyen.
- Vízjelmaszk nincs, és a maszk szivárgásával sem kell törődni.
- A sorrendi kimenet úszó horgonnyal a javaslat szerint (lent).

**Futás:** `runs/update3_small_gop_ladder_v1`, konfiguráció `configs/update3_small_gop_ladder.json`, indító
`round3_inditas.bat configs\update3_small_gop_ladder.json`.

### A recept

**Ami a round3 Smalltól eltér:**
1. **Fotók:** folytonos, GOP-ban kódolt létra (lent), a két bináris fotóosztály (0–3 px éles, 16–30 px blur)
   helyett.
2. **Fej:** sorrendi (ordinális) kimenet úszó horgonnyal, a bináris fotócél helyett.
3. **Maszk:**
   - vízjelmaszk sem a frame-eken, sem a fotókon (`frames_mask_share` és `synth_mask_share` 0,5 → 0);
   - `mask_leak_fix` ki.
   A frame-ek így a vízjel képpontjaival együtt tanulnak, ahogy a v9 és ahogy a FrameBlurStudio pontoz.
4. **Adatbetöltő:** 16 folyamat 10 helyett, a drágább szintézis miatt.

**Ami ugyanaz:**
- az index (`data/round3_index.csv`), a frame-válogatás (v8), a frame-címkék: éles = sss (+ ssb az
  ssb-domináns filmekben), blur = bbb + bsb;
- a rétegek: frames/sharp 12 000, frames/blurred 12 000, photo_ladder/ladder 24 000 epochonként. A fotó : frame
  arány 1 : 1, mint a round3-ban;
- 30 epoch × 48 000 minta, batch 8 × 2;
- a tanulási ráta (1e-4 / 1e-3), a bemelegítés (0,25 epoch), a koszinuszos csökkenés 1%-ig, a seed (20261016);
- a fp32 fingerprint, a ConvNeXt-Small ImageNet-1K súlyokból, az LSE-pooling (r induláskor 4);
- a sűrű (cellaszintű) fotóveszteség súlya 0,5, az élküszöb 0,001.

**A fotóminta** (`synth.gop_recipe` és `make_gop`, determinisztikus a (fotó, epoch) párból):
- **Hossz:** 20% pontosan 0 px, 40% egyenletes (0; 8] px, 40% egyenletes (8; 30] px.
- **Irány:** tetszőleges (0–180°). Kameramozgás lineáris fényben (`motion_kernel_v2`).
- **Klip:** 8 frame, 30 fps. A kivágás oldalanként 16 px-et veszít. Pásztázás a blur irányában, 0–4 egész px /
  frame, egyenlő eséllyel (a 0 az álló klip). Az elmosott fotóból vágott ablakokból áll, újramintavételezés nincs.
- **Kodek:** egyharmad-egyharmad eséllyel:
  - x264 (high, preset medium, 8 bit);
  - x265 (preset fast, 10 bit);
  - AV1 (libaom, cpu-used 8, 10 bit).
  - Az AV1-minták 30%-a filmszemcse-szintézist kap (denoise-noise-level 5–25).
- **Szerkezet:** csak I 0,2, IP 0,3, IPB 0,5.
  - x264 és x265: keyint 8, 3 B-frame, zárt GOP, scenecut 0.
  - AV1: IPB = lag-in-frames 8 (rejtett alt-ref frame-ek), IP = lag 0.
- **CRF:** a kodek listájából. A listák PSNR szerint vannak összehangolva, kb. 35–41 dB-re (24 tanítófotón,
  2026-10-29):
  - x264: 18, 21, 23, 26, 29, 32;
  - x265: 20, 23, 26, 29, 32, 34;
  - AV1: 24, 32, 38, 44, 50, 55.
- **Lánc:** bt709, tv tartomány, 4:2:0.
- **A minta** a klip egyik frame-je, egyenletes eséllyel; minden frame-je ugyanazt az L-et hordozza. A cellacélhoz a
  szerkezetet az eredeti (elmosatlan) fotó ugyanazon ablakán nézzük.
- **Validáció:** a validációs fotók rögzített változata (epoch None).

**A fej:**
- Küszöbök: T = 0,5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 30 px.
- logit(L ≥ t_k) = s − b_k. A b monoton (kumulált softplus-hézagok plusz egy tanult eltolás), és egyik küszöb sincs
  0-hoz kötve.
- A frame-ek bináris VLM-címkéje BCE(s) képszinten, vagyis s = 0 a frame-ek saját határa. A cella- és képlogit így
  a frame-döntés logitja marad: a küszöbök, az exportálás és a FrameBlurStudio változatlanul működnek.
- A fotó minden küszöbre kap célt: képszinten, és cellánként minden szerkezetes cellán.
- Induláskor a 3 px-es küszöb van 0-n.
- Mért hossz: L̂ = Σ σ(s − b_k)(t_k − t_{k−1}).

**Kiválasztott epoch:** előre rögzítve az utolsó, a 29. (mint a round3 Base-nél és a Smallnál). A best.pt-t nem
használjuk.

**Előzetes ellenőrzések:**
- 40 egységteszt (`tests/test_gop.py` és a régiek);
- próbafutás kis indexszel, 2 epoch;
- terheléses teszt: 1 200 klip, 0 hiba;
- a próbafutásban egy ffmpeg-összeomlás (0xC0000005), amelyet az újrapróbálkozás kezelt.

### Kontroll

- round3 Small, epoch 29 (`round3_small_p99_ep029.pt`).
- Az R4K-2-ben a round3 Base és a v9 is; ott a publikált számok az összevetés alapjai.

### Kiértékelés (az epoch 29 súlyaival)

Exportálás: az Update 1 formátumában. A küszöbök (r90 / r95 / r98) a validációs frame-ekből jönnek, mint eddig. A
meta tartalmazza a sorrendi küszöböket is (px-becsléshez).

1. **Kodekfüggés, R4K-2 (rögzített teszt, változatlan klipek és elemzés).**
   - Az új modell a meglévő 115 200 frame-en.
   - Elsődleges mérték: a P/B és a csak I közti páros logitkülönbség, és a blurnak mondott arány különbsége r95-nél,
     rátánként és mozgásonként.
   - Összevetés a round3 Smallal: a kodekrés különbsége (új − Small), ugyanazokkal a fotó-újramintavételekkel.
2. **Kimaradt filmek (az R4S-1 eljárása).**
   - Mérték: AUC, éles megtartás r90 / r95 / r98-nál, a mozgásmérés Spearman-korrelációja.
   - Páros különbség a round3 Smallal (bootstrap a snittekre).
   - Az intervallumok a hat film snittjeire vonatkoznak.
3. **Fotólétra.**
   - (a) A publikált Update 1-létra (300 fotó, egyframe-es CRF 23): átfordulás, 0–3 px-es pár-sorrend, szigorú
     növekedés 0–1–2–3 px-en.
   - (b) Ugyanez GOP-ban, x264 / x265 / AV1, csak I és IPB szerkezettel: a mért px és a valódi L eltérése (MAE,
     Spearman) kodekenként és frame-típusonként.
4. **Cellatérképek.**
   - A felhasználó szemrevételezése ismert filmeken (pl. Young Royals S01E01).
   - A cellaszintű mozgásmérés-tanú a validációs frame-eken (az R4T-1 9.6 módszere).
5. **Young Royals kodekmérce.**
   - Ugyanaz a frame AVC és HEVC SDR változatban (18 epizód), illetve az AV1 SDR (12 epizód): a pontszámok egyezése,
     a döntésváltások, a II. fázis I-frame-dúsulása.
   - Az új modell és a round3 Small között. A FrameBlurStudióban vagy szkripttel.
6. **px a natív frame-en.** A Qwen-kódok szerinti sorrend (bbb > bsb > sss) és a mozgás. Az R4T-1 ott bukott
   (a bbb frame-ek 17–29%-a 1 px alatt).

### Értelmezés (előre)

- **Javasolt mérce, a döntés a felhasználóé:**
  - kodekfüggés: az R4K-2 kodekrése a round3 Smallhoz képest legalább a felére csökken, és ennek intervalluma nem
    tartalmazza a 0-t;
  - frame-pontosság: a held-out AUC romlásának intervalluma nem lépi át a −0,005-öt;
  - rangsor az éles sávban: a 0–3 px-es pár-sorrend nagyobb, mint a round3 Small 0,659-e.
- Egyetlen futás: ha javul, nem választható szét, hogy a kódolás vagy a folytonos létra okozta. Ehhez a 2×2-es
  elrendezés másik két futása kellene.
- A maszk elhagyása harmadik eltérés a round3 Smalltól. A FrameBlurStudio maszk nélkül pontoz, így az összevetés
  ugyanazon a bemeneten történik.

### Nincs benne

- A 2×2-es elrendezés többi futása; a finomhangolás a round3 Smallból.
- HDR, a fekete sávok és a sötét jelenetek külön mérése (`update3_terv.md` 6. szakasz).

## Napló


# 2D difuzija toplote z MPI

Repozitorij vsebuje kodo za računanje 2D difuzije toplote paralelizirano z MPI (2D domenska dekompozicija), ter:

- skripto za benchmark, ki ustvari CSV s časi izvajanja
- skripto za analizo, ki izpiše Markdown tabelo in shrani grafe
- ločeno demo skripto, ki ustvari GIF animacijo heat matrike

## Opis naloge

**Cilj:** Implementacija 2D domenske dekompozicije in izmenjava robov v dveh dimenzijah.

**Navodila:** Rešite difuzijsko enačbo na kvadratni mreži, kjer je nova temperatura točke `(i,j)` povprečje njenih štirih sosedov. Mrežo razdelite na manjše kvadrate (podmreže / *sub-grids*) med procese.

**MPI naloga:** Vsak proces mora v vsakem časovnem koraku izmenjati robne vrstice in stolpce s štirimi sosedi (zgoraj, spodaj, levo, desno).

![Shema 2D domenske dekompozicije in halo izmenjave](assets/grids.png)

## Struktura projekta

```
heat-diffusion-fdm-mpi/
  assets/
    grids.png               # shema dekompozicije + halo izmenjave
  requirements.txt
  results/
    raw_times.csv            # ustvari scripts/bench.sh
    summary.md               # ustvari scripts/analyze.py
    runtime_vs_p.png         # ustvari scripts/analyze.py (opcijsko)
    speedup_vs_p.png         # ustvari scripts/analyze.py (opcijsko)
    efficiency_vs_p.png      # ustvari scripts/analyze.py (opcijsko)
    karp_flatt_vs_p.png      # ustvari scripts/analyze.py (opcijsko)
    heat_*.gif               # ustvari scripts/demo_animation.py
  scripts/
    heat2d_mpi.py            # MPI reševalnik
    bench.sh                 # benchmark -> results/raw_times.csv
    analyze.py               # analiza + grafi
    demo_animation.py        # MPI demo -> GIF animacija
```

## Okolje

Okolje, v katerem so bile izvedene meritve:

- OS: Windows + WSL (Ubuntu)
- CPU: Intel Core i7-12700H (14 jeder / 20 niti)
- Število MPI procesov: `p = 1, 2, 4, 8`
- Ponovitve: 3 zagoni za vsak `p`
- Mreža: `N = 512` (globalna mreža je `N x N`)
- Iteracije: `iters = 20000`
- Robni pogoji (Dirichlet): `TOP=100, LEFT=0, BOTTOM=0, RIGHT=0`

Opomba: mogoče je poganjati tudi več procesov (npr. 16) z uporabo strojnih niti (SMT)

## Namestitev

### 1) Sistemske zahteve (WSL)

V WSL Ubuntu potrebujete Python 3 in MPI okolje (OpenMPI ali MPICH). Primer za OpenMPI:

```bash
sudo apt update
sudo apt install -y python3 python3-pip openmpi-bin libopenmpi-dev
```

### 2) Python paketi

Iz roota repozitorija:

```bash
python3 -m pip install -r requirements.txt
```

## Zagon

Iz WSL (root repozitorija):

```bash
mpirun -np 4 python3 scripts/heat2d_mpi.py --N 512 --iters 20000 --bc 100 0 0 0
```

Izpiše se ena vrstica povzetka (iz rank 0), ki vsebuje skupni čas, število iteracij in dimenzije MPI mreže.

## Benchmark

Skripta za benchmark zapiše CSV s tremi zagoni za vsak `p`.

Iz WSL:

```bash
bash scripts/bench.sh --N 512 --iters 20000 --bc 100 0 0 0 --maxp 8 --csv results/raw_times.csv
```

Kaj naredi:

- Zažene `p = 1, 2, 4, 8` (potence 2 do `--maxp`)
- Vsak `p` zažene natanko 3-krat
- Zapiše/prepiše `results/raw_times.csv`

## Analiza (tabela + grafi)

### Ustvari Markdown povzetek

```bash
python3 scripts/analyze.py --in results/raw_times.csv --out results/summary.md
```

### Ustvari grafe skaliranja (PNG)

```bash
python3 scripts/analyze.py --in results/raw_times.csv --plot all --plot-dir results --plot-format png
```

Ustvari:

- `results/summary.md` (tabela z `Sₚ`, `Eₚ` in Karp-Flatt `e_p`)
- `results/runtime_vs_p.png` (čas na logaritemski osi)
- `results/speedup_vs_p.png` (vsebuje idealno referenčno premico `S=p`)
- `results/efficiency_vs_p.png`
- `results/karp_flatt_vs_p.png`

### Grafi

![Čas izvajanja glede na število procesov](results/runtime_vs_p.png)

![Pospešek glede na število procesov](results/speedup_vs_p.png)

![Učinkovitost glede na število procesov](results/efficiency_vs_p.png)

![Karp-Flattova metrika glede na število procesov](results/karp_flatt_vs_p.png)

## Interpretacija rezultatov

Izvedli smo meritve **strong scalinga** (fiksna velikost problema) za Jacobi 2D reševalnik difuzije toplote pri velikosti mreže `N=512` in `20000` iteracijah. Za vsako število procesov `p ∈ {1, 2, 4, 8}` smo meritve ponovili trikrat in uporabili povprečne vrednosti.

Osnovne metrike, ki jih uporabljamo, so:

- pospešek: `Sₚ = T₁ / Tₚ`
- učinkovitost: `Eₚ = Sₚ / p`
- Karp-Flattova metrika: `e_p = (1/Sₚ - 1/p) / (1 - 1/p)`

Graf pospeška vključuje tudi idealno referenčno premico `S = p`, ki ne predstavlja dejanskega modela, ampak služi kot orientacija za primerjavo z idealnim linearno skaliranim izvajanjem.

### Povzetek rezultatov

Iz `results/summary.md` dobimo naslednje povprečne čase:

- `T₁ ≈ 32.29 s`
- `T₂ ≈ 19.07 s`
- `T₄ ≈ 11.34 s`
- `T₈ ≈ 11.70 s`

Rezultati kažejo jasno izboljšanje do `p = 4`, kjer se čas izvajanja občutno zmanjša. Pri `p = 8` pa se trend ustavi — čas se ne izboljša več, ampak se celo rahlo poslabša v primerjavi z `p = 4`. To pomeni, da se pri tej velikosti problema začnejo dominatno pojavljati paralelni nadglavni stroški.

To lahko vidimo tudi iz pospeška in učinkovitosti:

- `p = 2`: `S₂ ≈ 1.69`, `E₂ ≈ 0.85`
- `p = 4`: `S₄ ≈ 2.85`, `E₄ ≈ 0.71`
- `p = 8`: `S₈ ≈ 2.76`, `E₈ ≈ 0.35`

### Karp-Flattova metrika

Karp-Flattova metrika nam pomaga oceniti delež ne-paralelizabilnega dela oziroma nadglavnih stroškov:

- `e₂ ≈ 0.18`
- `e₄ ≈ 0.14`
- `e₈ ≈ 0.27`

Opazen porast pri `p = 8` kaže, da se delež časa, ki ni skalabilen (komunikacija, sinhronizacija in drugi nadglavni stroški), bistveno poveča. To je skladno z opažanjem, da dodatni procesi ne prinesejo več izboljšav.

### Verjetna ozka grla

Pri Jacobi metodi z izmenjavo halo robov se pri večjem številu procesov (pri fiksnem `N`) pojavijo tipična ozka grla:

- **MPI komunikacija**: vsak korak zahteva izmenjavo robnih podatkov med sosednjimi procesi, kar postane relativno dražje, ko se velikost lokalnega dela manjša.
- **Sinhronizacija**: procesi morajo čakati na najpočasnejšega soseda, kar omejuje skupno hitrost.
- **Omejitve pomnilniške pasovne širine**: Jacobi je zelo odvisen od dostopa do pomnilnika, zato se hitro približa saturaciji.
- **Slabše razmerje računanje/komunikacija**: z več procesi se zmanjša količina računanja na proces, medtem ko komunikacija ostaja relativno konstantna.

Zato pri `p = 8` dodatna paralelizacija ne prinese več koristi — komunikacijski stroški in sinhronizacija preprosto izničijo dobiček računanja.

### Sklep

Rezultati kažejo, da se reševalnik dobro skalira do približno `p = 4`, kjer še vedno ohranja visoko učinkovitost in občuten pospešek. Pri `p = 8` pa postanejo dominantni komunikacijski in sinhronizacijski stroški, kar se odrazi v stagnaciji pospeška in izrazitem padcu učinkovitosti. Za velikost problema `N = 512` je zato optimalno območje skaliranja do približno štirih procesov.

## Demo animacija (GIF)

`scripts/demo_animation.py` je samostojen MPI demo, ki ustvari GIF v `results/`.

### Primer 1: zgornji rob topel (vrednost 100), ostali robovi hladni, začetno stanje vseh notranjih celic je 0

```bash
mpirun -np 2 python3 scripts/demo_animation.py \
  --N 64 \
  --iters 200 \
  --init zero \
  --bc 100 0 0 0 \
  --fps 20 \
  --frame-step 1
```

### Primer 2: naključno začetno stanje notranjih celic, vsi robovi 0

```bash
mpirun -np 2 python3 scripts/demo_animation.py \
  --N 64 \
  --iters 200 \
  --init random \
  --bc 0 0 0 0 \
  --fps 20 \
  --frame-step 2
```

## Zagon iz Windowsa

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Irina/Documents/2.2/VisokoZmogljivoRacunalnistvo/heat-diffusion-fdm-mpi && bash scripts/bench.sh --N 512 --iters 20000 --bc 100 0 0 0 --maxp 8 --csv results/raw_times.csv"
```

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/Irina/Documents/2.2/VisokoZmogljivoRacunalnistvo/heat-diffusion-fdm-mpi && python3 scripts/analyze.py --in results/raw_times.csv --out results/summary.md --plot all --plot-dir results --plot-format png"
```


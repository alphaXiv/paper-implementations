# elo rating

Calculates elo rating of olmOCR vs other tools.

## Data

The pairwise judgment data is stored in `ratings.csv` as win/loss counts:
```
MethodA,MethodB,A_wins,B_wins,A_rate(%),B_rate(%)
marker,mineru,53,26,67.1,32.9
mineru,pdelf,22,55,28.6,71.4
gotocr_format,marker,26,45,36.6,63.4
marker,pdelf,31,49,38.8,61.3
gotocr_format,pdelf,29,41,41.4,58.6
gotocr_format,mineru,38,37,50.7,49.3
```

*Note* `pdfelf` is olmOCR.

## Usage

To calculate elo ratings, run the following command:
```bash
python calculate_elo_ratings.py ratings.csv --num-bootstrap 5000 --num-elo-sims 100 --confidence-level 95 --seed 123
```

It should print something like:
```
Bootstrapped Elo Ratings (95% CI):
--------------------------------------------------
pdelf        1813.0 ± 84.9 [1605.9, 1930.0]
mineru       1545.2 ± 99.7 [1336.7, 1714.1]
marker       1429.1 ± 100.7 [1267.6, 1645.5]
gotocr_format 1212.7 ± 82.0 [1097.3, 1408.3]

Pairwise Significance Tests:
--------------------------------------------------
gotocr_format vs marker       Δ = -216.3 [-470.8,  135.0] p = 0.218
gotocr_format vs mineru       Δ = -332.5 [-567.5,   19.3] p = 0.051
gotocr_format vs pdelf        Δ = -600.3 [-826.1, -344.3] p = 0.000*
marker       vs mineru       Δ = -116.1 [-365.4,  246.5] p = 0.430
marker       vs pdelf        Δ = -383.9 [-610.6,  -10.9] p = 0.044*
mineru       vs pdelf        Δ = -267.8 [-517.3,  104.0] p = 0.135
```

which is also already saved in `results.txt`.

To generate boxplots of elo ratings, run the following command:
```bash
python draw_boxplots.py results.txt boxplots.png
```

which should save boxplots as `boxplots.png`.
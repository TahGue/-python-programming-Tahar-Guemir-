# Laboration 3 – Linjär klassificering

## Vad gör programmet?
Programmet läser in `unlabelled_data.csv`, klassificerar varje punkt som antingen klass 0 eller 1 baserat på om den ligger under eller ovanför linjen `y = x`, sparar resultatet till `labelled_data.csv`, och visar en graf med de klassificerade punkterna och beslutsgränsen.

## Hur kör man?
1. Se till att `unlabelled_data.csv` finns i samma mapp.
2. Kör: `python classifier.py`
3. Programmet skapar `labelled_data.csv` och visar en graf.

## Antaganden
- Linjen `y = x` används som beslutsgräns.
- Punkter på linjen räknas som klass 1.
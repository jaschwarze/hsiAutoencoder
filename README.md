# Segmented Autoencoder für die Bandselektion bei Hyperspektraldaten

## Ablauf
### 1. Fliebandfilterskript (Preprocessing)
### 2. Segmentierung in informative Hyperspektralregionen
### 3. Berechnung der Distance Density (DD) für jedes Segment
### 4. Für jedes Segment einen Autoencoder mit Sparsity Constraint trainieren
### 5. Pro Segment die Bänder mit dem höchsten Gewicht wählen

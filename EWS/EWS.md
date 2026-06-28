# Sistema d'Alerta Primerenca

En aquesta carpeta es troba un prototip del Sistema d'Alerta Primerenca desenvolupat
en el treball. És una aplicació Streamlit que permet identificar, amb antelació,
els estudiants amb risc de no superar l'assignatura, a partir dels models i les
dades de les carpetes `models/` i `resultats/` (no cal connectar amb Google Drive
ni cap servei extern).

## Com executar-la

1. Instal·la les dependències (només cal la primera vegada):

   ```
   pip install -r requirements.txt
   ```

2. Arrenca l'app des d'aquesta carpeta:

   ```
   streamlit run app.py
   ```

3. S'obrirà sola al navegador (normalment a `http://localhost:8501`). Si no, obre
   l'adreça que surti a la terminal.


## Estructura necessària (ja present)

```
EWS_TFG/
├── app.py
├── requirements.txt
├── models/       (model_*.txt + thresholds_optuna.json)
└── resultats/    (b1.csv, b2.csv, b3.csv, b4.csv, qualificacions.csv, ...)
```

Si mous `app.py` a una altra carpeta, mou-hi també `models/` i `resultats/`.

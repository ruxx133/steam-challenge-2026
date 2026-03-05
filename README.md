# Fire Risk Detection (ML + DST) – `fire_risk_dst_ml`

Acest repository conține un prototip de algoritm pentru **detecția timpurie a riscului de incendiu** pe baza senzorilor de mediu, gândit pentru integrare cu un sistem tip Arduino (alarmă + acționare).

## Descriere pe scurt

Fișierul **`fire_risk_dst_ml`** conține algoritmul de Machine Learning (ML) – **Regresie Logistică (Logistic Regression)** – care:

* calculează un **scor de risc de incendiu** (`risk`, între 0 și 1);
* generează un **output digital 0/1** (comandă), pe baza unui prag (`threshold`):

  * **1 = incendiu / activare**
  * **0 = fără incendiu / nu activează**
* primește ca **input** valorile:

  * **Temperature (°C)**
  * **Humidity (%)**
  * **IR / Radiation (0/1)**
    (unde 1 sugerează prezența flăcării/radiației IR, iar 0 absența ei)

În plus, scriptul folosește un strat de **fuziune a senzorilor prin Dempster–Shafer Theory (DST)** pentru a modela incertitudinea și a reduce alarmele false înainte de calculul final al riscului.



## Cum funcționează (pe scurt)

1. **Citire valori senzori:** temperatură, umiditate, IR (0/1)
2. **DST (fuziune + incertitudine):** combină evidența din senzori într-o estimare robustă (ex. `BetP(Fire)`)
3. **ML (Logistic Regression):** calculează `risk` = probabilitatea de incendiu
4. **Decizie finală:** `digital_output = 1` dacă `risk >= threshold`, altfel `0`
   (output-ul poate fi transmis către Arduino prin Serial)



## Cerințe

* Python 3.9+ recomandat
* Pachete:

  * `numpy`
  * `pandas`
  * `scikit-learn`

Instalare:

```bash
pip install numpy pandas scikit-learn
```

*(Opțional, dacă folosești modul de comunicare Serial cu Arduino: `pyserial`)*

```bash
pip install pyserial
```


## Cum rulezi

### Mod interactiv (introduci manual valorile)

```bash
python fire_risk_dst_ml.py
```

Apoi introduci:

* Temperature (°C)
* Humidity (%)
* IR (0/1)

Scriptul va afișa:

* `Risk (0..1): ...`
* `Digital output (0/1): ...`


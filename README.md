# DNN Stock Prediction

Progetto di deep learning per la previsione del prezzo di chiusura delle azioni Microsoft (MSFT) usando:

- MLP (Multi-Layer Perceptron)
- CNN 1D
- LSTM
- GRU

## File principali

- `DNN_stock_prediction.ipynb`: notebook con tutto il codice.
- `microsoft_stock.csv`: dataset storico dei prezzi Microsoft.

## Come eseguire

Il progetto è contenuto nel file `DNN_stock_prediction.ipynb` (notebook Jupyter).  
Puoi aprirlo ed eseguirlo su:

- Google Colab
- Jupyter Notebook / JupyterLab locale
- Altre piattaforme che supportano notebook `.ipynb`

Librerie principali richieste:

- `torch` (PyTorch)
- `numpy`
- `pandas`
- `matplotlib`

 **Su Google Colab**: queste librerie sono già disponibili, non serve installare nulla.
**In ambiente locale** (Jupyter sul proprio PC): assicurati di avere installate le librerie sopra.  
  Se necessario, puoi installarle con:

  ```bash
  pip install torch numpy pandas matplotlib
  ```
 
 
## Risultati principali

Il notebook:

- addestra quattro architetture di rete neurale sullo stesso dataset;
- valuta le prestazioni sul test set usando MSE, RMSE e MAE;
- confronta visivamente le predizioni con il prezzo reale, sia sull’intero test set sia sugli ultimi 60 giorni.

I risultati mostrano quale modello segue meglio l’andamento del prezzo e quale ha l’errore medio più basso.

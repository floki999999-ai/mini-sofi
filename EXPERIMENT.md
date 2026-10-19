# Protocollo: due reti comunicanti e controlli

## Ipotesi

Misurare se lo scambio di rappresentazioni migliora la generalizzazione su `mondo.csv`, rispetto alle stesse reti private dello scambio e a una rete monolitica con numero di parametri simile. Non si assume che comunicare migliori il risultato e non si attribuiscono coscienza o comprensione alle reti.

## Architetture implementate

1. **Communicating (2.242 parametri)**: A vede solo x1 e B solo x2. Ciascuna produce un messaggio continuo di 32 valori tramite uno strato sigmoid 1→32. A riceve [messaggio A, messaggio B]; B riceve [messaggio B, messaggio A]. Ogni ramo applica 64→16→1 sigmoid. La predizione congiunta è la media delle due probabilità. C'è un solo scambio simultaneo per forward, non un dialogo ricorrente. Il gradiente della MSE congiunta attraversa entrambi i messaggi: l'addestramento è centralizzato e cooperativo.
2. **Isolated (2.242 parametri allocati, 1.218 effettivamente attivi)**: struttura, inizializzazione e aggregazione identiche, ma ogni messaggio ricevuto dall'altro ramo è sostituito da zeri, durante train e valutazione. I 1.024 pesi associati alle connessioni bloccate hanno gradiente nullo. Ogni ramo ha accesso solo al proprio input; un aggregatore comune media le risposte. Il confronto misura l'effetto di abilitare lo scambio, che aumenta anche la capacità effettiva. Non è un controllo a capacità effettiva identica.
3. **Monolithic (2.273 parametri)**: 2→32→64→1, accesso a entrambi gli input. Ha 31 parametri in più del comunicante, circa 1,4%. È un controllo approssimato per dimensione, non per costo computazionale o capacità rappresentativa.
4. **Baseline originale (6.465 parametri)**: 2→64→64→32→1, eseguita separatamente. È un riferimento storico, non un controllo di pari dimensione.

Le prime due architetture partono da tensori identici per lo stesso seed. Nessun messaggio include etichette o esempi test. L'aggregatore non ha parametri addestrabili.

## Procedura predisposta

- Fissare seed split 42 per il confronto iniziale; i tre modelli dello stesso comando condividono gli indici.
- Stessa MSE, stesso learning rate 0,4, stesso budget full batch e soglia 0,5.
- Scegliere il checkpoint migliore esclusivamente mediante MSE validation; in caso di parità mantenere il primo.
- Confrontare primariamente MSE e balanced accuracy test; riportare anche accuracy, precision, recall, F1, matrici, epoca selezionata e parametri. Un classificatore sempre positivo ha accuracy test circa 57,56%: superare il 50% non basta.
- Prima di una campagna lunga, definire e approvare budget di epoche e semi. Per una campagna successiva si possono preregistrare 42,43,44,45,46 e riportare tutti i risultati, media e deviazione standard. Nell'interfaccia attuale cambiare seed cambia sia split sia inizializzazione: la dispersione combina entrambe le fonti. Un disegno futuro dovrebbe separarle.
- Non usare i risultati test delle prove brevi per scegliere iperparametri. Una campagna esplorativa ripetuta richiederebbe un ulteriore holdout intatto per una stima finale indipendente.

Comando già verificato, da lanciare con una cartella nuova:

```powershell
python mini_sofi.py --mode experiment --seed 42 --epochs 5 --out runs/nuova_prova
```

## Stato e limiti

Per i tre modelli del confronto sono state eseguite solo 5 epoche per modello; la baseline storica ha poi completato separatamente 30.000 epoche. Gli artefatti inclusi dimostrano la funzionalità della pipeline, non un vantaggio delle reti comunicanti. Il confronto comunicante/isolato/monolitico non ha ancora eseguito una campagna estesa o una ricerca di iperparametri. Il confronto monolitico non pareggia esattamente FLOP o capacità; il controllo isolato riduce i parametri attivi e l'accesso all'informazione. In futuro un controllo addizionale con messaggi permutati e uno isolato di capacità aumentata aiuterebbero a distinguere informazione utile e capacità. Questi controlli aggiuntivi non sono implementati nella versione 1.0.

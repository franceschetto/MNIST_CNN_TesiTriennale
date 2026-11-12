# MNIST_CNN_TesiTriennale

Git repository per Tesi su Reti Neurali ed In-Memory Computing (IMC).
Contiene **MNIST** dataset (`.\MNIST`), [**Cross-Sim**](https://github.com/sandialabs/cross-sim) simulator per IMC evaluation(`.\applications` e (`.\simulator`)), ed alcuni Python notebooks con esempi:
- `1_Example_MNIST_CNN_Train.ipynb` : Esempio PyTorch per definire ed allenare Convolutional Neural Network (CNN) per digit recognition su MNIST dataset; allena due reti: la CNN classica e una variante con Dropout, allenata con data augmentation
- `2_Example_MNIST_CNN_Load.ipynb` : Esempio PyTorch per leggere rete giá allenata e testarla (entrambe le reti); include la lavagna interattiva e il fine-tuning delle reti sul dataset personale (il confronto tra tutte le reti è nel notebook 3)
- `3_Example_MNIST_CNN_InMemoryComputing.ipynb` : Esempio Cross-Sim per valutare rete allenata in contesto IMC e, nel caso, riallenarla; confronta tutte le reti (digitali, su IMC e dopo l'Hardware-Aware Training) in accuratezza e robustezza (cifre deformate e disegni della lavagna)
- `4_Example_MNIST_Personal_Dataset.ipynb` : Raccolta di un dataset personale di cifre disegnate a mano con la lavagna; i disegni sono solo rimpiccioliti a 28x28, senza adattarli a MNIST, e salvati con la stessa struttura dei file MNIST (tensori di immagini 28x28 e di etichette)

Altri file aggiunti:
- `lavagna.py` : funzioni riutilizzabili da qualsiasi notebook: `apri_lavagna` (lavagna interattiva per classificare cifre disegnate e confrontare più reti) e `raccogli_cifre` / `salva_dataset` (raccolta del dataset personale e dei disegni per la verifica della robustezza). In tutte le lavagne il disegno viene solo rimpicciolito a 28x28, senza adattarlo a MNIST; la funzione di adattamento (`preprocess_drawing`) resta nel file, non usata, per poterla rimettere
- `dataset_personale/` e `dataset_personale.pt` : dataset personale (immagini 28x28 e disegni originali; file unico con la stessa struttura dei file MNIST, ma con i disegni non adattati a MNIST)
- `trained_mnist_cnn.pt` e `trained_mnist_cnn_dropout.pt` : pesi delle due reti allenate nel notebook 1
- `trained_mnist_cnn_personale.pt` e `trained_mnist_cnn_dropout_personale.pt` : pesi delle due reti allenate anche sul dataset personale nel notebook 2
- `trained_mnist_cnn_*_hat.pt` (4 file) : pesi delle 4 reti dopo l'Hardware-Aware Training del notebook 3
- `robustezza_lavagna.pt` : disegni (4 per cifra) usati nel notebook 3 per verificare la robustezza delle reti
- `grafici/` : grafici del notebook 3 (scelta del weight clipping, pesi prima e dopo l'HAT, accuratezza per chip, robustezza alla rotazione, riepilogo); `train_val_loss_hat.png` : curve di loss dell'HAT

# MNIST_CNN_TesiTriennale

Git repository per Tesi su Reti Neurali ed In-Memory Computing (IMC).
Contiene **MNIST** dataset (`.\MNIST`), [**Cross-Sim**](https://github.com/sandialabs/cross-sim) simulator per IMC evaluation(`.\applications` e (`.\simulator`)), ed alcuni Python notebooks con esempi:
- `1_Example_MNIST_CNN_Train.ipynb` : Esempio PyTorch per definire ed allenare Convolutional Neural Network (CNN) per digit recognition su MNIST dataset; allena due reti: la CNN classica e una variante con Dropout, allenata con data augmentation
- `2_Example_MNIST_CNN_Load.ipynb` : Esempio PyTorch per leggere rete giá allenata e testarla (entrambe le reti); include la lavagna interattiva e il fine-tuning delle reti sul dataset personale, con confronto finale sulla lavagna
- `3_Example_MNIST_CNN_InMemoryComputing.ipynb` : Esempio Cross-Sim per valutare rete allenata in contesto IMC e, nel caso, riallenarla
- `4_Example_MNIST_Personal_Dataset.ipynb` : Raccolta di un dataset personale di cifre disegnate a mano con la lavagna, salvato nello stesso formato di MNIST

Altri file aggiunti:
- `lavagna.py` : funzioni riutilizzabili da qualsiasi notebook: `apri_lavagna` (lavagna interattiva per classificare cifre disegnate e confrontare più reti) e `raccogli_cifre` / `salva_dataset` (raccolta del dataset personale)
- `dataset_personale/` e `dataset_personale.pt` : dataset personale (immagini 28x28 e disegni originali; file unico in formato MNIST)
- `trained_mnist_cnn.pt` e `trained_mnist_cnn_dropout.pt` : pesi delle due reti allenate nel notebook 1

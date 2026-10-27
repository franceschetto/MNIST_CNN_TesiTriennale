"""
Lavagna interattiva per classificare cifre disegnate a mano con una o più reti MNIST.

Uso da qualsiasi notebook che si trova nella stessa cartella di questo file:

    from lavagna import apri_lavagna
    apri_lavagna({'CNN classica': mnist_cnn_pt, 'CNN con dropout': mnist_cnn_dropout_pt})

Il dizionario associa a ogni nome (mostrato in legenda e nel risultato) una rete già allenata:
va bene sia il SequentialWrapper (si usa il suo attributo .net) sia direttamente un torch.nn.Module.
La finestra si apre in un pop-up; la cella del notebook resta in esecuzione finché non la si chiude.

Il file contiene anche raccogli_cifre(), la lavagna per creare un dataset personale di cifre:

    from lavagna import raccogli_cifre, salva_dataset
    images, labels, originali = raccogli_cifre(campioni_per_cifra=10)   # raccolta in memoria, parte sempre da zero
    salva_dataset(images, labels, originali, 'dataset_personale')        # scrive su disco sostituendo il dataset precedente

Nota: se si modifica questo file con il notebook già aperto, bisogna riavviare il kernel
(oppure usare importlib.reload) perché le modifiche vengano lette.
"""
import os
import shutil
import time
import tkinter as tk

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageTk

# Colori delle reti nel grafico, nell'ordine in cui sono passate (stessi colori in barre, legenda e risultato)
COLORI_RETI = ['#4472c4', '#ed7d31', '#70ad47', '#ffc000', '#7030a0', '#a5a5a5']

# --- FUNZIONI DI SUPPORTO PER GRAFICA ARROTONDATA ---
def create_rounded_rect(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """Disegna un rettangolo con bordi arrotondati tramite coordinate poligonali smussate."""
    points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1,
              x2, y1, x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius,
              x2, y2, x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2,
              x1, y2, x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
    
    # Rimuove il bordo nero di default di Tkinter
    if 'outline' not in kwargs:
        kwargs['outline'] = ""
        
    return canvas.create_polygon(points, **kwargs, smooth=True)

class RoundedButton(tk.Canvas):
    """Classe personalizzata per creare bottoni tondeggianti con effetto click 3D."""
    def __init__(self, parent, width, height, text, bg_color, fg_color, command):
        super().__init__(parent, width=width, height=height, bg='#e8e8e8', highlightthickness=0)
        self.command = command
        # Ombra del bottone (grigio scuro, spostata in basso). Raggio ottimizzato a 20.
        self.shadow = create_rounded_rect(self, 2, 4, width-2, height, radius=20, fill='#cccccc')
        # Corpo centrale del bottone
        self.rect = create_rounded_rect(self, 2, 2, width-2, height-2, radius=20, fill=bg_color)
        self.text = self.create_text(width//2, height//2, text=text, fill=fg_color, font=("Helvetica", 12, "bold"))
        
        # Colleghiamo gli eventi del mouse
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        # Quando clicchi, il corpo e il testo scendono di 2 pixel coprendo l'ombra
        self.move(self.rect, 0, 2)
        self.move(self.text, 0, 2)

    def on_release(self, event):
        # Quando rilasci, tornano su ed esegue il comando
        self.move(self.rect, 0, -2)
        self.move(self.text, 0, -2)
        self.command()


# --- PRE-ELABORAZIONE DEL DISEGNO (stile MNIST) ---
# Le cifre MNIST sono ritagliate, scalate in un riquadro 20x20 (proporzioni mantenute) e centrate
# nel 28x28 usando il baricentro. Senza questo passaggio un disegno a tutta lavagna, decentrato,
# viene riconosciuto peggio anche se la rete e' ottima.
def shift_image(img, dy, dx):
    out = np.zeros_like(img)
    out[max(0, dy):min(28, 28 + dy), max(0, dx):min(28, 28 + dx)] = \
        img[max(0, -dy):min(28, 28 - dy), max(0, -dx):min(28, 28 - dx)]
    return out

def preprocess_drawing(img_pil):
    arr = np.array(img_pil)
    ys, xs = np.nonzero(arr > 0)
    if len(xs) == 0:
        return None                                   # lavagna vuota
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # ritaglio sulla cifra
    h, w = arr.shape
    scale = 20.0 / max(h, w)                          # il lato piu' lungo diventa 20 pixel
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    img28 = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    img28[top:top + new_h, left:left + new_w] = np.array(digit, dtype=np.float32)
    # centratura con il baricentro (come in MNIST)
    total = img28.sum()
    cy = (img28.sum(axis=1) * np.arange(28)).sum() / total
    cx = (img28.sum(axis=0) * np.arange(28)).sum() / total
    img28 = shift_image(img28, int(round(13.5 - cy)), int(round(13.5 - cx)))
    return np.clip(img28, 0, 255)


# Probabilità Softmax (in %) che una rete assegna alle 10 cifre
def softmax_percent(net, img_tensor):
    with torch.no_grad():
        logits = net(img_tensor)
        return F.softmax(logits, dim=1).squeeze().numpy() * 100


def apri_lavagna(reti):
    """Apre la lavagna e classifica il disegno con tutte le reti del dizionario {nome: rete}."""
    if isinstance(reti, dict):
        reti = list(reti.items())
    if not 1 <= len(reti) <= len(COLORI_RETI):
        raise ValueError(f"Servono da 1 a {len(COLORI_RETI)} reti, ricevute {len(reti)}")

    # Per ogni rete: nome, modulo PyTorch (dal Wrapper o diretto) e colore
    nomi = [nome for nome, _ in reti]
    nets = [getattr(rete, 'net', rete) for _, rete in reti]
    colori = COLORI_RETI[:len(reti)]

    # Modalità inferenza: con il Dropout attivo ogni classificazione darebbe un risultato diverso.
    for net in nets:
        net.eval()

    # --- SETUP VARIABILI ---
    canvas_size = 400
    stato = {'last_x': None, 'last_y': None}

    img_pil = Image.new("L", (canvas_size, canvas_size), color=0)
    draw_mem = ImageDraw.Draw(img_pil)

    # --- CREAZIONE FINESTRA POP-UP ---
    root = tk.Tk()
    root.title("Classificatore MNIST In-Memory Computing")
    root.geometry("1400x720")
    root.configure(bg='#e8e8e8')
    root.resizable(False, False)

    # --- LAYOUT A COLONNE ---
    left_frame = tk.Frame(root, bg='#e8e8e8')
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    right_frame = tk.Frame(root, bg='#e8e8e8')
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=20, pady=20)

    # --- COLONNA SINISTRA: MESSAGGI E GRAFICO ---
    # Una sola riga di testo per i messaggi (lavagna vuota, pulita...): i risultati sono nel grafico.
    lbl_messaggio = tk.Label(left_frame, text="Disegna una cifra a destra!", font=("Helvetica", 14, "bold"),
                             bg='#e8e8e8', fg='#333333')
    lbl_messaggio.pack(pady=(0, 5))

    def mostra_messaggio(testo, colore='#333333'):
        lbl_messaggio.config(text=testo, fg=colore)

    fig = Figure(figsize=(8.6, 6.2))
    fig.patch.set_facecolor('#e8e8e8')
    ax = fig.add_subplot(111)

    # Legenda sotto il grafico, sempre visibile: un riquadro colorato per ogni rete, al massimo 2 per riga
    # così anche i nomi lunghi restano leggibili. Lo spazio in basso viene riservato con tight_layout(rect=...).
    n_col_legenda = 1 if len(reti) == 1 else 2
    righe_legenda = -(-len(reti) // n_col_legenda)
    margine_legenda = 0.04 + 0.05 * righe_legenda
    fig.legend(handles=[Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=n) for n, c in zip(nomi, colori)],
               loc='lower center', ncol=n_col_legenda, frameon=False, fontsize=11)

    def aggiorna_layout():
        fig.tight_layout(rect=(0, margine_legenda, 1, 1))

    # Stile comune degli assi (usato all'avvio, dopo "Pulisci" e dopo ogni classificazione)
    def setup_axes():
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.arange(10))
        ax.set_xlim(-0.6, 9.6)
        ax.set_ylim(0, 118)
        ax.set_ylabel('Probabilità (%)', color='black')
        ax.set_xlabel('Cifre (0-9)', color='black')
        ax.set_title('Distribuzione Softmax', fontweight='bold', color='black')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        ax.set_axisbelow(True)

    setup_axes()
    aggiorna_layout()

    canvas_matplot = FigureCanvasTkAgg(fig, master=left_frame)
    canvas_matplot.get_tk_widget().pack()

    # --- COLONNA DESTRA: LAVAGNA ARROTONDATA ---
    cv = tk.Canvas(right_frame, width=canvas_size, height=canvas_size, bg='#e8e8e8', cursor="cross", highlightthickness=0)
    cv.pack(pady=(0, 20))
    # Disegniamo il quadrato nero con gli angoli smussati senza bordo nero
    create_rounded_rect(cv, 0, 0, canvas_size, canvas_size, radius=35, fill='black')

    def paint(event):
        x, y = event.x, event.y
        if stato['last_x'] is not None and stato['last_y'] is not None:
            cv.create_line((stato['last_x'], stato['last_y'], x, y), width=24, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            draw_mem.line([stato['last_x'], stato['last_y'], x, y], fill=255, width=24, joint='curve')
        stato['last_x'], stato['last_y'] = x, y

    def reset(event):
        stato['last_x'], stato['last_y'] = None, None

    cv.bind("<B1-Motion>", paint)
    cv.bind("<ButtonRelease-1>", reset)

    # --- FUNZIONI DEI BOTTONI ---
    def clear():
        cv.delete("all")
        create_rounded_rect(cv, 0, 0, canvas_size, canvas_size, radius=35, fill='black')
        draw_mem.rectangle([0, 0, canvas_size, canvas_size], fill=0)

        ax.clear()
        setup_axes()
        aggiorna_layout()
        canvas_matplot.draw()
        mostra_messaggio("Lavagna pulita. Disegna!")
        mostra_input(None)

    def predict():
        img_small = preprocess_drawing(img_pil)
        mostra_input(img_small)

        if img_small is None:
            mostra_messaggio("Errore: Lavagna vuota!", "#ff4d4d")
            return

        img_tensor = torch.tensor(img_small, dtype=torch.float32) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        # Stesso disegno passato a tutte le reti
        probs = [softmax_percent(net, img_tensor) for net in nets]

        mostra_messaggio("Classificazione completata")

        ax.clear()
        setup_axes()

        # Barre affiancate: per ogni cifra una barra per rete, nell'ordine del dizionario
        classes = np.arange(10)
        width = 0.8 / len(nets)
        tutte_le_barre = []
        for k, (nome, prob, colore) in enumerate(zip(nomi, probs, colori)):
            offset = (k - (len(nets) - 1) / 2) * width
            bars = ax.bar(classes + offset, prob, width, color=colore, edgecolor='black', linewidth=0.5)
            tutte_le_barre.extend(bars)

        # Percentuale sopra ogni barra non trascurabile (senza decimali sopra il 10%).
        # Con più di 2 reti le barre sono strette: le etichette vengono scritte in verticale per non sovrapporsi.
        verticale = len(nets) > 2
        for bar in tutte_le_barre:
            height = bar.get_height()
            if height > 0.5:
                label = f'{height:.0f}%' if height >= 10 else f'{height:.1f}%'
                ax.text(bar.get_x() + bar.get_width()/2, height + 1, label, va='bottom', ha='center',
                        fontsize=8, rotation=90 if verticale else 0, color='black')

        aggiorna_layout()
        canvas_matplot.draw()

    # --- BOTTONI CUSTOM (Tondeggianti e Interattivi) ---
    btn_frame = tk.Frame(right_frame, bg='#e8e8e8')
    btn_frame.pack()

    btn_pred = RoundedButton(btn_frame, width=130, height=45, text="🎯 Classifica", bg_color='#5cb85c', fg_color='white', command=predict)
    btn_pred.pack(side=tk.LEFT, padx=10)

    btn_clr = RoundedButton(btn_frame, width=130, height=45, text="🗑️ Pulisci", bg_color='#f0ad4e', fg_color='white', command=clear)
    btn_clr.pack(side=tk.LEFT, padx=10)

    # --- ANTEPRIMA DELL'INPUT DELLE RETI ---
    # Mostra l'immagine 28x28 che le reti ricevono davvero (dopo ritaglio, ridimensionamento a 20x20 e
    # centratura con padding), ingrandita 3 volte senza sfumature: ogni quadratino è un pixel MNIST.
    preview_frame = tk.Frame(btn_frame, bg='#e8e8e8')
    preview_frame.pack(side=tk.LEFT, padx=10)
    lbl_preview = tk.Label(preview_frame, bg='black', bd=0)
    lbl_preview.pack()
    tk.Label(preview_frame, text="Input rete 28x28", font=("Helvetica", 8), bg='#e8e8e8', fg='#555555').pack()

    def mostra_input(img28):
        if img28 is None:
            img28 = np.zeros((28, 28))
        foto = ImageTk.PhotoImage(Image.fromarray(img28.astype(np.uint8)).resize((84, 84), Image.Resampling.NEAREST))
        lbl_preview.config(image=foto)
        lbl_preview.image = foto   # riferimento necessario, altrimenti l'immagine viene cancellata

    mostra_input(None)

    root.mainloop()


# ==========================================
# RACCOLTA DI UN DATASET PERSONALE
# ==========================================
# Sottocartelle del dataset personale: le immagini 28x28 pronte per la rete (come MNIST) e i disegni
# originali 400x400, conservati per poterli ri-elaborare in futuro se la pre-elaborazione cambia.
CARTELLA_28 = '28x28'
CARTELLA_ORIGINALI = 'originali'


def conta_campioni(cartella):
    """Numero di immagini salvate per ogni cifra (i file si chiamano cifra_<c>_<timestamp>.png)."""
    conteggi = [0] * 10
    path_28 = os.path.join(cartella, CARTELLA_28)
    if os.path.isdir(path_28):
        for nome in os.listdir(path_28):
            if nome.startswith('cifra_') and nome.endswith('.png'):
                conteggi[int(nome.split('_')[1])] += 1
    return conteggi


def salva_dataset(images, labels, originali, cartella='dataset_personale'):
    """Scrive su disco il dataset raccolto, SOSTITUENDO quello salvato in precedenza nella cartella.

    Per ogni disegno vengono scritti due file PNG:
      - <cartella>/28x28/cifra_<c>_<n>.png      immagine 28x28 pre-elaborata come MNIST
      - <cartella>/originali/cifra_<c>_<n>.png  disegno originale 400x400
    """
    for sotto, imgs in ((CARTELLA_28, images), (CARTELLA_ORIGINALI, originali)):
        path = os.path.join(cartella, sotto)
        if os.path.isdir(path):
            shutil.rmtree(path)          # elimina i disegni del dataset precedente
        os.makedirs(path)
        numero = [0] * 10                # numerazione progressiva dei file per ogni cifra
        for img, c in zip(imgs, labels):
            c = int(c)
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(os.path.join(path, f"cifra_{c}_{numero[c]:03d}.png"))
            numero[c] += 1
    print(f"Salvati {len(labels)} disegni nella cartella '{cartella}' (dataset precedente sostituito)")


def raccogli_cifre(campioni_per_cifra=10):
    """Apre una lavagna che chiede di disegnare le cifre da 0 a 9, pre-elaborando ogni disegno come MNIST.

    La raccolta parte sempre da zero e i disegni restano in memoria: su disco non viene scritto nulla,
    quindi il dataset salvato in precedenza non viene toccato. Per salvarli si usa salva_dataset().
    Non è obbligatorio raggiungere campioni_per_cifra: il bottone "Fine" (o la X della finestra) chiude
    la raccolta in qualsiasi momento.

    Restituisce tre array numpy:
      - images:    uint8 (N, 28, 28), le immagini pre-elaborate come MNIST
      - labels:    int64 (N,), la cifra disegnata
      - originali: uint8 (N, 400, 400), i disegni originali
    """
    conteggi = [0] * 10
    raccolta = []                    # disegni raccolti in questa sessione: (cifra, immagine 28x28, originale)

    # --- SETUP VARIABILI ---
    canvas_size = 400
    stato = {'last_x': None, 'last_y': None, 'cifra': 0, 'vuota': True}

    img_pil = Image.new("L", (canvas_size, canvas_size), color=0)
    draw_mem = ImageDraw.Draw(img_pil)

    # Prossima cifra da chiedere: quella con meno campioni, partendo dalla successiva all'ultima disegnata.
    # Così le cifre vengono chieste a rotazione (0, 1, ..., 9, 0, 1, ...) e il dataset resta bilanciato.
    def prossima_cifra(dopo):
        ordine = [(dopo + 1 + k) % 10 for k in range(10)]
        return min(ordine, key=lambda c: conteggi[c])

    stato['cifra'] = prossima_cifra(-1)

    # --- CREAZIONE FINESTRA POP-UP ---
    root = tk.Tk()
    root.title("Dataset personale MNIST")
    root.geometry("1100x600")
    root.configure(bg='#e8e8e8')
    root.resizable(False, False)

    left_frame = tk.Frame(root, bg='#e8e8e8')
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    right_frame = tk.Frame(root, bg='#e8e8e8')
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=20, pady=20)

    # --- COLONNA SINISTRA: ISTRUZIONI, CONTATORI E ULTIMO SALVATAGGIO ---
    lbl_istruzione = tk.Label(left_frame, text="", font=("Helvetica", 26, "bold"), bg='#e8e8e8', fg='#333333')
    lbl_istruzione.pack(pady=(10, 0))
    lbl_stato = tk.Label(left_frame, text="", font=("Helvetica", 12), bg='#e8e8e8', fg='#555555')
    lbl_stato.pack(pady=(5, 20))

    # Riga di contatori, uno per cifra: cliccandone uno si sceglie a mano la cifra da disegnare.
    tk.Label(left_frame, text="Disegni raccolti per cifra (clicca per scegliere la cifra):",
             font=("Helvetica", 10), bg='#e8e8e8', fg='#555555').pack()
    counter_frame = tk.Frame(left_frame, bg='#e8e8e8')
    counter_frame.pack(pady=(5, 20))
    lbl_contatori = []
    for c in range(10):
        lbl = tk.Label(counter_frame, width=4, height=2, font=("Helvetica", 11, "bold"), relief=tk.FLAT, cursor="hand2")
        lbl.grid(row=0, column=c, padx=3)
        lbl.bind("<Button-1>", lambda event, c=c: scegli_cifra(c))
        lbl_contatori.append(lbl)

    tk.Label(left_frame, text="Ultimo disegno (come lo vedrà la rete, 28x28):",
             font=("Helvetica", 10), bg='#e8e8e8', fg='#555555').pack()
    lbl_ultimo = tk.Label(left_frame, bg='black', bd=0)
    lbl_ultimo.pack(pady=5)

    def mostra_ultimo(img28):
        if img28 is None:
            img28 = np.zeros((28, 28))
        foto = ImageTk.PhotoImage(Image.fromarray(img28.astype(np.uint8)).resize((168, 168), Image.Resampling.NEAREST))
        lbl_ultimo.config(image=foto)
        lbl_ultimo.image = foto   # riferimento necessario, altrimenti l'immagine viene cancellata

    def aggiorna_interfaccia(messaggio=None, colore='#555555'):
        c = stato['cifra']
        lbl_istruzione.config(text=f"Disegna la cifra:  {c}")
        totale = sum(conteggi)
        if all(n >= campioni_per_cifra for n in conteggi):
            testo = f"Obiettivo raggiunto: {totale} disegni ({campioni_per_cifra} per cifra). Premi Fine o continua a disegnare."
        else:
            testo = f"Campione {min(conteggi[c] + 1, campioni_per_cifra)} di {campioni_per_cifra} per la cifra {c}  -  totale raccolti: {totale}"
        lbl_stato.config(text=messaggio or testo, fg=colore if messaggio else '#555555')
        for k, lbl in enumerate(lbl_contatori):
            completa = conteggi[k] >= campioni_per_cifra
            lbl.config(text=f"{k}\n{conteggi[k]}",
                       bg='#5cb85c' if completa else ('#4472c4' if k == c else 'white'),
                       fg='white' if (completa or k == c) else '#333333')

    # --- COLONNA DESTRA: LAVAGNA ARROTONDATA ---
    cv = tk.Canvas(right_frame, width=canvas_size, height=canvas_size, bg='#e8e8e8', cursor="cross", highlightthickness=0)
    cv.pack(pady=(0, 20))
    create_rounded_rect(cv, 0, 0, canvas_size, canvas_size, radius=35, fill='black')

    def paint(event):
        x, y = event.x, event.y
        if stato['last_x'] is not None and stato['last_y'] is not None:
            cv.create_line((stato['last_x'], stato['last_y'], x, y), width=24, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            draw_mem.line([stato['last_x'], stato['last_y'], x, y], fill=255, width=24, joint='curve')
            stato['vuota'] = False
        stato['last_x'], stato['last_y'] = x, y

    def reset(event):
        stato['last_x'], stato['last_y'] = None, None

    cv.bind("<B1-Motion>", paint)
    cv.bind("<ButtonRelease-1>", reset)

    # --- FUNZIONI DEI BOTTONI ---
    def pulisci_lavagna():
        cv.delete("all")
        create_rounded_rect(cv, 0, 0, canvas_size, canvas_size, radius=35, fill='black')
        draw_mem.rectangle([0, 0, canvas_size, canvas_size], fill=0)
        stato['vuota'] = True

    def clear():
        pulisci_lavagna()
        aggiorna_interfaccia()

    def scegli_cifra(c):
        stato['cifra'] = c
        aggiorna_interfaccia()

    def salva():
        img_small = preprocess_drawing(img_pil)
        if stato['vuota'] or img_small is None:
            aggiorna_interfaccia("Errore: Lavagna vuota!", "#ff4d4d")
            return
        c = stato['cifra']
        raccolta.append((c, img_small.astype(np.uint8), np.array(img_pil, dtype=np.uint8)))
        conteggi[c] += 1
        mostra_ultimo(img_small)
        pulisci_lavagna()
        stato['cifra'] = prossima_cifra(c)
        aggiorna_interfaccia()

    def fine():
        # Chiude la raccolta: i disegni raccolti vengono restituiti al notebook, anche se l'obiettivo non è stato raggiunto.
        root.destroy()

    def annulla_ultimo():
        if not raccolta:
            aggiorna_interfaccia("Nessun disegno da annullare.", "#ff4d4d")
            return
        c, _, _ = raccolta.pop()
        conteggi[c] -= 1
        stato['cifra'] = c
        mostra_ultimo(raccolta[-1][1] if raccolta else None)
        aggiorna_interfaccia(f"Ultimo disegno della cifra {c} eliminato: ridisegnalo.", "#f0ad4e")

    # --- BOTTONI CUSTOM (Tondeggianti e Interattivi) ---
    btn_frame = tk.Frame(right_frame, bg='#e8e8e8')
    btn_frame.pack()

    RoundedButton(btn_frame, width=120, height=45, text="💾 Salva", bg_color='#5cb85c', fg_color='white', command=salva).pack(side=tk.LEFT, padx=6)
    RoundedButton(btn_frame, width=120, height=45, text="🗑️ Pulisci", bg_color='#f0ad4e', fg_color='white', command=clear).pack(side=tk.LEFT, padx=6)
    RoundedButton(btn_frame, width=120, height=45, text="↩️ Annulla", bg_color='#d9534f', fg_color='white', command=annulla_ultimo).pack(side=tk.LEFT, padx=6)

    # Scorciatoie da tastiera: Invio = Salva, Canc/Backspace = Pulisci
    root.bind("<Return>", lambda event: salva())
    root.bind("<BackSpace>", lambda event: clear())
    root.bind("<Delete>", lambda event: clear())

    tk.Label(right_frame, text="Invio = Salva    Canc = Pulisci", font=("Helvetica", 9), bg='#e8e8e8', fg='#777777').pack(pady=(8, 0))

    # Bottone "Fine" sotto l'anteprima: termina la raccolta con i disegni raccolti fin qui.
    RoundedButton(left_frame, width=160, height=45, text="✅ Fine", bg_color='#4472c4', fg_color='white', command=fine).pack(pady=(15, 0))
    tk.Label(left_frame, text="Puoi terminare in qualsiasi momento: i disegni raccolti passano al notebook.",
             font=("Helvetica", 9), bg='#e8e8e8', fg='#777777').pack(pady=(5, 0))

    mostra_ultimo(None)
    aggiorna_interfaccia()
    root.mainloop()

    if not raccolta:
        return np.zeros((0, 28, 28), np.uint8), np.zeros(0, np.int64), np.zeros((0, canvas_size, canvas_size), np.uint8)
    labels = np.array([c for c, _, _ in raccolta], dtype=np.int64)
    images = np.stack([img for _, img, _ in raccolta])
    originali = np.stack([orig for _, _, orig in raccolta])
    return images, labels, originali

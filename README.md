# blopTracking

Sistema de rastreamento de blobs em camera em tempo real: detecta objetos em
movimento, segue cada um com um ID estavel e mapeia a atividade ao longo de toda
a cena (heatmap + grade de ocupacao).

## Como funciona

1. **Deteccao** — `tracker/detector.py` usa subtracao de fundo (MOG2) com
   suavizacao e operacoes morfologicas para isolar os blobs em movimento e
   extrair seus bounding boxes.
2. **Rastreamento** — `tracker/tracks.py` associa cada blob ao track mais
   proximo (centroide) entre frames, mantendo IDs estaveis e descartando tracks
   perdidos apos alguns frames.
3. **Mapeamento** — `tracker/mapping.py` acumula onde a atividade acontece,
   gerando um heatmap espacial e uma grade de ocupacao, alem de listar as zonas
   mais ativas.
4. **Visualizacao** — `tracker/overlay.py` desenha caixas, rastros, IDs, velocidade
   e o HUD (FPS, numero de blobs, tracks ativos).

## Requisitos

- Python 3.10+
- `opencv-python>=4.8`
- `numpy>=1.24`

## Instalacao e execucao

O script `run.sh` cria o virtualenv, instala as dependencias e executa o tracker:

```bash
./run.sh
```

Para rodar direto com Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python blob_tracker.py
```

Pressione `q` ou `Esc` na janela para encerrar.

## Uso

```bash
./run.sh [opcoes]
```

Exemplos:

```bash
./run.sh --source 0                  # webcam padrao
./run.sh --source video.mp4          # arquivo de video
./run.sh --record saida.mp4          # salva a saida anotada
./run.sh --no-display --record out.mp4  # processa sem abrir janela
./run.sh --show-grid                 # desenha a grade de ocupacao
./run.sh --no-heatmap                # desativa o heatmap
./run.sh --color                     # mantem a imagem colorida
```

### Opcoes

| Opcao | Padrao | Descricao |
| --- | --- | --- |
| `--source` | `0` | Indice da camera (`0`) ou caminho de arquivo de video |
| `--width` / `--height` | `1280` / `720` | Resolucao solicitada |
| `--fps` | `30` | FPS solicitado / usado na gravacao |
| `--min-area` | `400` | Area minima do blob (pixels) |
| `--max-area` | `200000` | Area maxima do blob (pixels) |
| `--max-distance` | `80` | Salto maximo do centroide para manter o ID |
| `--max-missing` | `30` | Frames antes de descartar um track perdido |
| `--history` | `40` | Comprimento do rastro (frames) |
| `--grid` | `16x9` | Tamanho da grade de ocupacao (ex.: `16x9`) |
| `--no-heatmap` | desligado | Desativa o heatmap de atividade |
| `--show-grid` | desligado | Desenha as celulas da grade |
| `--color` | desligado | Mantem a imagem colorida (padrao: P&B) |
| `--record` | - | Salva a saida anotada em arquivo de video |
| `--display` / `--no-display` | ligado | Exibe ou nao a janela |

Ao final, o programa imprime o total de deteccoes mapeadas e as zonas mais
ativas da grade.

## Estrutura

```
blob_tracker.py       # ponto de entrada e loop principal
run.sh                # setup do venv + execucao
requirements.txt
tracker/
  __init__.py
  detector.py         # BlobDetector (MOG2 + morfologia)
  tracks.py           # Track / TrackManager (IDs estaveis)
  mapping.py          # ActivityMap (heatmap + grade)
  overlay.py          # desenho de tracks, rastros e HUD
```

## Dicas

- Deixe a camera parada e a cena sem movimento durante os primeiros frames para
  calibrar o fundo (o HUD mostra "Calibrando fundo...").
- Ajuste `--min-area` / `--max-area` para filtrar ruido ou ignorar objetos muito grandes.
- Diminua `--max-distance` se IDs estiverem trocando entre blobs proximos.

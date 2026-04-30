# CVAE MNIST - Proyecto final de MLOps

Autor: **Jairo**

Este repositorio contiene la adaptación de un notebook previo de Deep Learning a un proyecto de MLOps reproducible, modular, testeable y desplegable. El modelo utilizado es un **Conditional Variational Autoencoder (CVAE)** entrenado sobre **MNIST** para generar dígitos manuscritos condicionados por etiqueta. [1]

## Objetivo del proyecto

El objetivo de esta práctica no es rediseñar el mejor modelo posible, sino aplicar las prácticas de MLOps vistas en la asignatura sobre un proyecto previo ya existente. En concreto, este repositorio incluye notebook, código modular de entrenamiento e inferencia, API con FastAPI, tests, Docker, trazabilidad experimental con Weights & Biases y preparación para despliegue. [1][2][3]

## Estructura del repositorio

```text
cvae-mnist-mlops/
├── notebooks/
├── configs/
├── data/
│   └── raw/
├── models/
├── src/
│   ├── data/
│   │   └── mnist.py
│   ├── models/
│   │   └── cvae.py
│   ├── inference/
│   │   └── generate.py
│   ├── api/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── service.py
│   └── utils/
│       ├── seed.py
│       └── paths.py
├── tests/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

### Descripción de carpetas y archivos principales

- `notebooks/`: contiene el notebook original que se usó como punto de partida del proyecto.
- `configs/`: carpeta reservada para configuración del proyecto.
- `data/raw/`: datos originales o descargas en bruto.
- `models/`: checkpoints del modelo entrenado, incluido `models/cvae.ckpt`.
- `src/data/mnist.py`: carga y preparación del dataset MNIST.
- `src/models/cvae.py`: definición del modelo CVAE.
- `src/inference/generate.py`: utilidades de generación e inferencia.
- `src/api/app.py`: API FastAPI.
- `src/api/schemas.py`: esquemas Pydantic de entrada y salida.
- `src/api/service.py`: servicio que carga el modelo y genera imágenes.
- `src/utils/seed.py`: utilidades de reproducibilidad.
- `src/utils/paths.py`: gestión de rutas del proyecto.
- `tests/`: tests del proyecto.
- `requirements.txt`: dependencias necesarias.
- `Dockerfile`: imagen Docker del servicio.
- `docker-compose.yml`: levantado local simplificado.

## Requisitos previos

Para ejecutar este proyecto desde cero se recomienda disponer de lo siguiente:

- **Git** instalado.
- **Python 3.12** o una versión compatible con las dependencias.
- **pip** instalado.
- **Docker Desktop** instalado y en ejecución si se quiere usar la vía contenedorizada.
- Conexión a Internet para descargar dependencias y, en caso de reentrenar con tracking, sincronizar con Weights & Biases.

## Clonado del repositorio

Clonar el proyecto:

```bash
git clone https://github.com/labs21-cloud/mlops_upm.git
cd mlops_upm
```

> Nota: si el nombre de la carpeta local no coincide exactamente con `cvae-mnist-mlops`, no pasa nada. Lo importante es ejecutar los comandos desde la raíz del repositorio clonado.

## Opción 1: ejecución local con Python

### 1. Crear un entorno virtual

En Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verificar que existe el checkpoint

El proyecto espera encontrar un checkpoint en:

```text
models/cvae.ckpt
```

Si ese archivo ya está presente en el repositorio, puede usarse directamente para inferencia. Si no estuviera disponible, será necesario entrenar el modelo antes de usar la API.

## Entrenamiento del modelo

Para lanzar el entrenamiento desde la raíz del proyecto:

```bash
python -m src.train
```

Si se quiere registrar el experimento en **Weights & Biases**, usar:

```bash
python -m src.train --use-wandb --run-name cvae-mnist-baseline
```

Durante una ejecución real del proyecto ya se registró una run baseline llamada `cvae-mnist-baseline` dentro del proyecto `cvae-mnist-mlops`, con métricas como `train_loss`, `val_loss`, `trainer/global_step`, un checkpoint final en `models/cvae.ckpt` y una grid de muestras en `outputs/sample_grid.png`. [3]

## Lanzar la API en local sin Docker

Con el entorno virtual activado y las dependencias instaladas:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Una vez arrancada la API, se puede comprobar desde el navegador o con una petición HTTP.

### Endpoints principales

- `GET /health`: comprueba que el modelo está cargado y el servicio está listo.
- `POST /generate`: genera imágenes condicionadas a partir de una lista de etiquetas entre 0 y 9.

### Documentación interactiva

Abrir en el navegador:

```text
http://127.0.0.1:8000/docs
```

### Ejemplo de comprobación rápida

En el navegador:

```text
http://127.0.0.1:8000/health
```

O con PowerShell:

```bash
Invoke-WebRequest -Uri http://127.0.0.1:8000/health
```

## Ejemplo de uso del endpoint `/generate`

Ejemplo de petición con PowerShell:

```bash
$body = @{
  labels = @(0,1,2,3)
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

La respuesta devuelve:
- las etiquetas usadas;
- una lista de imágenes codificadas en base64;
- el nombre del modelo.

## Opción 2: ejecución con Docker

Esta es la vía recomendada para reproducir el proyecto de forma más controlada.

### 1. Construir la imagen

```bash
docker build -t cvae-mnist-api .
```

### 2. Ejecutar el contenedor

```bash
docker run -p 8000:8000 cvae-mnist-api
```

### 3. Probar la API

Abrir:

```text
http://127.0.0.1:8000/docs
```

O comprobar salud:

```text
http://127.0.0.1:8000/health
```

## Opción 3: ejecución con Docker Compose

Desde la raíz del proyecto:

```bash
docker compose up --build
```

Para detener el servicio:

```bash
docker compose down
```

Esta opción es la más cómoda para levantar el servicio de forma local durante la corrección.

## Ejecución de tests

Con el entorno virtual activado e instaladas las dependencias:

```bash
pytest tests/
```

Los tests permiten validar al menos la funcionalidad básica del servicio y del proyecto.

## Dependencias principales

El proyecto usa dependencias como PyTorch, Torchvision, PyTorch Lightning, W&B, FastAPI, Uvicorn, Pydantic, Pillow y Pytest. [4]

## Docker actual

El servicio se construye con una imagen `python:3.12-slim`, instala dependencias desde `requirements.txt`, copia el código fuente y el directorio `models/`, expone el puerto 8000 y arranca con Uvicorn sobre `src.api.app:app`. [5]

## Trazabilidad experimental con Weights & Biases

El proyecto integra Weights & Biases para registrar métricas, checkpoints y artefactos durante entrenamiento. [3]

Enlace al proyecto W&B: **PENDIENTE DE INSERTAR**

Enlace al W&B Report: **PENDIENTE DE INSERTAR**

## Endpoint en producción

Endpoint público del servicio: **PENDIENTE DE INSERTAR**

## Estado del proyecto respecto a la práctica

Actualmente este proyecto ya cubre:
- reutilización de un modelo DL previo; [1]
- estructura de proyecto MLOps con notebook, código, API y tests; [1][2]
- versionado en GitHub; [1]
- dockerización local del servicio; [1][5]
- tracking experimental con W&B. [3]

Pendiente de cierre final:
- publicar y enlazar el W&B Report; [1]
- dejar operativo y enlazado el endpoint público. [1]

## Notas para el evaluador

La forma más rápida de probar el proyecto es:

1. Clonar el repositorio.
2. Ejecutar `docker compose up --build`.
3. Abrir `http://127.0.0.1:8000/docs`.
4. Probar `GET /health`.
5. Probar `POST /generate` con etiquetas válidas.

Si se desea revisar la parte de entrenamiento y experimentación, se puede ejecutar `python -m src.train --use-wandb --run-name cvae-mnist-baseline` desde un entorno virtual con dependencias instaladas. [3]
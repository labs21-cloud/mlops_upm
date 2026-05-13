# CVAE MNIST - Proyecto final de MLOps

Autor: **Jairo**

Este repositorio contiene la adaptación de un notebook previo de Deep Learning a un proyecto de MLOps reproducible, modular, testeable y desplegable. El modelo utilizado es un **Conditional Variational Autoencoder (CVAE)** entrenado sobre **MNIST** para generar dígitos manuscritos condicionados por etiqueta.

## Objetivo del proyecto

El objetivo de esta práctica no es rediseñar el mejor modelo posible, sino aplicar las prácticas de MLOps vistas en la asignatura sobre un proyecto previo ya existente. El repositorio incluye notebook, código modular de entrenamiento e inferencia, API con FastAPI, tests, Docker, trazabilidad experimental con Weights & Biases, logging de entrenamiento, automatización básica con GitHub Actions y un endpoint desplegado en producción.

## Estructura del repositorio

```text
mlops_upm/
├── .github/
│   └── workflows/
│       └── ci.yml
├── notebooks/
├── data/
│   └── raw/
├── models/
│   ├── cvae.ckpt
│   └── runs/
├── outputs/
│   └── runs/
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
│   ├── utils/
│   │   └── seed.py
│   └── train.py
├── tests/
│   └── test_api.py
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

### Descripción de carpetas y archivos principales

- `notebooks/`: contiene el notebook original usado como punto de partida.
- `data/raw/`: datos originales o descargas en bruto.
- `models/cvae.ckpt`: checkpoint estable usado por la API para inferencia, tests y despliegue.
- `models/runs/`: checkpoints y modelos finales asociados a cada entrenamiento.
- `outputs/runs/`: artefactos de salida de cada run, como imágenes generadas y logs.
- `src/data/mnist.py`: carga y preparación del dataset MNIST.
- `src/models/cvae.py`: definición del modelo CVAE.
- `src/inference/generate.py`: lógica de generación e inferencia.
- `src/api/app.py`: aplicación FastAPI.
- `src/api/schemas.py`: esquemas de entrada y salida.
- `src/api/service.py`: carga del modelo y generación de imágenes.
- `src/utils/seed.py`: utilidades de reproducibilidad.
- `src/train.py`: script de entrenamiento configurable desde línea de comandos y con logging por run.
- `.github/workflows/ci.yml`: workflow de integración continua con GitHub Actions.
- `tests/test_api.py`: tests básicos de la API.
- `requirements.txt`: dependencias necesarias.
- `Dockerfile`: imagen Docker del servicio.
- `docker-compose.yml`: levantado local simplificado.

## Requisitos previos

### Para ejecutar la API con Docker

- Git
- Docker Desktop

### Para reentrenar el modelo o lanzar tests en local

- Python 3.10 o compatible
- Entorno virtual recomendado
- Dependencias instaladas desde `requirements.txt`

> La forma recomendada de probar el servicio es con Docker Compose.  
> Si se desea reentrenar el modelo o ejecutar tests directamente en local, sí es necesario tener Python y las dependencias instaladas.

## Clonado del repositorio

```bash
git clone https://github.com/labs21-cloud/mlops_upm.git
cd mlops_upm
```

## Ejecución del servicio

### 1. Levantar la API

Desde la raíz del repositorio:

```bash
docker compose up --build
```

Este comando construye la imagen, crea el contenedor y levanta la API.

### 2. Verificar que el servicio está funcionando

Abrir la documentación interactiva en local:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Health check:

[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Si todo ha arrancado correctamente, la API devolverá un estado `ok` y confirmará que el modelo está cargado.

### 3. Probar la generación de imágenes

Usar el endpoint `POST /generate` con un cuerpo como este:

```json
{
  "labels": ,
  "seed": 42
}
```

La respuesta devuelve:
- las etiquetas usadas;
- una lista de imágenes codificadas en base64;
- el nombre del modelo.

### 4. Detener el servicio

```bash
docker compose down
```

## Entrenamiento del modelo

Desde la raíz del proyecto, se puede lanzar un nuevo entrenamiento con:

```bash
python -m src.train
```

Para registrar un experimento en Weights & Biases:

```bash
python -m src.train --use-wandb --run-name cvae-mnist-baseline
```

Ejemplo de nuevo run cambiando nombre y semilla:

```bash
python -m src.train --use-wandb --run-name cvae-mnist-baseline-2 --seed 125
```

Cada entrenamiento crea su propio directorio de artefactos asociado al nombre del run:

- `outputs/runs/<run_name>/`
- `models/runs/<run_name>/`

## Logging y trazabilidad local

El script `src/train.py` genera un archivo de log por cada entrenamiento. Ese log queda guardado dentro de la carpeta del run correspondiente, por ejemplo:

```text
outputs/runs/cvae-mnist-baseline/train.log
```

El log permite seguir el inicio del entrenamiento, argumentos usados, creación de dataloaders, inicialización del modelo, checkpoints generados, generación de muestras y errores si ocurre alguna excepción.

## Trazabilidad experimental con Weights & Biases

El proyecto integra Weights & Biases para registrar métricas, hiperparámetros y resultados relevantes de los experimentos durante el entrenamiento.

Enlace al W&B Report:  
[[https://api.wandb.ai/links/ja-pirona-universidad-polit-cnica-de-madrid/vq72j1ot](https://api.wandb.ai/links/ja-pirona-universidad-polit-cnica-de-madrid/v9mpx5rn)

## Tests

Desde la raíz del proyecto:

```bash
pytest tests/
```

## Integración continua

El proyecto incluye un workflow de GitHub Actions en:

```text
.github/workflows/ci.yml
```

Este workflow ejecuta validaciones automáticas al hacer `push` o abrir un `pull_request`, incluyendo instalación de dependencias, tests y build de Docker.

## Cómo comprobar la automatización

1. Hacer cambios y guardarlos en el repositorio.
2. Subirlos a GitHub con `git add`, `git commit` y `git push`.
3. Entrar en la pestaña **Actions** del repositorio en GitHub.
4. Verificar que el workflow `CI` se ejecuta correctamente.
5. Confirmar que pasan los tests y que la imagen Docker construye sin error.

## Dependencias principales

El proyecto utiliza principalmente:
- PyTorch
- Torchvision
- PyTorch Lightning
- FastAPI
- Uvicorn
- Pydantic
- Pillow
- Pytest
- Weights & Biases
- httpx

## Docker

El servicio se ejecuta en un contenedor basado en `python:3.12-slim`, instala dependencias desde `requirements.txt`, copia el código fuente y el directorio `models/`, y arranca la API con Uvicorn.

## Endpoint en producción

Servicio desplegado en Render:

[https://cvae-mnist-api.onrender.com](https://cvae-mnist-api.onrender.com)

Documentación interactiva en producción:

[https://cvae-mnist-api.onrender.com/docs](https://cvae-mnist-api.onrender.com/docs)

Health check en producción:

[https://cvae-mnist-api.onrender.com/health](https://cvae-mnist-api.onrender.com/health)

## Estado del proyecto respecto a la práctica

Actualmente este proyecto ya cumple con:
- reutilización de un modelo de DL previo;
- estructura de proyecto MLOps con notebook, código, API y tests;
- versionado en GitHub;
- dockerización del servicio;
- trazabilidad experimental con Weights & Biases;
- logging de entrenamiento por run;
- automatización básica mediante GitHub Actions;
- endpoint accesible en producción;
- W&B Report.

## Pruebas realizadas

El proyecto incluye pruebas básicas en `tests/test_api.py` para validar que la API responde correctamente en `/health`, que `POST /generate` funciona con etiquetas válidas, que rechaza etiquetas fuera del rango permitido y que detecta listas vacías.

## Notas para el evaluador

La forma más rápida de probar el proyecto es:

1. Clonar el repositorio.
2. Ejecutar `docker compose up --build`.
3. Abrir `http://127.0.0.1:8000/docs`.
4. Probar `GET /health`.
5. Probar `POST /generate` usando una lista de etiquetas válidas, por ejemplo `[0, 1, 2, 3]`.

Para comprobar el despliegue en producción:

1. Abrir [https://cvae-mnist-api.onrender.com/docs](https://cvae-mnist-api.onrender.com/docs).
2. Probar `GET /health`.
3. Probar `POST /generate` con etiquetas válidas entre `0` y `9`.

Para comprobar la trazabilidad del entrenamiento:

1. Lanzar un run con `python -m src.train --use-wandb --run-name nombre-del-run`.
2. Verificar que se crea `outputs/runs/nombre-del-run/train.log`.
3. Verificar que el run aparece en W&B.
4. Revisar el checkpoint y la imagen generada dentro de las carpetas del run.

## Resultado de las pruebas

Las pruebas manuales de la API verificaron lo siguiente:

- `GET /health` devuelve `200` y confirma que el modelo está cargado.
- `POST /generate` con `labels: [0, 1, 2, 3]` devuelve `200` y responde con `labels`, `images_base64` y `model_name`.
- `POST /generate` con `labels: [-1, 10]` devuelve error porque las etiquetas deben estar entre `0` y `9`.
- `POST /generate` con `labels: []` devuelve error porque la lista debe contener al menos un elemento.

Estas pruebas son coherentes con `tests/test_api.py`, que valida el arranque de la API, la generación con etiquetas válidas y el rechazo de entradas inválidas.

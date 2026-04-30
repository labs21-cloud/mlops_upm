# CVAE MNIST - Proyecto final de MLOps

Autor: **Jairo**

Este repositorio contiene la adaptación de un notebook previo de Deep Learning a un proyecto de MLOps reproducible, modular, testeable y desplegable. El modelo utilizado es un **Conditional Variational Autoencoder (CVAE)** entrenado sobre **MNIST** para generar dígitos manuscritos condicionados por etiqueta.

## Objetivo del proyecto

El objetivo de esta práctica no es rediseñar el mejor modelo posible, sino aplicar las prácticas de MLOps vistas en la asignatura sobre un proyecto previo ya existente. El repositorio incluye notebook, código modular de entrenamiento e inferencia, API con FastAPI, tests, Docker, trazabilidad experimental con Weights & Biases y un endpoint desplegado en producción.

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

- `notebooks/`: contiene el notebook original usado como punto de partida.
- `configs/`: carpeta reservada para configuración del proyecto.
- `data/raw/`: datos originales o descargas en bruto.
- `models/`: checkpoints del modelo entrenado, incluido `models/cvae.ckpt`.
- `src/data/mnist.py`: carga y preparación del dataset MNIST.
- `src/models/cvae.py`: definición del modelo CVAE.
- `src/inference/generate.py`: lógica de generación e inferencia.
- `src/api/app.py`: aplicación FastAPI.
- `src/api/schemas.py`: esquemas de entrada y salida.
- `src/api/service.py`: carga del modelo y generación de imágenes.
- `src/utils/seed.py`: utilidades de reproducibilidad.
- `src/utils/paths.py`: gestión de rutas del proyecto.
- `tests/`: tests del proyecto.
- `requirements.txt`: dependencias necesarias.
- `Dockerfile`: imagen Docker del servicio.
- `docker-compose.yml`: levantado local simplificado.

## Requisitos previos

Antes de empezar, hay que tener instalado:

- Git
- Docker Desktop

> Esta guía está pensada para que cualquier persona pueda ejecutar el proyecto sin configurar manualmente Python ni dependencias locales. La forma recomendada y oficial de uso es con Docker Compose.

## Clonado del repositorio

```bash
git clone https://github.com/labs21-cloud/mlops_upm.git
cd mlops_upm
```

## Ejecución del proyecto

### 1. Levantar el servicio

Desde la raíz del repositorio:

```bash
docker compose up --build
```

Este comando construye la imagen, crea el contenedor y levanta la API.

### 2. Verificar que el servicio está funcionando

Abrir en el navegador:

```text
http://127.0.0.1:8000/docs
```

También se puede comprobar el estado del servicio en:

```text
http://127.0.0.1:8000/health
```

Si todo ha arrancado correctamente, la API devolverá un estado `ok` y confirmará que el modelo está cargado.

### 3. Probar la generación de imágenes

Entrar en la documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

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

Cuando se quiera parar el proyecto:

```bash
docker compose down
```

## Entrenamiento del modelo

Si se desea volver a entrenar el modelo desde la raíz del proyecto:

```bash
python -m src.train
```

Si se quiere registrar el experimento en Weights & Biases:

```bash
python -m src.train --use-wandb --run-name cvae-mnist-baseline
```

Durante una ejecución real del proyecto ya se registró una run baseline llamada `cvae-mnist-baseline` dentro del proyecto `cvae-mnist-mlops`, generando métricas, un checkpoint final en `models/cvae.ckpt` y una imagen de muestras.

## Ejecución de tests

Desde la raíz del proyecto:

```bash
pytest tests/
```

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

## Docker

El servicio se ejecuta en un contenedor basado en `python:3.12-slim`, instala dependencias desde `requirements.txt`, copia el código fuente y el directorio `models/`, y arranca la API con Uvicorn.

## Trazabilidad experimental con Weights & Biases

El proyecto integra Weights & Biases para registrar métricas, checkpoints y artefactos durante entrenamiento.

Enlace al proyecto W&B: **PENDIENTE DE INSERTAR**

Enlace al W&B Report: **PENDIENTE DE INSERTAR**

## Endpoint en producción

Servicio desplegado en Render:

```text
https://cvae-mnist-api.onrender.com
```

Documentación interactiva en producción:

```text
https://cvae-mnist-api.onrender.com/docs
```

Health check en producción:

```text
https://cvae-mnist-api.onrender.com/health
```

## Estado del proyecto respecto a la práctica

Actualmente este proyecto ya cumple con:
- reutilización de un modelo de DL previo;
- estructura de proyecto MLOps con notebook, código, API y tests;
- versionado en GitHub público;
- dockerización del servicio;
- trazabilidad experimental con Weights & Biases;
- endpoint accesible en producción.

Pendiente de cierre final:
- publicar y enlazar el W&B Report.

## Notas para el evaluador

La forma más rápida de probar el proyecto es:

1. Clonar el repositorio.
2. Ejecutar `docker compose up --build`.
3. Abrir `http://127.0.0.1:8000/docs`.
4. Probar `GET /health`.
5. Probar `POST /generate` con etiquetas válidas.

Para comprobar el despliegue en producción:

1. Abrir `https://cvae-mnist-api.onrender.com/docs`.
2. Probar `GET /health`.
3. Probar `POST /generate`.
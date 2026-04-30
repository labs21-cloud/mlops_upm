# CVAE MNIST - Proyecto MLOps

Proyecto de la asignatura MLOps realizado por: **Jairo**

Este repositorio contiene la transformación de un modelo de Deep Learning (CVAE sobre MNIST) en un proyecto MLOps profesional, estructurado para ser reproducible, testeable y desplegable.

## Funcionalidades principales
- **Modelo DL**: CVAE para generación condicionada de dígitos manuscritos.
- **Trazabilidad**: Experimentos registrados en Weights & Biases (métricas, checkpoints y artefactos).
- **API**: Servicio con FastAPI para inferencia.
- **Contenedorización**: Despliegue mediante Docker.
- **Testing**: Suite básica de tests unitarios y de integración.

## Configuración y ejecución local

### 1. Requisitos previos
- Docker y Docker Compose instalados.
- Python 3.x para pruebas locales.

### 2. Clonar el repositorio
```bash
git clone https://github.com/labs21-cloud/mlops_upm.git
cd mlops_upm
```

### 3. Lanzar el servicio con Docker
Para levantar la API de inferencia, ejecuta:
```bash
docker compose up --build
```
El servicio estará disponible en `http://localhost:8000`. Puedes acceder a la documentación interactiva de la API en `http://localhost:8000/docs`.

### 4. Ejecución de tests
Para verificar el proyecto, ejecuta:
```bash
pytest tests/
```

## Enlaces de interés
- **Proyecto en W&B**: [INSERTAR AQUÍ EL ENLACE A TU PROYECTO W&B]
- **W&B Report**: [INSERTAR AQUÍ EL ENLACE A TU REPORT PÚBLICO]
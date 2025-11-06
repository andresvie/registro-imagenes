# Trabajo 2: Fusión de Perspectivas - Registro de Imágenes y Medición del Mundo Real

**Visión por Computador 3009228**  
Semestre 2025-02 - Facultad de Minas  
Universidad Nacional de Colombia – Departamento de Ciencias de la Computación y de la Decisión

## 📋 Descripción del Proyecto

Este proyecto implementa técnicas de registro de imágenes (image registration) para fusionar múltiples perspectivas de un comedor y realizar mediciones del mundo real usando objetos de referencia. El trabajo se divide en tres partes:

1. **Parte 1: Validación con Imágenes Sintéticas (30%)** - Implementado ✅
2. **Parte 2: Registro de las Imágenes del Comedor (40%)** - En progreso
3. **Parte 3: Calibración y Medición (30%)** - En progreso

## 🎯 Objetivo

En este trabajo se ponen en práctica los conceptos fundamentales del registro de imágenes para:
- Crear una vista unificada a partir de múltiples perspectivas
- Utilizar técnicas de detección de características, emparejamiento robusto y transformaciones geométricas
- Fusionar tres imágenes de un comedor
- Extraer mediciones del mundo real a partir de la calibración con objetos de referencia

## 📁 Estructura del Proyecto

```
proyecto-registro-imagenes/
├── README.md                                          # Este archivo
├── requirements.txt                                   # Dependencias del proyecto
├── index.html                                        # Página principal para GitHub Pages
├── _config.yml                                       # Configuración para GitHub Pages
├── data/
│   ├── original/                                     # Imágenes originales del comedor
│   └── synthetic/                                    # Imágenes sintéticas para validación
├── src/
│   ├── __init__.py
│   ├── feature_detection.py                          # Detección de características (SIFT, ORB, AKAZE)
│   ├── matching.py                                   # Emparejamiento de características
│   ├── registration.py                               # Registro y fusión de imágenes
│   ├── evaluator.py                                  # Evaluación de parámetros y estudios
│   ├── utils.py                                      # Utilidades generales
│   └── measurement.py                                # Calibración y medición (Pendiente)
├── notebooks/
│   └── 01_registro_imagenes_proyecto.ipynb          # Validación con imágenes sintéticas ✅
├── results/
│   ├── figures/
│   │   └── punto_1/                                 # Figuras de validación sintética
│   └── measurements/
│       └── punto_1/                                  # Mediciones de validación sintética
└── tests/                                            # Pruebas unitarias ✅
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.12 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio** (o descargar los archivos)

2. **Crear un entorno virtual** (recomendado):
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 📖 Uso

### Parte 1: Validación con Imágenes Sintéticas

Para ejecutar la validación con imágenes sintéticas:

1. **Abrir Jupyter Notebook**:
```bash
jupyter notebook notebooks/01_registro_imagenes_proyecto.ipynb
```

O usando JupyterLab:
```bash
jupyter lab notebooks/01_registro_imagenes_proyecto.ipynb
```

2. **Ejecutar todas las celdas** del notebook. Este notebook:
   - Crea imágenes sintéticas con transformaciones conocidas
   - Aplica el algoritmo de registro usando diferentes detectores (SIFT, ORB, AKAZE)
   - Compara las transformaciones estimadas con las verdaderas (ground truth)
   - Calcula métricas de error (RMSE, error angular, etc.)
   - Analiza cómo los parámetros afectan la calidad del registro

### Resultados Esperados

El notebook genera:
- Imágenes sintéticas base y transformadas en `data/synthetic/`
- Visualizaciones de matches y registros en `results/figures/punto_1/`
- Gráficos comparativos de diferentes métodos de detección
- Análisis del efecto de parámetros en la calidad del registro
- Datasets de resultados y métricas en `results/measurements/punto_1/`

### Pruebas Unitarias

El proyecto incluye pruebas unitarias completas:

```bash
pytest tests/
```

Las pruebas cubren:
- Detección de características
- Emparejamiento de características
- Registro de imágenes
- Evaluación de parámetros
- Utilidades

## 🔧 Funcionalidades Implementadas

### Módulos de Código

1. **`src/feature_detection.py`**: Detección de características
   - Soporte para SIFT, ORB, AKAZE
   - Visualización de keypoints

2. **`src/matching.py`**: Emparejamiento de características
   - FLANN y Brute Force matching
   - Ratio test de Lowe para filtrar matches
   - Visualización de matches

3. **`src/registration.py`**: Registro de imágenes
   - Estimación de homografías con RANSAC
   - Registro de pares de imágenes
   - Transformación de imágenes (warping)

4. **`src/utils.py`**: Utilidades
   - Creación de imágenes sintéticas con transformaciones conocidas
   - Cálculo de métricas de error
   - Visualización de resultados

5. **`src/evaluator.py`**: Evaluación de parámetros
   - Estudios de parámetros (ratio test, detectores)
   - Análisis del efecto de parámetros en la calidad del registro

## 📊 Métricas de Validación

La Parte 1 calcula las siguientes métricas:
- **RMSE (Root Mean Square Error)**: Error en la matriz de homografía y en puntos
- **Error de Rotación**: Diferencia en grados entre rotación verdadera y estimada
- **Error de Escala**: Diferencia en el factor de escala
- **Error de Traslación**: Diferencia en píxeles entre traslación verdadera y estimada
- **Número de Inliers**: Cantidad de matches válidos después de RANSAC
- **Número de Matches**: Total de correspondencias encontradas

## 📈 Resultados Preliminares

### Comparación de Detectores

Los resultados completos están disponibles en `results/measurements/punto_1/comparacion_detectores.csv`:
- **SIFT**: Generalmente el más robusto y preciso, mejor para transformaciones complejas
- **ORB**: Más rápido pero menos preciso en algunos casos, adecuado para tiempo real
- **AKAZE**: Buen balance entre velocidad y precisión, robusto a variaciones de iluminación

### Efecto de Parámetros

Estudio detallado disponible en `results/measurements/punto_1/estudio_ratio_test.csv`:
- El `ratio_threshold` afecta significativamente la calidad del registro
- Valor óptimo típicamente entre 0.7-0.8 para la mayoría de casos
- Factores que afectan la calidad: rotación grande, escala diferente, combinación de transformaciones
- Se incluyen visualizaciones en `results/figures/punto_1/estudio_ratio_test.png`

### Resultados del Dataset Completo

Análisis exhaustivo sobre múltiples transformaciones sintéticas en `results/measurements/punto_1/resultados_dataset.csv`, incluyendo:
- Análisis de errores por tipo de transformación
- Distribución de inliers y matches
- Comparación de precisión entre detectores

## 🔮 Próximos Pasos

- [ ] Parte 2: Registro de las Imágenes del Comedor
  - [ ] Implementar detección de características en imágenes reales
  - [ ] Emparejar y fusionar tres imágenes del comedor
  - [ ] Técnicas de blending para transiciones suaves

- [ ] Parte 3: Calibración y Medición
  - [ ] Calibrar usando objetos de referencia conocidos
  - [ ] Implementar herramienta interactiva de medición
  - [ ] Estimar dimensiones de elementos adicionales

## 📚 Referencias

1. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60(2), 91-110.

2. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.

3. OpenCV Documentation: [Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

4. OpenCV Documentation: [Finding Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

5. PyImageSearch: [Image Stitching](https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)

## 👥 Contribuciones

Este trabajo debe realizarse en equipos. Cada integrante debe contribuir equitativamente.

## 📝 Licencia

Este proyecto es parte de un trabajo académico de la Universidad Nacional de Colombia.

## 🔗 Enlaces Útiles

- [GitHub Pages del Proyecto](https://andresvie.github.io/proyecto-registro-imagenes/)
- [Documentación de OpenCV](https://docs.opencv.org/)
- [Paper original de SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

---

**Nota**: Este README se actualizará conforme se completen las partes 2 y 3 del trabajo.


# Trabajo 2: Fusión de Perspectivas - Registro de Imágenes y Medición del Mundo Real

**Visión por Computador 3009228**  
Semestre 2025-02 - Facultad de Minas  
Universidad Nacional de Colombia – Departamento de Ciencias de la Computación y de la Decisión

## 📋 Descripción del Proyecto

Este proyecto implementa técnicas avanzadas de registro de imágenes (image registration) para fusionar múltiples perspectivas de un comedor y realizar mediciones del mundo real usando objetos de referencia. El trabajo se divide en tres partes principales:

1. **Parte 1: Validación con Imágenes Sintéticas** - Validación de algoritmos con transformaciones conocidas
2. **Parte 2: Registro de las Imágenes del Comedor** - Creación de panorámicas a partir de imágenes reales
3. **Parte 3: Calibración y Medición** - Medición de objetos reales usando calibración con referencias

## 📁 Estructura del Proyecto

```
registro-imagenes/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── pytest.ini                        # Configuración de pytest
├── index.html                        # Página web para GitHub Pages
├── _config.yml                       # Configuración de GitHub Pages
│
├── data/                             # Datos del proyecto
│   ├── original/                     # Imágenes originales del comedor
│   │   ├── IMG01.jpg
│   │   ├── IMG02.jpg
│   │   └── IMG03.jpg
│   └── synthetic/                    # Imágenes sintéticas para validación
│       └── image_*.png
│
├── src/                              # Código fuente principal
│   ├── __init__.py
│   ├── feature_detection.py          # Detección de características (SIFT, ORB, AKAZE)
│   ├── matching.py                   # Emparejamiento de características (FLANN, Brute Force)
│   ├── registration.py               # Registro básico de imágenes
│   ├── stitcher.py                   # Creación de panorámicas con pirámides
│   ├── evaluator.py                  # Evaluación de parámetros y métricas
│   └── utils.py                      # Utilidades generales
│
├── notebooks/                        # Notebooks de Jupyter para ejecución
│   ├── 01_registro_imagenes_proyecto.ipynb      # Parte 1: Validación sintética
│   ├── 02_registro_imagenes_comedor.ipynb       # Parte 2: Registro del comedor
│   ├── 03_registro_imagenes_calibracion_y_medicion.ipynb  # Parte 3: Calibración
│   └── punto3.py                     # Script interactivo para mediciones
│
├── results/                          # Resultados generados
│   ├── figures/                      # Visualizaciones y gráficos
│   │   ├── punto_1/                  # Figuras de validación sintética
│   │   └── punto_3/                  # Figuras de mediciones
│   ├── measurements/                 # Datos y métricas
│   │   ├── punto_1/                  # Resultados de validación
│   │   ├── punto_3/                  # Resultados de calibración
│   │   └── panoramic/                # Resultados de panorámicas
│   ├── homographies/                 # Matrices de homografía guardadas
│   └── panoramic/                    # Panoramas generados
│
└── tests/                            # Pruebas unitarias
    ├── __init__.py
    ├── test_feature_detection.py
    ├── test_matching.py
    ├── test_registration.py
    ├── test_evaluator.py
    └── test_utils.py
```

## 🚀 Instalación

### Requisitos Previos

- **Python 3.12** o superior
- **pip** (gestor de paquetes de Python)
- **Git** (opcional, para clonar el repositorio)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**:
```bash
git clone <url-del-repositorio>
cd registro-imagenes
```

2. **Crear un entorno virtual** (altamente recomendado):
```bash
# En macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python3 -m venv venv
venv\Scripts\activate
```

3. **Instalar las dependencias**:
```bash
pip install -r requirements.txt
```

Las dependencias principales incluyen:
- `opencv-python` y `opencv-contrib-python` (para SIFT y otras funciones avanzadas)
- `numpy` y `scipy` (cálculo numérico)
- `matplotlib` y `seaborn` (visualización)
- `jupyter` y `jupyterlab` (para ejecutar notebooks)
- `pandas` (manejo de datos)
- `pytest` (para pruebas unitarias)

## 📖 Cómo Ejecutar el Proyecto

El proyecto se ejecuta principalmente a través de **Jupyter Notebooks**. Cada notebook corresponde a una parte del trabajo.

### Parte 1: Validación con Imágenes Sintéticas

Esta parte valida los algoritmos de registro usando imágenes sintéticas con transformaciones conocidas.

**Ejecución:**
```bash
# Opción 1: Jupyter Notebook (interfaz clásica)
jupyter notebook notebooks/01_registro_imagenes_proyecto.ipynb

# Opción 2: JupyterLab (interfaz moderna, recomendada)
jupyter lab notebooks/01_registro_imagenes_proyecto.ipynb
```

**Qué hace este notebook:**
- Genera imágenes sintéticas con transformaciones conocidas (rotación, escala, traslación)
- Aplica algoritmos de registro usando diferentes detectores (SIFT, ORB, AKAZE)
- Compara las transformaciones estimadas con las verdaderas (ground truth)
- Calcula métricas de error (RMSE, error angular, error de escala, etc.)
- Analiza el efecto de parámetros (ratio test) en la calidad del registro
- Genera visualizaciones y guarda resultados en `results/`

**Resultados generados:**
- `data/synthetic/`: Imágenes sintéticas generadas
- `results/figures/punto_1/`: Visualizaciones de matches y registros
- `results/measurements/punto_1/`: CSV con métricas y análisis

### Parte 2: Registro de las Imágenes del Comedor

Esta parte crea panorámicas fusionando tres imágenes reales del comedor.

**Ejecución:**
```bash
jupyter lab notebooks/02_registro_imagenes_comedor.ipynb
```

**Qué hace este notebook:**
- Carga las tres imágenes del comedor desde `data/original/`
- Detecta características en cada imagen usando diferentes detectores
- Empareja características entre imágenes adyacentes
- Estima homografías usando RANSAC
- Crea panorámicas usando técnicas de blending (feather, laplacian pyramid)
- Compara resultados entre diferentes detectores
- Guarda panorámicas finales y matrices de homografía

**Resultados generados:**
- `results/panoramic/`: Panoramas finales (SIFT, ORB, AKAZE)
- `results/homographies/`: Matrices de homografía guardadas en JSON
- `results/measurements/panoramic/`: Métricas comparativas

### Parte 3: Calibración y Medición

Esta parte permite medir objetos reales usando calibración con objetos de referencia.

**Ejecución:**

**Opción 1: Usando el notebook (recomendado)**
```bash
jupyter lab notebooks/03_registro_imagenes_calibracion_y_medicion.ipynb
```

**Opción 2: Usando el script interactivo**
```bash
python notebooks/punto3.py
```

**Qué hace esta parte:**
- Carga la panorámica generada en la Parte 2
- Permite seleccionar interactivamente un objeto de referencia conocido
- Calcula la escala (cm/píxel) basada en la referencia
- Valida la calibración midiendo otro objeto conocido
- Permite medir objetos adicionales usando la escala calculada
- Genera visualización con todas las mediciones marcadas
- Guarda resultados en CSV

**Resultados generados:**
- `results/figures/punto_3/mediciones_visualizacion.jpg`: Imagen con mediciones marcadas
- `results/measurements/punto_3/calibracion.csv`: Información de calibración
- `results/measurements/punto_3/mediciones.csv`: Todas las mediciones realizadas

## 🧪 Pruebas Unitarias

El proyecto incluye pruebas unitarias completas para validar la funcionalidad de los módulos.

**Ejecutar todas las pruebas:**
```bash
pytest tests/
```

**Ejecutar pruebas con cobertura:**
```bash
pytest tests/ --cov=src --cov-report=html
```

**Ejecutar pruebas específicas:**
```bash
# Pruebas de detección de características
pytest tests/test_feature_detection.py

# Pruebas de emparejamiento
pytest tests/test_matching.py

# Pruebas de registro
pytest tests/test_registration.py
```

Las pruebas cubren:
- ✅ Detección de características (SIFT, ORB, AKAZE)
- ✅ Emparejamiento de características (FLANN, Brute Force)
- ✅ Registro de imágenes y estimación de homografías
- ✅ Evaluación de parámetros
- ✅ Utilidades y funciones auxiliares

## 🔧 Módulos del Proyecto

### `src/feature_detection.py`
Clase `FeatureDetector` para detectar características en imágenes.
- Soporta SIFT, ORB y AKAZE
- Método `detect_and_compute()` para obtener keypoints y descriptores

### `src/matching.py`
Clase `FeatureMatcher` para emparejar características entre imágenes.
- Soporta FLANN y Brute Force matching
- Implementa ratio test de Lowe para filtrar matches
- Visualización de matches

### `src/registration.py`
Funciones para registro básico de imágenes.
- Estimación de homografías con RANSAC
- Registro de pares de imágenes
- Transformación de imágenes (warping)

### `src/stitcher.py`
Clase `Stitcher` para crear panorámicas avanzadas.
- Usa pirámides gaussianas para detección multi-escala
- Usa pirámides laplacianas para blending multi-banda
- Manejo mejorado de diferencias de exposición
- Transiciones suaves en regiones superpuestas

### `src/evaluator.py`
Herramientas para evaluación y análisis.
- Estudios de parámetros (ratio test, detectores)
- Cálculo de métricas de error
- Análisis comparativo de métodos

### `src/utils.py`
Utilidades generales.
- Creación de imágenes sintéticas con transformaciones conocidas
- Cálculo de métricas de error (RMSE, error angular, etc.)
- Visualización de resultados
- Funciones auxiliares

## 📊 Métricas y Resultados

### Métricas de Validación (Parte 1)

- **RMSE (Root Mean Square Error)**: Error en la matriz de homografía y en puntos
- **Error de Rotación**: Diferencia en grados entre rotación verdadera y estimada
- **Error de Escala**: Diferencia en el factor de escala
- **Error de Traslación**: Diferencia en píxeles entre traslación verdadera y estimada
- **Número de Inliers**: Cantidad de matches válidos después de RANSAC
- **Número de Matches**: Total de correspondencias encontradas

### Comparación de Detectores

Los resultados completos están disponibles en `results/measurements/punto_1/comparacion_detectores.csv`:

- **SIFT**: Mayor robustez y precisión, mejor para transformaciones complejas
- **ORB**: Más rápido pero menos preciso, adecuado para tiempo real
- **AKAZE**: Buen balance entre velocidad y precisión, robusto a variaciones de iluminación

### Resultados de Panorámicas

Las panorámicas generadas se encuentran en `results/panoramic/`:
- `panorama_sift_pyramid.jpg`: Panorama usando SIFT
- `panorama_orb_pyramid.jpg`: Panorama usando ORB
- `panorama_akaze_pyramid.jpg`: Panorama usando AKAZE
- Comparaciones y visualizaciones adicionales

## 🎯 Flujo de Trabajo Recomendado

1. **Instalar dependencias** (ver sección Instalación)
2. **Ejecutar Parte 1** para validar algoritmos con imágenes sintéticas
3. **Revisar resultados** en `results/figures/punto_1/` y `results/measurements/punto_1/`
4. **Ejecutar Parte 2** para crear panorámicas del comedor
5. **Revisar panorámicas** en `results/panoramic/`
6. **Ejecutar Parte 3** para realizar mediciones usando la panorámica
7. **Revisar mediciones** en `results/figures/punto_3/` y `results/measurements/punto_3/`

## 📚 Referencias

1. Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints*. International Journal of Computer Vision, 60(2), 91-110.

2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). *ORB: An efficient alternative to SIFT or SURF*. IEEE International Conference on Computer Vision (ICCV).

3. Alcantarilla, P. F., Nuevo, J., & Bartoli, A. (2013). *Fast explicit diffusion for accelerated features in nonlinear scale spaces*. British Machine Vision Conference (BMVC).

4. Fischler, M. A., & Bolles, R. C. (1981). *Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography*. Communications of the ACM, 24(6), 381-395.

5. Burt, P. J., & Adelson, E. H. (1983). *A multiresolution spline with application to image mosaics*. ACM Transactions on Graphics, 2(4), 217-236.

6. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. Chapter 6: Feature Detection and Matching.

7. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. Chapter 4: Estimation - 2D Projective Transformations.

8. Brown, M., & Lowe, D. G. (2007). *Automatic panoramic image stitching using invariant features*. International Journal of Computer Vision, 74(1), 59-73.

## 🔗 Enlaces Útiles

- [GitHub Pages del Proyecto](https://andresvie.github.io/proyecto-registro-imagenes/)
- [Documentación de OpenCV](https://docs.opencv.org/)
- [Paper original de SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

## 👥 Contribuciones

Este trabajo fue desarrollado en equipo por:

- **Parte 1: Validación con Imágenes Sintéticas**
  - Carlos Andrés Viera Mosquera (cviera@unal.edu.co)

- **Parte 2: Registro de las Imágenes del Comedor**
  - Carlos Andrés Viera Mosquera (cviera@unal.edu.co)
  - Yenifer Tatiana Guavita Ospino (yguavita@unal.edu.co)

- **Parte 3: Calibración y Medición**
  - Lina María Montoya Zuluaga (limontoyaz@unal.edu.co)
  - Yojan Tamayo Montoya (ytamayom@unal.edu.co)

## 📝 Licencia

Este proyecto es parte de un trabajo académico de la Universidad Nacional de Colombia.

---

**Nota**: Para más detalles sobre los resultados y análisis, consulta los notebooks y los archivos en `results/`.

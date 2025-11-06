import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ==========================================================
# 1. Función para seleccionar puntos y medir distancia
# ==========================================================
def medir_distancia_interactiva(img, titulo):
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    puntos = plt.ginput(2, timeout=0)
    plt.close()

    if len(puntos) != 2:
        print("Debes seleccionar exactamente dos puntos.")
        return None, None, None
    
    (x1, y1), (x2, y2) = puntos
    dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"Coordenadas seleccionadas:")
    print(f"  Punto 1: ({x1:.1f}, {y1:.1f})")
    print(f"  Punto 2: ({x2:.1f}, {y2:.1f})")
    print(f"Distancia medida: {dist_px:.2f} píxeles\n")
    return dist_px, (x1, y1), (x2, y2)

# ==========================================================
# 2. Cargar la imagen fusionada
# ==========================================================
image_path = r"results/panoramic/panorama_sift_pyramid.jpg"
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta.")

# ==========================================================
# 3. Calibración con un objeto de referencia
# ==========================================================
print("\n Selecciona el largo del cuadro como objeto de referencia.")
dist_ref_px, p1_ref, p2_ref = medir_distancia_interactiva(img_bgr, "Selecciona el largo del cuadro")
if dist_ref_px is None:
    raise ValueError("No se seleccionó correctamente el objeto de referencia.")

longitud_real_ref = float(input("Confirma la longitud real del largo de cuadro en cm (117): "))
escala = longitud_real_ref / dist_ref_px
print(f"\n Escala calculada: {escala:.4f} cm/píxel\n")

# ==========================================================
# 4. Validación con otro objeto conocido (para incertidumbre)
# ==========================================================
print("\n Ahora selecciona el ancho de la mesa para validar.")
dist_valid_px, p1_val, p2_val = medir_distancia_interactiva(img_bgr, "Selecciona el ancho de la mesa")
if dist_valid_px is None:
    raise ValueError("No se seleccionó correctamente el objeto de validación.")

longitud_real_valid = float(input("Introduce la longitud real del ancho de la mesa en cm (ancho de la mesa = 161.1): "))
longitud_medida_valid = dist_valid_px * escala

error_relativo = abs(longitud_medida_valid - longitud_real_valid) / longitud_real_valid * 100
print(f"Validación:")
print(f"  Longitud medida: {longitud_medida_valid:.2f} cm")
print(f"  Longitud real:   {longitud_real_valid:.2f} cm")
print(f"  Error relativo: ±{error_relativo:.2f}%\n")

# ==========================================================
# 5. Medición de otros objetos desconocidos
# ==========================================================
num_obj = int(input("¿Cuántos objetos deseas medir? "))

objs = []
for i in range(num_obj):
    print(f"\n Objeto {i+1}: selecciona los extremos para medir.")
    dist_px, p1, p2 = medir_distancia_interactiva(img_bgr, f"Selecciona los extremos del objeto {i+1} de {num_obj}")
    if dist_px is None:
        continue
    medida_real = dist_px * escala
    print(f" Longitud estimada: {medida_real:.2f} cm")
    objs.append({
        "dist_px": dist_px,
        "medida_real": medida_real,
        "p1": p1,
        "p2": p2
    })

# ==========================================================
# 6. Mostrar gráficamente todas las mediciones
# ==========================================================
img_vis = img_bgr.copy()

# Calcular factores de escala basados en la resolución de la imagen
height, width = img_vis.shape[:2]
diagonal = np.sqrt(width**2 + height**2)
line_thickness = max(2, int(diagonal * 0.002))  # Grosor de línea proporcional
font_scale = diagonal * 0.0007  # Escala de fuente proporcional
text_thickness = max(2, int(diagonal * 0.001))  # Grosor del texto proporcional
text_offset = int(diagonal * 0.02)  # Desplazamiento del texto proporcional

# Referencia (azul)
cv2.line(img_vis, (int(p1_ref[0]), int(p1_ref[1])), (int(p2_ref[0]), int(p2_ref[1])), (255, 0, 0), line_thickness)
cv2.putText(img_vis, f"Ref: {longitud_real_ref}cm", (int(p1_ref[0]), int(p1_ref[1])-text_offset),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), text_thickness)

# Validación (rojo)
cv2.line(img_vis, (int(p1_val[0]), int(p1_val[1])), (int(p2_val[0]), int(p2_val[1])), (0, 0, 255), line_thickness)
cv2.putText(img_vis, f"Val: {longitud_medida_valid:.1f}cm ({error_relativo:.1f}%)",
            (int(p1_val[0]), int(p1_val[1])-text_offset),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)

# Objetos medidos (verde)
for o in objs:
    cv2.line(img_vis, (int(o["p1"][0]), int(o["p1"][1])), (int(o["p2"][0]), int(o["p2"][1])), (0, 255, 0), line_thickness)
    cv2.putText(img_vis, f"{o['medida_real']:.1f}cm", (int(o["p1"][0]), int(o["p1"][1])-text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)

# Guardar la imagen con las mediciones
figures_dir = Path("results/figures/punto_3")
figures_dir.mkdir(parents=True, exist_ok=True)
image_path = figures_dir / f'mediciones_visualizacion.jpg'
cv2.imwrite(str(image_path), img_vis)
print(f"\nImagen con mediciones guardada en: {image_path}")

# Mostrar la imagen
plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title("Calibración, validación e incertidumbre")
plt.axis("off")
plt.show()

# ==========================================================
# 7. Reporte final y guardado de resultados
# ==========================================================
print("\n Resumen:")
print(f"Escala: {escala:.4f} cm/píxel")
print(f"Incertidumbre (error relativo de validación): ±{error_relativo:.2f}%")

# Crear el directorio si no existe
output_dir = Path("results/measurements/punto_3")
output_dir.mkdir(parents=True, exist_ok=True)

# Preparar los datos de medición de referencia y validación
mediciones = [{
    'tipo': 'Referencia',
    'objeto': 'Cuadro',
    'longitud_medida_px': dist_ref_px,
    'longitud_medida_cm': longitud_real_ref,
    'longitud_real_cm': longitud_real_ref,
    'error_relativo': 0.0
}, {
    'tipo': 'Validación',
    'objeto': 'Mesa',
    'longitud_medida_px': dist_valid_px,
    'longitud_medida_cm': longitud_medida_valid,
    'longitud_real_cm': longitud_real_valid,
    'error_relativo': error_relativo
}]

# Añadir las mediciones de otros objetos
for i, obj in enumerate(objs, 1):
    mediciones.append({
        'tipo': 'Medición',
        'objeto': f'Objeto {i}',
        'longitud_medida_px': obj['dist_px'],
        'longitud_medida_cm': obj['medida_real'],
        'longitud_real_cm': None,  # No conocemos la longitud real
        'error_relativo': None     # No podemos calcular el error
    })

# Crear el DataFrame y guardar a CSV
df = pd.DataFrame(mediciones)
csv_path = output_dir / f'mediciones.csv'
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"\nResultados guardados en: {csv_path}")

# Guardar también un resumen de la calibración
calibration_info = {
    'escala_cm_px': [escala],
    'error_validacion': [error_relativo],
}
df_calibration = pd.DataFrame(calibration_info)
calibration_path = output_dir / f'calibracion.csv'
df_calibration.to_csv(calibration_path, index=False, encoding='utf-8')

print(f"Información de calibración guardada en: {calibration_path}")

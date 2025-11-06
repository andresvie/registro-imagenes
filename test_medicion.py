import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Función para seleccionar puntos y medir distancia
# ==========================================================
def medir_distancia_interactiva(img, titulo="Selecciona dos puntos"):
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
image_path = r"C:\Users\linam\Downloads\IMG01.jpg"
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta.")

# ==========================================================
# 3. Calibración con un objeto de referencia
# ==========================================================
print("\n Selecciona los extremos del objeto de referencia (por ejemplo, el cuadro).")
dist_ref_px, p1_ref, p2_ref = medir_distancia_interactiva(img_bgr)
if dist_ref_px is None:
    raise ValueError("No se seleccionó correctamente el objeto de referencia.")

longitud_real_ref = float(input("Introduce la longitud real del objeto de referencia (en cm): "))
escala = longitud_real_ref / dist_ref_px
print(f"\n Escala calculada: {escala:.4f} cm/píxel\n")

# ==========================================================
# 4. Validación con otro objeto conocido (para incertidumbre)
# ==========================================================
print("\n Ahora selecciona los extremos de otro objeto conocido (por ejemplo, la mesa) para validar.")
dist_valid_px, p1_val, p2_val = medir_distancia_interactiva(img_bgr)
if dist_valid_px is None:
    raise ValueError("No se seleccionó correctamente el objeto de validación.")

longitud_real_valid = float(input("Introduce la longitud real del objeto de validación (en cm): "))
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
    dist_px, p1, p2 = medir_distancia_interactiva(img_bgr)
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

# Referencia (azul)
cv2.line(img_vis, (int(p1_ref[0]), int(p1_ref[1])), (int(p2_ref[0]), int(p2_ref[1])), (255, 0, 0), 2)
cv2.putText(img_vis, f"Ref: {longitud_real_ref}cm", (int(p1_ref[0]), int(p1_ref[1])-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Validación (rojo)
cv2.line(img_vis, (int(p1_val[0]), int(p1_val[1])), (int(p2_val[0]), int(p2_val[1])), (0, 0, 255), 2)
cv2.putText(img_vis, f"Val: {longitud_medida_valid:.1f}cm ({error_relativo:.1f}%)",
            (int(p1_val[0]), int(p1_val[1])-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Objetos medidos (verde)
for o in objs:
    cv2.line(img_vis, (int(o["p1"][0]), int(o["p1"][1])), (int(o["p2"][0]), int(o["p2"][1])), (0, 255, 0), 2)
    cv2.putText(img_vis, f"{o['medida_real']:.1f}cm", (int(o["p1"][0]), int(o["p1"][1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title("Calibración, validación e incertidumbre")
plt.axis("off")
plt.show()

# ==========================================================
# 7. Reporte final
# ==========================================================
print("\n Resumen:")
print(f"Escala: {escala:.4f} cm/píxel")
print(f"Incertidumbre (error relativo de validación): ±{error_relativo:.2f}%")

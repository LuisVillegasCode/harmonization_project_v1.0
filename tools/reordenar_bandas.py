import rasterio
import numpy as np
from rasterio.enums import Resampling

# ===========================
# CONFIGURACIÓN DE ENTRADA
# ===========================

# Archivo original y de salida
entrada = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\data\processed\IMG_PER1_ORT_MS_001586\analysis_ready_data.tif"
salida = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\data\processed\IMG_PER1_ORT_MS_001586\P1_DOS.tif"

# Nuevo orden de bandas (1-based)
# Ejemplo: [3, 2, 1, 4] = reordenar R,G,B,NIR → G,B,R,NIR
nuevo_orden = [3, 2, 1, 4]

# ===========================
# REORDENAMIENTO DE BANDAS
# ===========================

print("=== Reordenando bandas ===")

with rasterio.open(entrada) as src:
    # Copiamos el perfil completo (no mutamos el original)
    perfil = src.profile.copy()
    perfil.update(count=len(nuevo_orden))

    with rasterio.open(salida, "w", **perfil) as dst:
        # Intentar copiar la interpretación de color (RGB/NIR)
        try:
            dst.colorinterp = tuple(src.colorinterp[b - 1] for b in nuevo_orden)
        except Exception:
            pass  # No es crítico si no existe esta propiedad

        # Copiar cada banda en nuevo orden
        for i, banda_idx in enumerate(nuevo_orden, start=1):
            banda = src.read(banda_idx)  # sin remuestreo
            dst.write(banda, i)

            # Copiar descripción de banda (si existe)
            if src.descriptions and len(src.descriptions) >= banda_idx:
                dst.set_band_description(i, src.descriptions[banda_idx - 1])

            # Copiar tags específicos de la banda (si existen)
            band_tags = src.tags(banda_idx)
            if band_tags:
                dst.update_tags(i, **band_tags)

        # Copiar tags del dataset
        dst.update_tags(**src.tags())

print("✅ Reordenamiento completado correctamente.")
print("===========================================")

# ===========================
# VERIFICACIÓN DE CALIDAD
# ===========================

print("\n=== Verificando integridad radiométrica ===")

with rasterio.open(entrada) as src, rasterio.open(salida) as dst:
    for i, b in enumerate(nuevo_orden, start=1):
        orig = src.read(b)
        new = dst.read(i)

        # Verificar igualdad exacta (bit a bit)
        if np.array_equal(orig, new):
            print(f"✅ Banda original {b} → nueva {i}: idéntica")
        else:
            # Si no son iguales, mostrar diferencias mínimas/máximas
            diff = new.astype(np.int64) - orig.astype(np.int64)
            print(f"⚠️  Banda {b} → nueva {i}: difiere (min={diff.min()}, max={diff.max()})")

print("===========================================")
print("Verificación finalizada.")
print("Si todas las bandas son 'idénticas', la calidad del input se preservó totalmente.")

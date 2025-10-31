import rasterio
from rasterio.enums import Resampling

# Archivo original
entrada = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\harmonization_project_v1.0\inputs\P1_aligned_to_S2.tif"
salida = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\harmonization_project_v1.0\inputs\P1_aligned_to_S2_ordenada.tif"

# Orden deseado (por ejemplo: [3, 4, 2, 1])
nuevo_orden = [3, 2, 1, 4]

with rasterio.open(entrada) as src:
    perfil = src.profile
    perfil.update(count=len(nuevo_orden))

    with rasterio.open(salida, "w", **perfil) as dst:
        for i, banda_idx in enumerate(nuevo_orden, start=1):
            banda = src.read(banda_idx)
            dst.write(banda, i)
            # Si existen descripciones:
            if src.descriptions and len(src.descriptions) >= banda_idx:
                dst.set_band_description(i, src.descriptions[banda_idx - 1])
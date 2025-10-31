import rasterio

ruta = r"C:\Users\51926\Desktop\Proyecto de Investigación\IA\proyecto_deforestacion\harmonization_project_v1.0\inputs\P1_aligned_to_S2_ordenada.tif"
with rasterio.open(ruta) as src:
    print("Número de bandas:", src.count)
    for i in range(1, src.count + 1):
        desc = src.descriptions[i-1]  # nombre o descripción de la banda
        print(f"Banda {i}: {desc}")
    print(src.tags())  # metadatos generales
    print(src.tags(1)) # metadatos específicos de la banda 1

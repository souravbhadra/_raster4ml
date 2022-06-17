import rasterio


def check_projection(src, shape):
    """
    Check if the raster and shapefile has same projection or not. If not, it reprojects
    the shapefile.
    """
    raster_epsg = src.crs.to_epsg()
    shape_epsg = shape.crs.to_epsg()
    if shape_epsg != raster_epsg:
        shape.set_crs(epsg=raster_epsg)
    else:
        pass
    return shape


def get_all_feature_names():
    """
    Get a list containing all the feature names.
    """
    features = ['ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI', 'EVI', 'GEMI', 'GARI',
                'GCI', 'GDVI', 'GLI', 'GNDVI', 'GOSAVI', 'GRVI', 'GSAVI', 'GVI', 'IPVI',
                'LCAI', 'MCARI', 'MNLI', 'MNDWI', 'MRENDVI', 'MRESR', 'MSR', 'MSAVI_2', 
                'MTVI', 'MSI', 'NLI', 'NBR', 'NBRT_1', 'NDBI', 'NDII', 'NDLI', 'NDMI', 
                'NDNI', 'NDSI', 'NDVI', 'NDWI', 'NMDI', 'OSAVI', 'PRI', 'PSRI', 'RENDVI', 
                'RDVI', 'SR_1', 'SAVI', 'SIPI', 'TCARI', 'TDVI', 'TVI', 'VARI', 'VREI_1', 
                'VREI_2', 'WBI', 'WDRVI']
    return features


def save_raster(src, array, out_path, **kwargs):
    """
    Save a raster data.
    """
    profile = src.profile
    profile.update(**kwargs)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array)

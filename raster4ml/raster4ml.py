import os
import glob
import time
import rasterio
from rasterio.mask import mask
import cv2
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from raster4ml import utils


def stack_bands(image_paths, out_file):
    """
        Stack the bands together into a single image.
            Args:
                image_paths (list): Path of the images to stack.
                out_file (str): Full path of the output data (has to be .tif). If not given, the np array of the result is returned as np array.
            Output:
                stack_image saved as a tif image.
        """
    # Read all the individual bands
    try:
        srcs = [rasterio.open(image_path) for image_path in image_paths]
    except:
        raise ValueError("Invalid image paths. Please check the band paths again.")
    
    # Check if the x and y are same for all the bands or not
    xy = np.array([(src.height, src.width) for src in srcs])
    # Get max x and y
    max_x = xy[:, 0].max()
    max_y = xy[:, 1].max()  

    if srcs[0].nodata is None:
        nodata_value = 0
    else:
        nodata_value = srcs[0].nodata

    # Empty array to hold stack image
    stack_img = np.zeros(shape=(len(image_paths), xy[:, 0].max(), xy[:, 1].max()))
    # Loop through each src
    for i, src in enumerate(srcs):
        x, y = src.height, src.width
        if x < max_x:
            img = src.read(1)
            img[img==nodata_value] = np.nan
            img = cv2.resize(img, (max_x, max_y), interpolation=cv2.INTER_NEAREST)
            print(f"{image_paths[i]} resized.")
            stack_img[i, :, :] = img
        else:
            img = src.read(1)
            stack_img[i, :, :] = img

    # Save
    utils.save_raster(srcs[0], stack_img, out_file,
                      driver='GTiff', width=max_y, height=max_x,
                      count=len(image_paths))


class FeatureCalculator():
    

    def __init__(self, image_path, wavelengths, bit_depth=None):

        self.image_path = image_path

        self.src = rasterio.open(self.image_path)
        self.img = self.src.read()
        self.img = self.img.astype('float')
        self.img = np.moveaxis(self.img, 0, 2)

        # Nodata stuffs
        if self.src.nodata is None:
            self.img[self.img==0] = np.nan
        else:
            self.img[self.img==self.src.nodata] = np.nan
        
        # Apply bit_depth if given
        if bit_depth is not None:
            if isinstance(bit_depth, int):
                self.img = self.img / 2.**bit_depth
            else:
                raise ValueError("Invalid input in bit_depth. Only int is accepted.")
        else:
            pass

        # Check if number of elements in wavelengths matches the number of bands
        if len(wavelengths) != self.img.shape[2]:
            raise ValueError("The number of elements in wavelengths and number of \
                bands do not match.")
        
        # Find band wavelengths
        try:
            if any(isinstance(el, list) for el in wavelengths):
                self.wavelengths = [(el[0] + el[1])/2 for el in wavelengths]
            else: #list of lists
                self.wavelengths = [float(i) for i in wavelengths]
        except:
            raise ValueError("Invalid input of wavelengths. It has to be a list of \
                integers for center wavelengths and list of lists with start and end \
                wavelength for each band.")
            

        self.R445 = self.nearest_band(445)
        self.R450 = self.nearest_band(450)
        self.R475 = self.nearest_band(475)
        self.R485 = self.nearest_band(485)
        self.R500 = self.nearest_band(500)
        self.R510 = self.nearest_band(510)
        self.R531 = self.nearest_band(531)
        self.R550 = self.nearest_band(550)
        self.R560 = self.nearest_band(560)
        self.R570 = self.nearest_band(570)
        self.R660 = self.nearest_band(660)
        self.R670 = self.nearest_band(670)
        self.R680 = self.nearest_band(680)
        self.R700 = self.nearest_band(700)
        self.R705 = self.nearest_band(705)
        self.R715 = self.nearest_band(715)
        self.R720 = self.nearest_band(720)
        self.R726 = self.nearest_band(726)
        self.R734 = self.nearest_band(734)
        self.R740 = self.nearest_band(740)
        self.R747 = self.nearest_band(747)
        self.R750 = self.nearest_band(750)
        self.R795 = self.nearest_band(795)
        self.R800 = self.nearest_band(800)
        self.R819 = self.nearest_band(819)
        self.R830 = self.nearest_band(830)
        self.R850 = self.nearest_band(850)
        self.R857 = self.nearest_band(857)
        self.R860 = self.nearest_band(860)
        self.R900 = self.nearest_band(900)
        self.R970 = self.nearest_band(970)
        self.R990 = self.nearest_band(990)
        self.R1145 = self.nearest_band(1145)
        self.R1241 = self.nearest_band(1241)
        self.R1510 = self.nearest_band(1510)
        self.R1599 = self.nearest_band(1599)
        self.R1640 = self.nearest_band(1640)
        self.R1649 = self.nearest_band(1649)
        self.R1650 = self.nearest_band(1650)
        self.R1680 = self.nearest_band(1680)
        self.R1754 = self.nearest_band(1754)
        self.R2000 = self.nearest_band(2000)
        self.R2100 = self.nearest_band(2100)
        self.R2130 = self.nearest_band(2130)
        self.R2165 = self.nearest_band(2165)
        self.R2200 = self.nearest_band(2200)
        self.R2205 = self.nearest_band(2205)
        self.R2215 = self.nearest_band(2215)
        self.R2330 = self.nearest_band(2330)


    def nearest_band(self, wavelength):
        difference = np.abs(np.array(self.wavelengths) - wavelength)
        if difference.min() < 100:
            return self.img[:, :, difference.argmin()]
        else:
            return None

    def ARI_1(self):
        # Anthocyanin Reflectance Index 1 (ARI1)
        # https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2
        return (1./self.R550)-(1./self.R700)

    def ARI_2(self):
        # Anthocyanin Reflectance Index 2 (ARI2)
        # https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2
        return self.R800*((1./self.R550)-(1./self.R700))

    def ARVI(self):
        # Atmospherically Resistant Vegetation Index (ARVI)
        # https://doi.org/10.1109/36.134076
        return (self.R800-(self.R680-(self.R450-self.R680)))/(self.R800+(self.R680-(self.R450-self.R680)))

    def CRI_1(self):
        # Carotenoid Reflectance Index 1 (CRI1)
        # https://calmit.unl.edu/people/agitelson2/pdf/08_2002_P&P_carotenoid.pdf
        return (1./self.R510)-(1./self.R550)

    def CRI_2(self):
        # Carotenoid Reflectance Index 2 (CRI2)
        # https://calmit.unl.edu/people/agitelson2/pdf/08_2002_P&P_carotenoid.pdf
        return (1./self.R510)-(1./self.R700)

    def CAI(self):
        # Cellulose Absorption Index (CAI)
        # https://naldc.nal.usda.gov/download/12951/PDF
        return 0.5*(self.R2000+self.R2200)-self.R2100

    def DVI(self):
        # Difference Vegetation Index (DVI)
        # https://doi.org/10.1016/0034-4257(79)90013-0
        return self.R850-self.R660

    def EVI(self):
        # Enhanced Vegetation Index (EVI)
        # https://doi.org/10.1016/S0034-4257(02)00096-2
        return 2.5*((self.R850-self.R660)/(self.R850+(6.*self.R660)-(7.5*self.R475)+1.))

    def GEMI(self):
        # Global Environmental Monitoring Index (GEMI)
        # https://link.springer.com/article/10.1007/BF00031911
        eta = (2.*(np.square(self.R850)-np.square(self.R660))+1.5*self.R850+0.5*self.R660)/(self.R850+self.R660+0.5)
        return eta*(1.-0.25*eta)-((self.R660-0.125)/(1.-self.R660))

    def GARI(self):
        # Green Atmospherically Resistant Index (GARI)
        # https://doi.org/10.1016/S0034-4257(96)00072-7
        return (self.R850-(self.R550-1.7*(self.R475-self.R660)))/(self.R850+(self.R550-1.7*(self.R475-self.R660)))

    def GCI(self):
        # Green Chlorophyll Index (GCI)
        # https://doi.org/10.1078/0176-1617-00887
        return (self.R850/self.R550)-1.

    def GDVI(self):
        # Green Difference Vegetation Index (GDVI)
        # https://repository.lib.ncsu.edu/handle/1840.16/4200
        return self.R850-self.R550

    def GLI(self):
        # Green Leaf Index (GLI)
        # https://doi.org/10.1080/10106040108542184
        return (self.R550-self.R660+self.R550-self.R475)/(2.*self.R550+self.R660+self.R475)

    def GNDVI(self):
        # Green Normalized Difference Vegetation Index (GNDVI)
        # https://doi.org/10.1016/S0273-1177(97)01133-2
        return (self.R850-self.R550)/(self.R850+self.R550)

    def GOSAVI(self):
        # Green Optimized Soil Adjusted Vegetation Index (GOSAVI)
        # https://repository.lib.ncsu.edu/handle/1840.16/4200
        return 'GOSAVI', (self.R850-self.R550)/(self.R850+self.R550+0.16)

    def GRVI(self):
        # Green Ratio Vegetation Index (GRVI)
        # https://doi.org/10.2134/agronj2005.0200
        return self.R850/self.R550

    def GSAVI(self):
        # Green Soil Adjusted Vegetation Index (GSAVI)
        # https://repository.lib.ncsu.edu/handle/1840.16/4200
        return 1.5*((self.R850-self.R550)/(self.R850+self.R550+0.5))

    def GVI(self):
        # Green Vegetation Index (GVI)
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.461.6381&rep=rep1&type=pdf
        return (-0.2848*self.R485)+(-0.2435*self.R560)+(-0.5436*self.R660)+(0.7243*self.R830)+(0.0840*self.R1650)+(-0.1800*self.R2215)

    def IPVI(self):
        # Infrared Percentage Vegetation Index (IPVI)
        # https://doi.org/10.1016/0034-4257(90)90085-Z
        return self.R850/(self.R850+self.R660)

    def LCAI(self):
        # Lignin Cellulose Absorption Index (LCAI)
        # https://doi.org/10.2134/agronj2003.0291
        return 100.*(self.R2205-self.R2165+self.R2205-self.R2330)

    def MCARI(self):
        # Modified Chlorophyll Absorption Ratio Index (MCARI)
        # https://doi.org/10.1016/S0034-4257(00)00113-9
        return ((self.R700-self.R670)-0.2*(self.R700-self.R550))*(self.R700/self.R670)

    def MNLI(self):
        # Modified Non-Linear Index (MNLI)
        # https://www.asprs.org/a/publications/proceedings/pecora17/0041.pdf
        return ((np.square(self.R850)-self.R660)*1.5)/(np.square(self.R850)+self.R660+0.5)

    def MNDWI(self):
        # Modified Normalized Difference Water Index (MNDWI)
        # https://doi.org/10.1080/01431160600589179
        # https://doi.org/10.1080/01431169608948714
        return (self.R550-self.R1650)/(self.R550+self.R1650)

    def MRENDVI(self):
        # Modified Red Edge Normalized Difference Vegetation Index (MRENDVI)
        # https://doi.org/10.1016/S0176-1617(99)80314-9
        # https://doi.org/10.1016/S0034-4257(02)00010-X
        return (self.R750-self.R705)/(self.R750+self.R705-2.*self.R445)

    def MRESR(self):
        # Modified Red Edge Simple Ratio (MRESR)
        # https://doi.org/10.1016/S0034-4257(02)00010-X
        # https://doi.org/10.1016/S0176-1617(99)80314-9
        return (self.R750-self.R445)/(self.R705-self.R445)

    def MSR(self):
        # Modified Red Edge Simple Ratio (MRESR)
        # https://doi.org/10.1080/07038992.1996.10855178
        return ((self.R850/self.R660)-1.)/(np.sqrt(self.R850/self.R660)+1.)

    def MSAVI_2(self):
        # Modified Soil Adjusted Vegetation Index 2 (MSAVI2)
        # https://doi.org/10.1016/0034-4257(94)90134-1
        return (2.*self.R850+1.-np.sqrt(np.square(2.*self.R850)-8.*(self.R850-self.R660)))/2.

    def MTVI(self):
        # Modified Triangular Vegetation Index (MTVI)
        # https://doi.org/10.1016/j.rse.2003.12.013
        return 1.2*(1.2*(self.R800-self.R550)-2.5*(self.R670-self.R550))

    def MSI(self):
        # Moisture Stress Index (MSI)
        # https://doi.org/10.1016/0034-4257(89)90046-1
        # https://doi.org/10.1016/S0034-4257(01)00191-2
        return self.R1599/self.R819

    def NLI(self):
        # Non-Linear Index (NLI)
        # https://doi.org/10.1080/02757259409532252
        return (np.square(self.R850)-self.R660)/(np.square(self.R850)+self.R660)

    def NBR(self):
        # Normalized Burn Ratio (NBR)
        # https://doi.org/10.1080/10106049109354290
        # https://pubs.er.usgs.gov/publication/2002085
        return (self.R850-self.R2215)/(self.R850+self.R2215)

    def NBRT_1(self):
        # Normalized Burn Ratio Thermal 1 (NBRT1)
        # https://www.fs.usda.gov/treesearch/pubs/24608
        return (self.R850-self.R2215*(self.R1145/1000.))/(self.R850+self.R2215*(self.R1145/1000.))

    def NDBI(self):
        # Normalized Difference Built-Up Index (NDBI)
        # https://doi.org/10.1080/01431160304987
        return (self.R1650-self.R830)/(self.R1650+self.R830)

    def NDII(self):
        # Normalized Difference Infrared Index (NDII)
        # https://www.asprs.org/wp-content/uploads/pers/1983journal/jan/1983_jan_77-83.pdf
        return (self.R819-self.R1649)/(self.R819+self.R1649)

    def NDLI(self):
        # Normalized Difference Lignin Index (NDLI)
        # https://doi.org/10.1016/S0034-4257(02)00011-1
        # https://doi.org/10.1016/0034-4257(95)00234-0
        # https://doi.org/10.2307/1936780
        return (np.log(1./self.R1754)-np.log(1./self.R1680))/(np.log(1./self.R1754)+np.log(1./self.R1680))

    def NDMI(self):
        # Normalized Difference Mud Index (NDMI)
        # https://doi.org/10.1117/1.OE.51.11.111719
        return (self.R795-self.R990)/(self.R795+self.R990)

    def NDNI(self):
        # Normalized Difference Nitrogen Index (NDNI)
        # https://doi.org/10.1016/S0034-4257(02)00011-1
        # https://doi.org/10.1016/0034-4257(95)00234-0
        return (np.log(1./self.R1510)-np.log(1./self.R1680))/(np.log(1./self.R1510)+np.log(1./self.R1680))

    def NDSI(self):
        # Normalized Difference Snow Index (NDSI)
        # https://doi.org/10.1016/0034-4257(95)00137-P
        # https://doi.org/10.1016/j.rse.2003.10.016
        return (self.R550-self.R1650)/(self.R550+self.R1650)

    def NDVI(self):
        # Normalized Difference Vegetation Index (NDVI)
        # https://ntrs.nasa.gov/citations/19740022614
        return (self.R850-self.R660)/(self.R850+self.R660)

    def NDWI(self):
        # Normalized Difference Water Index (NDWI)
        # https://doi.org/10.1016/S0034-4257(96)00067-3
        # https://doi.org/10.1016/j.rse.2003.10.021
        return (self.R857-self.R1241)/(self.R857+self.R1241)

    def NMDI(self):
        # Normalized Multi-band Drought Index (NMDI)
        # https://doi.org/10.1029/2007GL031021
        # https://doi.org/10.1016/j.agrformet.2008.06.005
        return (self.R860-(self.R1640-self.R2130))/(self.R860+(self.R1640-self.R2130))

    def OSAVI(self):
        # Optimized Soil Adjusted Vegetation Index (OSAVI)
        # https://doi.org/10.1016/0034-4257(95)00186-7
        return (self.R850-self.R660)/(self.R850+self.R660+0.16)

    def PRI(self):
        # Photochemical Reflectance Index (PRI)
        # https://doi.org/10.1111/j.1469-8137.1995.tb03064.x
        # https://link.springer.com/article/10.1007/s004420050337
        return (self.R531-self.R570)/(self.R531+self.R570)

    def PSRI(self):
        # Plant Senescence Reflectance Index (PSRI)
        # https://doi.org/10.1034/j.1399-3054.1999.106119.x
        return (self.R680-self.R500)/self.R750

    def RENDVI(self):
        # Red Edge Normalized Difference Vegetation Index (RENDVI)
        # https://doi.org/10.1016/S0176-1617(11)81633-0
        # https://doi.org/10.1016/S0034-4257(02)00010-X
        return (self.R750-self.R705)/(self.R750+self.R705)

    def RDVI(self):
        # Renormalized Difference Vegetation Index (RDVI)
        # https://doi.org/10.1016/0034-4257(94)00114-3
        return (self.R850-self.R660)/(np.sqrt(self.R850+self.R660))

    def SR_1(self):
        # Simple Ratio (SR_1)
        # https://doi.org/10.2134/agronj1968.00021962006000060016x
        return self.R850/self.R660

    def SAVI(self):
        # Simple Ratio (SR_1)
        # https://doi.org/10.2134/agronj1968.00021962006000060016x
        return (1.5*(self.R850-self.R660))/(self.R850+self.R660+0.5)

    def SIPI(self):
        # Structure Insensitive Pigment Index (SIPI)
        # https://publons.com/publon/483937/
        return (self.R800-self.R445)/(self.R800-self.R680)

    def TCARI(self):
        # Transformed Chlorophyll Absorption Reflectance Index (TCARI)
        # https://doi.org/10.1016/j.rse.2003.12.013
        return 3.*((self.R700-self.R670)-0.2*(self.R700-self.R550)*(self.R700/self.R670))

    def TDVI(self):
        # Transformed Difference Vegetation Index (TDVI)
        # https://doi.org/10.1109/IGARSS.2002.1026867
        return 1.5*((self.R850-self.R660)/(np.sqrt(np.square(self.R850)+self.R660+0.5)))

    def TVI(self):
        # Triangular Vegetation Index (TVI)
        # https://doi.org/10.1016/S0034-4257(00)00197-8
        return (120.*(self.R750-self.R550)-200.*(self.R670-self.R550))/2.

    def VARI(self):
        # Visible Atmospherically Resistant Index (VARI)
        # https://doi.org/10.1080/01431160110107806
        return (self.R550-self.R660)/(self.R550+self.R660-self.R475)

    def VREI_1(self):
        # Vogelmann Red Edge Index 1 (VREI1)
        # https://doi.org/10.1080/01431169308953986
        return self.R740/self.R720

    def VREI_2(self):
        # Vogelmann Red Edge Index 2 (VREI2)
        # https://doi.org/10.1080/01431169308953986
        return (self.R734-self.R747)/(self.R715-self.R726)

    def WBI(self):
        # Water Band Index (WBI)
        # https://doi.org/10.1080/01431169308954010
        # https://www.researchgate.net/publication/260000104_Mapping_crop_water_stress_issues_of_scale_in_the_detection_of_plant_water_status_using_hyperspectral_indices
        return self.R970/self.R900

    def WDRVI(self):
        # Wide Dynamic Range Vegetation Index (WDRVI)
        # https://doi.org/10.1078/0176-1617-01176
        # https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1264&context=natrespapers
        return (0.2*self.R850-self.R660)/(0.2*self.R850+self.R660)

    def save_raster(self, feature, out_path):
        # Get and update profile
        profile = self.src.profile
        profile.update(count=1, driver='GTiff', dtype='float32', nodata=0)

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(feature, 1)


    def calculate(self,
                  out_dir,
                  features='all'):
        """
        Calculate the given features.

        Parameters:
        -----------

        out_dir: str
            Path of the directory where the calculated features will be saved.
        
        features: list
            List of features that needs to be calculated. Allowed list of features are:
            'ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI', 'EVI', 'GEMI', 'GARI',
            'GCI', 'GDVI', 'GLI', 'GNDVI', 'GOSAVI', 'GRVI', 'GSAVI', 'GVI', 'IPVI',
            'LCAI', 'MCARI', 'MNLI', 'MNDWI', 'MRENDVI', 'MRESR', 'MSR', 'MSAVI_2', 
            'MTVI', 'MSI', 'NLI', 'NBR', 'NBRT_1', 'NDBI', 'NDII', 'NDLI', 'NDMI', 
            'NDNI', 'NDSI', 'NDVI', 'NDWI', 'NMDI', 'OSAVI', 'PRI', 'PSRI', 'RENDVI', 
            'RDVI', 'SR_1', 'SAVI', 'SIPI', 'TCARI', 'TDVI', 'TVI', 'VARI', 'VREI_1', 
            'VREI_2', 'WBI', 'WDRVI'

        Returns:
        --------
        None
        """

        if features == 'all':
            features = utils.get_all_feature_names()
            print(f'Calculating all features')
        elif isinstance(features, list):
            print(f'Calculating {len(features)} features')
        else:
            raise ValueError("Invalid input in features. Only support list of featues or 'all'.")

        for feature in tqdm(features):
            out_path = os.path.join(out_dir, feature+'.tif')
            method = getattr(self, feature)
            try:
                feature = method()
                feature.reshape(1, feature.shape[0], feature.shape[0])
                utils.save_raster(self.src, feature, out_path,
                                  driver='GTiff', dtype='float32', nodata=0, count=1)
            except:
                warnings.warn(f"{feature} calculation failed.")
                pass


def extract_values_by_points(
        image_path,
        shape_path,
        unique_id):
    """
    Extract value by a point shapefile.

    The function will extract the pixel values underneat each point in a given point
    shapefile.

    Parameters:
    -----------
    image_path: str or list
        Path of the image.
    
    shape_path: str
        Path of the shapefile.
            
    unique_id: str
        A unique column in the shapefile which will be retained as the id.

    Returns:
    --------
    pixel_values: pandas dataframe
        A pandas dataframe containing all the values per id.
    """

    # Open files
    src = rasterio.open(image_path)
    img = src.read(1)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Point') | (geoms == 'MultiPoint')):
        pass
    else:
        raise ValueError("The shapefile must be either Point or MultiPoint.")

    # Image names
    image_name = os.path.basename(image_path).split('.')[0]

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)

    # Check the type of geometry in shape
    
    pixel_values = {}
    for i, point in enumerate(shape['geometry']):
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = src.index(x, y)
        value = img[row, col]
        pixel_values[shape.loc[i, f'{unique_id}']] = [value]
    pixel_values = pd.DataFrame(pixel_values).T
    pixel_values.columns = [image_name]

    src.close()
    
    return pixel_values


def batch_extract_values_by_points(
        image_paths, 
        *args):
    """
    Btach extract value by either point from a set of images.

    Point extraction only results the corresponding pixel value underneath that point.
    The polygon extraction allows statistical values as output within each polygon feature
    for that raster.

    Parameters:
    -----------
    image_paths: list
        List of image paths.
    
    shape_path: str
        Path of the shapefile.
            
    unique_id: str
        A unique column in the shapefile which will be retained as the id.

    Returns:
    --------
    stat: pandas dataframe
        A pandas dataframe containing all the values per id.
    """
    pixel_values_df = []
    for image_path in tqdm(image_paths):
        pixel_values = extract_values_by_points(image_path, *args)
        pixel_values_df.append(pixel_values)
    pixel_values_df = pd.concat(pixel_values_df, axis=1)

    return pixel_values_df



def extract_values_by_polygons(
        image_path,
        shape_path,
        unique_id,
        statistics,
        prefix=None):
    """
    Extract value by a polygon shapefile. Similar to zonal statistics.

    Point extraction only results the corresponding pixel value underneath that point.
    The polygon extraction allows statistical values as output within each polygon feature
    for that raster.

    Parameters:
    -----------

    image_path: str
        Path of the image.
    
    shape_path: str
        Path of the shapefile.
            
    unique_id: str
        A unique column in the shapefile which will be retained as the id.
    
    statistics: str or list
        List of statistics to be calculated if shape is polygon.
        Accepted statsitcs are either 'all' or a list containing follwoing statistics:
        'mean', 'median', 'mode', 'sum', 'min', 'max', 'std', 'range', 'iqr', 'unique'
        If only one statistic to be calculated, that should be inside a list. For example,
        if only 'mean' is to be calculated, it should be given as ['mean'].

    prefix: str, optional
        If predix is given, then the prefix will be used in front of the statistics name
        in the final dataframe column.

    Returns:
    --------
    stats: pandas dataframe
        A pandas dataframe containing all the statistics values per id.
    """

    # Open files
    src = rasterio.open(image_path)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Polygon') | (geoms == 'MultiPolygon')):
        pass
    else:
        raise ValueError("The shapefile must be either Polygon or MultiPolygon.")

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)
    
    stats = {}

    for i, polygon in enumerate(shape['geometry']):
        mask_img, _ = mask(src, [polygon], nodata=np.nan, crop=True)  
        mask_img = mask_img.reshape(-1)

        if statistics == 'all':
            statistics = ['mean', 'median', 'mode', 'sum', 'min', 'max', 'std', 'range',
                          'iqr', 'unique']

        # Check if all values are nan
        if np.isnan(mask_img).all():
            stats_values = [np.nan]*len(statistics)
        else:
            mask_img = mask_img[~np.isnan(mask_img)]
            stats_values = []
            if 'mean' in statistics:
                stats_values.append(np.mean(mask_img))
            if 'median' in statistics:
                stats_values.append(np.median(mask_img))
            if 'mode' in statistics:
                stats_values.append(np.bincount(mask_img).argmax())
            if 'sum' in statistics:
                stats_values.append(np.sum(mask_img))
            if 'min' in statistics:
                stats_values.append(np.min(mask_img))
            if 'max' in statistics:
                stats_values.append(np.max(mask_img))
            if 'std' in statistics:
                stats_values.append(np.std(mask_img))
            if 'range' in statistics:
                stats_values.append(np.max(mask_img)-np.min(mask_img))
            if 'iqr' in statistics:
                stats_values.append(np.subtract(*np.percentile(mask_img, [75, 25])))
            if 'unique' in statistics:
                stats_values.append(np.unique(mask_img).shape[0])

        stats[shape.loc[i, f'{unique_id}']] = stats_values

    stats = pd.DataFrame(stats).T
    if prefix is None:
        stats.columns = statistics
    else:
        stats.columns = [f'{str(prefix)}_{statistic}' for statistic in statistics]

    src.close()
    
    return stats


def batch_extract_values_by_polygons(
        image_paths,
        *args):
    """
    Batch extract value by a polygon shapefile from a given image paths.
    Similar to zonal statistics.

    Parameters:
    -----------
    image_paths: list
        List of image paths.
    
    shape_path: str
        Path of the shapefile.
            
    unique_id: str
        A unique column in the shapefile which will be retained as the id.
    
    statistics: str or list
        List of statistics to be calculated if shape is polygon.
        Accepted statsitcs are either 'all' or a list containing follwoing statistics:
        'mean', 'median', 'mode', 'sum', 'min', 'max', 'std', 'range', 'iqr', 'unique'
        If only one statistic to be calculated, that should be inside a list. For example,
        if only 'mean' is to be calculated, it should be given as ['mean'].

    Returns:
    --------
    stats: pandas dataframe
        A pandas dataframe containing all the statistics values per id. Each column name 
        will be made through automatically adding a prefix (which is the filename of each
        image) and the corresponding statistics.
    """
    stats_df = []
    for image_path in tqdm(image_paths):
        prefix = os.path.basename(image_path).split('.')[0]
        stats = extract_values_by_polygons(image_path, prefix=prefix, *args)
        stats_df.append(stats)
    stats_df = pd.concat(stats_df, axis=1)

    return stats_df



def clip_raster_by_polygons(
        image_path,
        shape_path,
        unique_id,
        out_dir,
        out_type='numpy'):
    """
    Clip a raster image by a polygon shapefile.

    Based on the geometry of each polygon, the function will clip the images and save it
    in a given directory with a unique name.

    Parameters:
    -----------
    image_path: str
        Path of the image.
    
    shape_path: str
        Path of the shapefile.
            
    unique_id: str
        A unique column in the shapefile which will be retained as the id.

    out_dir: str
        Path of the directory where the clipped images will be saved.

    out_type: str
        The type of output data. It can be either 'numpy' or 'tif'.

    Returns:
    --------
    None
    """
    # Open files
    src = rasterio.open(image_path)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Polygon') | (geoms == 'MultiPolygon')):
        pass
    else:
        raise ValueError("The shapefile must be either Polygon or MultiPolygon.")

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)

    for i, polygon in enumerate(shape['geometry']):
        mask_img, transform = mask(src, [polygon], nodata=0, crop=True)
        if out_type == 'numpy':
            out_path = os.path.join(out_dir, str(shape.loc[i, f'{unique_id}'])+'.npy')
            mask_img = np.moveaxis(mask_img, 0, 2)
            np.save(out_path, mask_img)
        elif out_type == 'tif':
            out_path = os.path.join(out_dir, str(shape.loc[i, f'{unique_id}'])+'.tif')
            #print(src.profile)
            utils.save_raster(src, mask_img, out_path,
                              driver='GTiff', nodata=0,
                              width=mask_img.shape[2], height=mask_img.shape[1],
                              transform=transform)
    
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.pipeline import Pipeline
import sys
import logging
import sklearn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to system path for custom module imports
import sys
sys.path.append('../src/utils')
#from utils.preprocess import *
from preprocessing_pipeline import preprocessing_pipeline, preprocessing_pipeline_part2

# Set sklearn configuration
sklearn.set_config(display="diagram")

# Set pandas options for better visualization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Función para cargar datos
def cargar_datos(filepath):
    df = pd.read_excel(filepath)
    logging.info("Datos cargados con dimensiones iniciales: %s", df.shape)
    return df

# Load and preprocess data
def load_and_preprocess_data(file_path, columns_to_drop, default_values, columns_to_convert, replacements, group_mapping, columns_tests):
    df = cargar_datos(file_path)
    pipeline = preprocessing_pipeline(columns_to_drop, default_values, columns_to_convert, replacements, group_mapping, columns_tests)
    df = pipeline.fit_transform(df)
    return df

# Main function to run the data processing pipeline
def main():
    columns_to_drop = [
        'Codigo','Sujeto', 'Fecha Nacimiento', 'Fecha de exploración', 'Motivo de consulta',
        'NºPatologia', 'Patologia', 'Enf Sistemicas 1', 'Enfermedades Sistemicas 2', 'Sintomas narrados',
        'Diagnóstico 1', 'Dagnóstico 2', 'IQ', 'Fecha IQ', 'Resultado IQ',
        'Tabaco', 'Alcohol', 'Medicación', 'Fármacos', 'Alergias', 'Alergenos',
        'Min Tilt_walk', 'Max Tilt_walk', 'Min Obliquity_walk', 'Max Obliquity_walk',
        'Min Rotation_walk', 'Max Rotation_walk', 'Max Tilt Rate_walk', 'Max Obliquity Rate_walk',
        'Max Rotation Rate_walk', 'Medio Lateral Gs2_walk', 'Elevation Gain_walk',
        'Time (Max Swing->FS)_walk', 'Time (FS->MPV)_walk', 'Time (MPV->MP)_walk', 'Time (MP->TO)_walk',
        'Time (Min Swing->Max Swing)_walk', 'Time (TO->Min Swing)_walk', 'Medio Lateral Gs_walk',
        'Braking Gs (Amplitude)_walk', 'Impact Gs (Amplitude)_walk', 'Vertical Speed_walk',
        'Yaw Excursion_walk', 'Swing Excursion_walk', 'Yaw Excursion (MP->TO)_walk', 'Yaw Excursion (Swing)_walk',
        'Max Stance Velocity (FS->MP)_walk', 'Max Stance Velocity (MP->TO)_walk',
        'Braking Gs2_walk', 'Impact Gs2_walk',
        'Min Tilt_run', 'Max Tilt_run', 'Min Obliquity_run', 'Max Obliquity_run',
        'Min Rotation_run', 'Max Rotation_run', 'Max Tilt Rate_run', 'Max Obliquity Rate_run',
        'Max Rotation Rate_run', 'Medio Lateral Gs2_run', 'Elevation Gain_run',
        'Time (Max Swing->FS)_run', 'Time (FS->MPV)_run', 'Time (MPV->MP)_run', 'Time (MP->TO)_run',
        'Time (Min Swing->Max Swing)_run', 'Time (TO->Min Swing)_run', 'Medio Lateral Gs_run',
        'Braking Gs (Amplitude)_run', 'Impact Gs (Amplitude)_run', 'Vertical Speed_run',
        'Yaw Excursion_run', 'Swing Excursion_run', 'Yaw Excursion (MP->TO)_run', 'Yaw Excursion (Swing)_run',
        'Max Stance Velocity (FS->MP)_run', 'Max Stance Velocity (MP->TO)_run',
        'Braking Gs2_run', 'Impact Gs2_run',
        'Rot Ext Cadera Izquierda', 'Rot Ext Cadera Derecha', 'Rot Int Cadera Derecha',
        'Rot Int Cadera Izquierda', 'Dismetría', 'POPLITEO CL Neutra', 'Valgo-Varo_I',
        'Pelvis AV', 'Rot_Pelvis_Izda', 'EIAS_Down_Izda', 'Hiperlordosis cervical',
        'Hipercifosis torácica', 'Hiperlordosis lumbar', 'Antepulsión torax',
        'Antepulsión pelvis', 'Retropulsion Pelvis', 'Rectificación lumbar',
        'Retropulsion Torax', 'Dorso plano', 'Rectificación Cervical', 'GAZE Elevada',
        'GAZE Disminuida', 'Retrognatismo', 'Prognatismo', 'Inclinación izquierda cabeza',
        'Escoliosis Cervical Izquierda', 'Escoliosis Cervical Derecha', 'Escoliosis Dorsal Derecha',
        'Escoliosis Dorsal Izquierda', 'Escoliosis Lumbar Izquierda', 'Escoliosis Lumbar Derecha',
        'Escoliosis Rotación toracica derecha', 'Escoliosis Rotación toracica izquierda',
        'Rotación izquierda cabeza', 'Rotación derecha cabeza', 'Rotación derecha torax',
        'Rotación izquierda torax', 'Antepulsion Cabeza', 'Inclinación derecha cabeza',
        'Lateropulsión cabeza derecha', 'Lateropulsión cabeza izquierda', 'Lateropulsión torax izquierda',
        'Lateropulsión pelvis  derecha', 'Lateropulsión pelvis  izquierda', 'EIPS_Down_Izda', 'EIPS_Down_Dcha',
        'EIAS_Down_Dcha', 'Lateropulsión torax derecha', 'Descenso hombro izquierdo',
        'Elevación Hombro Izquierdo', 'Elevación hombro derecho', 'Descenso hombro derecho',
        'Retropulsión Cabeza', 'Impresión Diagnóstica de función', 'Pruebas complementarias',
        'Escaneo 25%', 'Escaneo 50%', 'Escaneo 75%', 'Escaneo 100%', 'TPU', 'TPU Run', 'PA12 Anterocapital',
        'PA12 Retro', 'Pelvis Rot Dcha', 'Videoconferencia con informe', 'Telemetría', 'Solo informe',
        'entrega y revisiones', 'Iliaco Izquierdo AV', 'Iliaco Izquierdo RV', 'Iliaco Dcho AV',
        'Iliaco Dcho RV', 'EIAS_Up_Izda', 'EIAS_Up_Dcha', 'EIPS_Up_I', 'EIPS_Up_Dcha', 'Calzado',
        'Escaneo 0%', 'Elementos en taller', 'Fisioterapia y ejercicios', 'Llamar cuando lleguen plantillas',
        'Elementos en Diseño', 'Iliaco Izquierdo Normal', 'Iliaco Dcho Normal', 'Lunge Izq', 'Lunge Dcho',
        'TF Normal', 'TT Normal', 'Arco normal', 'AP_Neutro',  'RP_Neutro',
        'Cavo Col medial', 'Cavo col Lateral', 'MTF1 Normal', 'Col lat corta', 'Col lat larga', 
        'Genu normal', 'Pierna Corta', 'AP Normal ABD-AD', 'Genu neutro',
        'PNCA AP Neutro', 'PNCA RP Neutro', 'Jack R aumentada', 'Jack R disminuida',
        'Pron max Normal', 'Rotula descendida','Index plus-minus', 'Index Plus',
        'AP_Varo', 'AP_Valgo', 'RP_Varo', 'RP_Valgo', 'M5 Dfx', 'M5 PFx', 'AP Adducto', 'AP Abducto',
        'PIe cavo posterior', 'PIe cavo anterior', 'Pie griego', 'Pie cuadrado', 'Pie egipcio',
        'FPI_1_I', 'FPI_1_D', 'FPI_2_I', 'FPI_2_D', 'FPI_3_I', 'FPI_3_D', 
        'FPI_4_I', 'FPI_4_D', 'FPI_5_I', 'FPI_5_D', 'FPI_6_I', 'FPI_6_D',
        'Flight Time_walk', 'Contact Time_walk', 'Horizontal GRF Rate_walk', 'Swing Force Rate_walk',
        'VO<sub>2</sub>_walk', 'Flight Time_run', 'Contact Time_run', 'Horizontal GRF Rate_run',
        'Swing Force Rate_run', 'VO<sub>2</sub>_run', 'Vertical Oscillation_walk'
    ]
    
    columns_to_impute = [
        'tfi', 'tfe', 'tti', 'tte', 'arco aplanado', 'arco elevado', 'm1 dfx', 'm1 pfx', 'm5 hipermovil', 
        'arco transverso disminuido', 'arco transverso aumentado', 'm1 hipermovil', 'hlf', 'hl', 'hr', 'hav', 
        'index minus', 'tibia vara proximal', 'tibia vara distal', 'rotula divergente', 'rotula convergente', 
        'rotula ascendida', 'genu valgo', 'genu varo', 'genu recurvatum', 'genu flexum', 'pnca ap varo', 
        'pnca ap valgo', 'pnca rp varo', 'pnca rp valgo', 't_hintermann', 'jack normal', 'jack no reconstruye', 
        'pronacion no disponible', '2heel raise', 'heel raise', 'thomas psoas', 'thomas rf', 'thomas tfl', 
        'ober', 'lunge', 'ober friccion', 'popliteo'
    ]

    default_values = {
        'Negativo': ['thomas psoas', 'thomas rf', 'thomas tfl', 'ober', 'lunge', 'ober friccion', 'popliteo'],
        'No': [col for col in columns_to_impute if col not in ['thomas psoas', 'thomas rf', 'thomas tfl', 'ober', 'lunge', 'ober friccion', 'popliteo']]
    }

    columns_to_convert = ['altura', 'peso', 'num calzado']
    
    replacements = {'b + r': 'b', 'b + i': 'b', 'ninguno': 'no especificado'}

    group_mapping = {
        'pie_plantar distal_i': 'pie_plantar distal_i',
        'pie_plantar distal_d': 'pie_plantar distal_d',
        'pie_plantar distal_b': 'pie_plantar distal_b',
        'pie_plantar proximal_i': 'pie_plantar proximal_i',
        'pie_plantar_i': 'pie_plantar proximal_i',
        'pie_plantar proximal_d': 'pie_plantar proximal_d',
        'pie_plantar_d': 'pie_plantar proximal_d',
        'pie_plantar proximal_b': 'pie_plantar proximal_b',
        'pie_plantar_b': 'pie_plantar proximal_b',
        'pie_medial_b': 'pie-tobillo_medial_b',
        'tobillo_medial_b': 'pie-tobillo_medial_b',
        'tobillo_anteromedial_i': 'pie-tobillo_medial_i',
        'pie_medial_i': 'pie-tobillo_medial_i',
        'tobillo_medial_i': 'pie-tobillo_medial_i',
        'pie_anteromedial_i': 'pie-tobillo_medial_i',
        'pie_medial_d': 'pie-tobillo_medial_d',
        'tobillo_medial_d': 'pie-tobillo_medial_d',
        'tobillo_lateral_d': 'pie-tobillo_lateral_d',
        'tobillo_anterolateral_d': 'pie-tobillo_lateral_d',
        'pie_lateral_d': 'pie-tobillo_lateral_d',
        'tobillo_mediolateral_d': 'pie-tobillo_lateral_d',
        'tobillo_lateral_i': 'pie-tobillo_lateral_i',
        'pie_lateral_i': 'pie-tobillo_lateral_i',
        'tobillo_anterolateral_i': 'pie-tobillo_lateral_i',
        'tobillo_lateral_b': 'pie-tobillo_lateral_b',
        'pie_lateral_b': 'pie-tobillo_lateral_b',
        'tobillo_mediolateral_b': 'pie-tobillo_lateral_b',
        'pie_mediolateral_b': 'pie-tobillo_lateral_b',
        'tobillo_anterior_d': 'pie-tobillo_anterior',
        'pie_dorsal proximal_d': 'pie-tobillo_anterior',
        'pie_no especificado_d': 'pie-tobillo_anterior',
        'pie_dorsal distal_d': 'pie-tobillo_anterior',
        'pie_dorsal distal_i': 'pie-tobillo_anterior',
        'tobillo_anterior_i': 'pie-tobillo_anterior',
        'pie_dorsal distal_b': 'pie-tobillo_anterior',
        'tobillo_anteroposterior_d': 'tobillo_posterior_d',
        'tobillo_posterior_d': 'tobillo_posterior_d',
        'tobillo_posterior_i': 'tobillo_posterior_i',
        'tobillo_anteroposterior_i': 'tobillo_posterior_i',
        'tobillo_posterior_b': 'tobillo_posterior_b',
        'rodilla_anterior_b': 'rodilla_anterior_b',
        'rodilla_anterolateral_b': 'rodilla_anterior_b',
        'rodilla_anterior_i': 'rodilla_anterior_i',
        'rodilla_anteromedial_i': 'rodilla_anterior_i',
        'rodilla_anterolateral_i': 'rodilla_anterior_i',
        'rodilla_anteroposterior_i': 'rodilla_anterior_i',
        'rodilla_anterior_d': 'rodilla_anterior_d',
        'rodilla_anteromedial_d': 'rodilla_anterior_d',
        'rodilla_anteroposterior_d': 'rodilla_anterior_d',
        'rodilla_anterolateral_d': 'rodilla_anterior_d',
        'rodilla_lateral_i': 'rodilla_lateral_i',
        'rodilla_lateral_d': 'rodilla_lateral_d',
        'rodilla_lateral_b': 'rodilla_lateral_b',
        'rodilla_mediolateral_b': 'rodilla_lateral_b',
        'rodilla_medial_i': 'rodilla_medial_i',
        'rodilla_medial_d': 'rodilla_medial_d',
        'rodilla_medial_b': 'rodilla_medial_b',
        'pierna_anterior_b': 'pierna_anteromedial',
        'pierna_anteromedial_b': 'pierna_anteromedial',
        'pierna_medial_b': 'pierna_anteromedial',
        'pierna_anterior_d': 'pierna_anteromedial',
        'pierna_medial_d': 'pierna_anteromedial',
        'pierna_anteroposterior_i': 'pierna_anteromedial',
        'pierna_medial_i': 'pierna_anteromedial',
        'pierna_anterior_i': 'pierna_anteromedial',
        'pierna_lateral_i': 'pierna_lateral',
        'pierna_anterolateral_i': 'pierna_lateral',
        'pierna_lateral_d': 'pie-tobillo_lateral_d',
        'pierna_posterior_b': 'pierna_posterior_b',
        'pierna_posteromedial_b': 'pierna_posterior_b',
        'rodilla_posterior_b': 'pierna_posterior_b',
        'rodilla_posteromedial_b': 'pierna_posterior_b',
        'pierna_posterior_d': 'pierna_posterior_d',
        'rodilla_posterior_d': 'pierna_posterior_d',
        'rodilla_posterolateral_d': 'pierna_posterior_d',
        'pierna_posterior_i': 'pierna_posterior_i',
        'rodilla_posterior_i': 'pierna_posterior_i',
        'rodilla_posterolateral_i': 'pierna_posterior_i',
        'cadera_lateral_d': 'cadera_lateral_d',
        'muslo_lateral_d': 'cadera_lateral_d',
        'cadera_lateral_i': 'cadera_lateral',
        'cadera_lateral_b': 'cadera_lateral',
        'cadera_anterolateral_b': 'cadera_lateral',
        'espalda_lumbar_b': 'lumbar_b',
        'espalda_posterior_b': 'lumbar_b',
        'cadera_posterior_b': 'lumbar_b',
        'cadera_posterior_no especificado': 'lumbar_b',
        'espalda_lumbar_i': 'lumbar_i',
        'cadera_posterior_i': 'lumbar_i',
        'espalda_lumbar_d': 'lumbar_d',
        'cadera_posterior_d': 'lumbar_d',
        'sin afectacion_no especificado_no especificado': 'sin patologia',
        'muslo_posterior_d': 'otro',
        'muslo_posterior_i': 'otro',
        'muslo_posterior_b': 'otro',
        'cadera_anterior_b': 'otro',
        'cadera_anterior_i': 'otro',
        'cadera_anterior_d': 'otro',
        'muslo_anteroposterior_i': 'otro',
        'muslo_anterior_d': 'otro',
        'muslo_no especificado_d': 'otro',
        'sin afectacion_no especificado_b': 'sin patologia',
        'complejo_no especificado_complejo': 'otro',
        'complejo_no especificado_b': 'otro',
        'complejo_no especificado_i': 'otro',
        'cadera_medial_i': 'otro'
    }
    
    columns_tests = [
        'm1 hipermovil', 'thomas psoas', 'thomas rf', 'thomas tfl', 'ober',
        'arco aplanado', 'arco elevado', 'm1 dfx', 'm5 hipermovil',
        'arco transverso disminuido', 'm1 pfx', 'arco transverso aumentado',
        'hlf', 'hl', 'hr', 'hav', 'index minus', 'tfi', 'tfe', 'tti', 'tte',
        'ober friccion', 'popliteo', 'pnca ap varo', 'pnca ap valgo',
        'pnca rp varo', 'pnca rp valgo', 't_hintermann', 'jack normal',
        'jack no reconstruye', 'pronacion no disponible', '2heel raise',
        'heel raise', 'tibia vara proximal', 'tibia vara distal', 'rotula divergente',
        'rotula convergente', 'rotula ascendida', 'genu valgo', 'genu varo',
        'genu recurvatum', 'genu flexum', 'lunge'
    ]

    df = load_and_preprocess_data('../data/raw/BBDD_gen.xlsx', columns_to_drop, default_values, columns_to_convert, replacements, group_mapping, columns_tests)
    
    if df is not None:
        # Guardar el DataFrame en un archivo CSV
        df.to_csv('../data/processed/dataset_corredores.csv', sep=";", index=False, encoding='utf-8')

        columns_to_drop_after = [
            'actividad principal', 'marca 10k', 'especialidad', 'km/sem', 'calzado 1', 'calzado 2',
            'pace_run', 'velocidad_run', 'step rate_run', 'stride length_run', 'contact ratio_run',
            'power_run', 'shock_run', 'impact gs_run', 'braking gs_run', 'footstrike type_run',
            'pronation excursion_run', 'max pronation velocity_run', 'peak vertical grf_run',
            'stride angle_run', 'leg spring stiffness_run', 'flight ratio_run', 'step length_run',
            'vertical spring stiffness_run', 'vertical grf rate_run', 'total force rate_run',
            'pronation excursion (mp->to)_run', 'stance excursion (fs->mp)_run',
            'stance excursion (mp->to)_run', 'vertical oscillation_run'
        ]

        df.drop(columns_to_drop_after, axis=1, inplace=True)
        df['fpi_total_i'].fillna(df['fpi_total_i'].mode()[0], inplace=True)
        df['fpi_total_d'].fillna(df['fpi_total_d'].mode()[0], inplace=True)
        
        cols_runscribe_walk = [
            'pace_walk', 'velocidad_walk', 'step rate_walk', 'stride length_walk', 'flight ratio_walk',
            'power_walk', 'shock_walk', 'impact gs_walk', 'braking gs_walk', 'footstrike type_walk',
            'pronation excursion_walk', 'max pronation velocity_walk', 'peak vertical grf_walk',
            'contact ratio_walk', 'stride angle_walk', 'leg spring stiffness_walk', 'vertical spring stiffness_walk',
            'vertical grf rate_walk', 'total force rate_walk', 'pronation excursion (mp->to)_walk',
            'stance excursion (fs->mp)_walk', 'stance excursion (mp->to)_walk', 'step length_walk'
        ]

        pipeline_part2 = preprocessing_pipeline_part2(cols_runscribe_walk)
        df = pipeline_part2.fit_transform(df)

        if df is not None:
            logging.info("Total NaN Final: %s", df.isna().sum().sum())
            logging.info("Nº de registros después del filtrado: %s", len(df))
        else:
            logging.info("Error during the second part of preprocessing.")
    else:
        logging.info("Error during the first part of preprocessing.")

if __name__ == "__main__":
    main()


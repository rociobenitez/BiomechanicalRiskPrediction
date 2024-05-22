def convert_to_float(value):
    """
    Convierte una cadena en un número decimal,
    reemplazando las comas por puntos y eliminando los espacios.
    """
    try:
        cleaned_value = value.strip()
        return float(cleaned_value.replace(',', '.'))
    except ValueError:
        return f"El valor proporcionado '{value}' no es un número válido."

def process_and_validate_form_data(form_data):
    # Definir el orden de las columnas según el archivo CSV
    ordered_fields = [
        "edad", "sexo", "altura", "peso", "num_calzado", "articulacion",
        "localizacion", "lado", "pace_walk", "velocidad_walk", "step_rate_walk",
        "stride_length_walk", "shock_walk", "impact_gs_walk", "braking_gs_walk",
        "footstrike_type_walk", "pronation_excursion_walk", "contact_ratio_walk",
        "total_force_rate_walk", "step_length_walk", "pronation_excursion_mp_to_walk",
        "stance_excursion_fs_mp_walk","stance_excursion_mp_to_walk", "m1_hipermovil",
        "thomas_psoas", "thomas_rf", "thomas_tfl", "ober", "arco_aplanado",
        "arco_elevado", "m1_dfx", "m5_hipermovil", "arco_transverso_disminuido",
        "m1_pfx", "arco_transverso_aumentado", "hlf", "hl", "hr", "hav",
        "index_minus", "tfi", "tfe", "tti", "tte", "ober_friccion", "popliteo",
        "t_hintermann", "jack_normal", "jack_no_reconstruye", "pronacion_no_disponible",
        "heel_raise_double", "heel_raise", "fpi_total_i", "fpi_total_d",
        "tibia_vara_proximal", "tibia_vara_distal", "rotula_divergente",
        "rotula_convergente", "rotula_ascendida", "genu_valgo",
        "genu_varo", "genu_recurvatum", "genu_flexum", "lunge"
    ]

    validated_data = {}
    for field in ordered_fields:
        value = form_data.get(field, '0')  # Asume '0' como valor predeterminado si no se proporciona
        if field in ['articulacion', 'localizacion', 'lado']:
            validated_data[field] = value
        else:
            try:
                validated_data[field] = float(value) if value is not None else 0.0
            except ValueError:
                validated_data[field]  = 0.0  # Establecer a 0.0 si la conversión falla
    return validated_data

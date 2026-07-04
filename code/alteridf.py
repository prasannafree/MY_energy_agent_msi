import copy
from pathlib import Path

def alteridf(idf_object, zone_name: str, parameter_type: str, new_value: float, save_directory: str, new_filename: str = "modified_model.idf"):
    """Intercepts an active IDF object, creates an independent copy, modifies 
    either the peak 'people' count or 'lights' density for a targeted zone, 
    and writes out a brand new physical .idf file to disk.
    
    Args:
        idf_object: The active eppy/geomeppy baseline IDF instance.
        zone_name: The exact string name of the target thermal zone.
        parameter_type: String choice, either 'people' or 'lights'.
        new_value: The new absolute numeric value to assign (Count or W/m²).
        save_directory: Path to the folder where the new IDF file should be saved.
        new_filename: Name of the generated file (defaults to 'modified_model.idf').
        
    Returns:
        The new modified eppy IDF object instance so it can be passed directly to justcsv().
    """
    # 1. Create a deep copy to ensure the baseline model remains pristine
    modified_idf = copy.deepcopy(idf_object)
    
    param_choice = parameter_type.strip().lower()
    zone_target = zone_name.strip()
    modified_flag = False
    
    # =================================================================
    # BRANCH A: OCCUPANCY OVERWRITES
    # =================================================================
    if param_choice == 'people':
        people_objects = modified_idf.idfobjects['PEOPLE']
        for p in people_objects:
            assigned_zone = getattr(p, 'Zone_or_ZoneList_or_Space_or_SpaceList_Name', '')
            if assigned_zone.strip() == zone_target:
                old_val = p.Number_of_People
                p.Number_of_People = new_value
                print(f"🎯 PEOPLE MATCH: Overwrote peak occupancy from {old_val} to {new_value} bodies.")
                modified_flag = True
                break
                
    # =================================================================
    # BRANCH B: ARTIFICIAL LIGHTING OVERWRITES
    # =================================================================
    elif param_choice == 'lights':
        if 'LIGHTS' not in modified_idf.idfobjects:
            print("❌ Error: No LIGHTS definitions exist in this specific model archive.")
            return None
            
        light_objects = modified_idf.idfobjects['LIGHTS']
        for l in light_objects:
            assigned_zone = getattr(l, 'Zone_or_ZoneList_Name', '')
            if assigned_zone.strip() == zone_target:
                method = getattr(l, 'Design_Level_Calculation_Method', '').lower()
                
                # Check how the lighting input was defined structurally
                if 'watts/area' in method or 'powerperzonefloorarea' in method:
                    old_val = l.Watts_per_Zone_Floor_Area
                    l.Watts_per_Zone_Floor_Area = new_value
                    print(f"💡 LIGHTS MATCH: Overwrote lighting density from {old_val} to {new_value} W/m².")
                else:
                    old_val = l.Lighting_Level
                    l.Lighting_Level = new_value
                    print(f"💡 LIGHTS MATCH: Overwrote raw absolute load from {old_val} to {new_value} Watts.")
                
                modified_flag = True
                break
    else:
        print("❌ Error: Invalid parameter selection. Choose either 'people' or 'lights'.")
        return None

    if not modified_flag:
        print(f"⚠️ Warning: Could not find any active {parameter_type.upper()} objects mapped to zone: '{zone_target}'")
        return None
        
    # =================================================================
    # SAVE PIPELINE: Save out the new physical IDF file to disk
    # =================================================================
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    final_file_path = save_path / new_filename
    
    # Save using eppy's built-in file writing protocol
    modified_idf.saveas(str(final_file_path))
    print(f"💾 File Success: Generated new altered IDF configuration at:\n   {final_file_path}")
    
    return modified_idf

def alteridf_global(idf_object, parameter_type: str, multiplier: float, save_directory: str, new_filename: str = "global_modified_model.idf"):
    """Intercepts an active IDF object, creates an independent copy, scales 
    the parameter value for ALL zones by a specified multiplier, 
    and writes out a brand new physical .idf file to disk.
    
    Args:
        idf_object: The active eppy/geomeppy baseline IDF instance.
        parameter_type: String choice, either 'people' or 'lights'.
        multiplier: The scaling factor (e.g., 1.50 for +50%, 0.50 for -50%).
        save_directory: Path to the folder where the new IDF file should be saved.
        new_filename: Name of the generated file.
        
    Returns:
        The new modified eppy IDF object instance.
    """
    # 1. Create a deep copy to keep the baseline model pristine
    modified_idf = copy.deepcopy(idf_object)
    
    param_choice = parameter_type.strip().lower()
    modified_count = 0
    
    # =================================================================
    # BRANCH A: GLOBAL OCCUPANCY SCALING (+50%)
    # =================================================================
    if param_choice == 'people':
        people_objects = modified_idf.idfobjects['PEOPLE']
        print(f"👥 Found {len(people_objects)} total PEOPLE objects. Scaling all entries...")
        
        for p in people_objects:
            old_val = p.Number_of_People
            # Apply the multiplier globally
            p.Number_of_People = old_val * multiplier
            modified_count += 1
            
        print(f"📊 Global update complete. Scaled {modified_count} occupancy zones by x{multiplier}.")

    # =================================================================
    # BRANCH B: GLOBAL ARTIFICIAL LIGHTING SCALING
    # =================================================================
    elif param_choice == 'lights':
        if 'LIGHTS' not in modified_idf.idfobjects:
            print("❌ Error: No LIGHTS definitions exist in this specific model archive.")
            return None
            
        light_objects = modified_idf.idfobjects['LIGHTS']
        print(f"💡 Found {len(light_objects)} total LIGHTS objects. Scaling all entries...")
        
        for l in light_objects:
            method = getattr(l, 'Design_Level_Calculation_Method', '').lower()
            
            # Scale whichever input metric the specific light object is actively using
            if 'watts/area' in method or 'powerperzonefloorarea' in method:
                old_val = l.Watts_per_Zone_Floor_Area
                l.Watts_per_Zone_Floor_Area = old_val * multiplier
            else:
                old_val = l.Lighting_Level
                l.Lighting_Level = old_val * multiplier
            modified_count += 1
            
        print(f"📊 Global update complete. Scaled {modified_count} lighting systems by x{multiplier}.")
        
    else:
        print("❌ Error: Invalid parameter selection. Choose either 'people' or 'lights'.")
        return None

    if modified_count == 0:
        print(f"⚠️ Warning: No objects modified for parameter type: '{parameter_type.upper()}'")
        return None
        
    # =================================================================
    # SAVE PIPELINE: Save the new physical IDF file to disk
    # =================================================================
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    final_file_path = save_path / new_filename
    
    modified_idf.saveas(str(final_file_path))
    print(f"💾 File Success: Generated global modification archive at:\n   {final_file_path}")
    
    return modified_idf

def altselect(idf_object, selected_zones: list, multiplier: float, save_directory: str, new_filename: str = "modified_model.idf"):
    """Intercepts an active IDF object, creates an independent copy, modifies 
    either the peak 'people' count or 'lights' density for a SPECIFIC ARRAY of targeted zones
    using a multiplier, and writes out a brand new physical .idf file to disk.
    
    Args:
        idf_object: The active eppy/geomeppy baseline IDF instance.
        selected_zones: A Python list/array containing the exact string names of target thermal zones.
        parameter_type: String choice, either 'people' or 'lights'.
        multiplier: The scaling factor (e.g., 1.50 for +50%, 0.70 for a 30% reduction).
        save_directory: Path to the folder where the new IDF file should be saved.
        new_filename: Name of the generated file (defaults to 'modified_model.idf').
        
    Returns:
        The new modified eppy IDF object instance.
    """
    # 1. Create a deep copy to ensure the baseline model remains pristine
    modified_idf = copy.deepcopy(idf_object)
    
    # Ensure all items in the comparison array are stripped cleanly of spaces
    target_zones_clean = [str(z).strip() for z in selected_zones]
    modified_count = 0
    
    # =================================================================
    people_objects = modified_idf.idfobjects['PEOPLE']
    for p in people_objects:
        assigned_zone = getattr(p, 'Zone_or_ZoneList_or_Space_or_SpaceList_Name', '').strip()
        
        # Check if this specific object belongs to our user-selected array
        if assigned_zone in target_zones_clean:
            old_val = float(p.Number_of_People)
            p.Number_of_People = old_val * multiplier
            print(f"🎯 PEOPLE MATCH [{assigned_zone}]: Scaled peak occupancy from {old_val} to {p.Number_of_People} (x{multiplier}).")
            modified_count += 1

    if modified_count == 0:
        print(f"⚠️ Warning: Found no active PEOPLE configurations inside the selected array zones.")
        return None
        
    # =================================================================
    # SAVE PIPELINE: Save out the new physical IDF file to disk
    # =================================================================
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    final_file_path = save_path / new_filename
    
    modified_idf.saveas(str(final_file_path))
    print(f"💾 File Success: Generated custom altered IDF configuration ({modified_count} items updated) at:\n   {final_file_path}")
    
    return modified_idf
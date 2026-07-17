import copy
from pathlib import Path

def altglo(idf_object, parameter_type: str, multiplier: float, save_directory: str, new_filename: str = "global_modified_model.idf"):
    """Intercepts an active IDF object, creates an independent copy, scales 
    the parameter value for ALL zones by a specified multiplier, 
    and writes out a brand new physical .idf file to disk.
    """
    # 1. Create a deep copy to keep the baseline model pristine
    modified_idf = copy.deepcopy(idf_object)
    
    param_choice = parameter_type.strip().lower()
    modified_count = 0

    if param_choice == 'people':
        people_objects = modified_idf.idfobjects['PEOPLE']
        print(f"👥 Found {len(people_objects)} total PEOPLE objects. Processing...")
        
        for p in people_objects:
            method_value = str(getattr(p,"Number_of_People_Calculation_Method","")).strip().lower()
            print(method_value)

            try:
                # Case 1: People
                if method_value == "people":
                    current = float(p.Number_of_People)
                    p.Number_of_People = current * multiplier
                    modified_count += 1

                elif method_value == "people/area":
                    current = float(p.People_per_Floor_Area)
                    p.People_per_Floor_Area = current * multiplier
                    modified_count += 1

                elif method_value == "area/person":
                    current = float(p.Floor_Area_per_Person)
                    p.Floor_Area_per_Person = current / multiplier
                    modified_count += 1

                else:
                    print("UNKNOWN METHOD:",repr(method_value),"OBJECT:",p.Name)
            except Exception as e:

                print(
                    f"❌ Failed to modify PEOPLE object "
                    f"'{p.Name}': {e}"
                )
        print(
            f"📊 Global update complete. "
            f"Scaled {modified_count} occupancy zones by x{multiplier}."
        )
    else:
        print("❌ Error: Invalid parameter selection. Choose either 'people' or 'lights'.")
        return None

    if modified_count == 0:
        print(f"⚠️ Warning: No objects modified for parameter type: '{parameter_type.upper()}'")
    
    print("METHOD =", method_value)

    for field in p.fieldnames:
        if (
            "number_of_people" in field.lower()
            or
            "people_per_zone_floor_area" in field.lower()
            or
            "zone_floor_area_per_person" in field.lower()
        ):
            print(field, getattr(p, field))
        
    # =================================================================
    # SAVE PIPELINE
    # =================================================================
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    final_file_path = save_path / new_filename
    
    modified_idf.saveas(str(final_file_path))
    print(f"💾 File Success: Generated global modification archive at:\n   {final_file_path}")
    
    return modified_idf
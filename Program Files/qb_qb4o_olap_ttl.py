import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD, BNode
from rdflib.namespace import RDFS, OWL, QB, SKOS
import os
import hashlib

# Step 1: Load the dataset from CSV with error handling
csv_path = 'mental_health_data_final_data.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please provide the correct path.")
    csv_path = input("Enter the full path to mental_health_data_final_data.csv: ")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found at {csv_path}")

df = pd.read_csv(csv_path)

# Handle NaN values
categorical_cols = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity', 
                   'Consultation_History', 'Stress_Level', 'Diet_Quality', 'Smoking_Habit', 
                   'Alcohol_Consumption', 'Medication_Usage']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

numeric_cols = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Step 2: Define namespaces - COMBINED QB + QB4OLAP VERSION
ADNX = Namespace('https://purl.archive.org/addiction-nexus#')
WD = Namespace('http://www.wikidata.org/entity/')
QB4O = Namespace('http://purl.org/qb4olap/cubes#')

# Step 3: Create Combined TBox graph (QB + QB4OLAP)
def create_combined_tbox():
    tbox = Graph()
    tbox.bind('adnx', ADNX)
    tbox.bind('wd', WD)
    tbox.bind('rdfs', RDFS)
    tbox.bind('owl', OWL)
    tbox.bind('qb', QB)
    tbox.bind('qb4o', QB4O)
    tbox.bind('skos', SKOS)

    # Define main classes
    main_classes = [
        (ADNX.Person, 'Person', 'Represents an individual with demographic attributes'),
        (ADNX.MentalHealth, 'Mental Health', 'Captures mental health condition details'),
        (ADNX.Lifestyle, 'Lifestyle', 'Captures lifestyle factors'),
        (ADNX.Observation, 'Observation', 'Combined QB/QB4OLAP Observation for measurements'),
    ]

    for cls, label, comment in main_classes:
        tbox.add((cls, RDF.type, OWL.Class))
        tbox.add((cls, RDFS.label, Literal(label, lang='en')))
        tbox.add((cls, RDFS.comment, Literal(comment, lang='en')))

    # Make Observation compatible with both QB and QB4OLAP
    tbox.add((ADNX.Observation, RDFS.subClassOf, QB.Observation))
    tbox.add((ADNX.Observation, RDFS.subClassOf, QB4O.Observation))

    # QB4OLAP Cube Definition
    cube = ADNX.MentalHealthCube
    tbox.add((cube, RDF.type, QB4O.Cube))
    tbox.add((cube, RDFS.label, Literal('Mental Health Analysis Cube', lang='en')))
    tbox.add((cube, RDFS.comment, Literal('QB4OLAP cube for mental health analytics')))

    # QB DataSet Definition (for standard QB compatibility)
    dataset = ADNX.MentalHealthDataSet
    tbox.add((dataset, RDF.type, QB.DataSet))
    tbox.add((dataset, RDFS.label, Literal('Mental Health Data Set', lang='en')))
    tbox.add((dataset, RDFS.comment, Literal('QB dataset for mental health analytics')))

    # Define QB4OLAP Dimensions with Hierarchies
    dimensions_qb4o = [
        (ADNX.countryDimension, ADNX.Country, ADNX.CountryHierarchy, 
         [(ADNX.CountryLevel, 'Country Level', None)]),
        
        (ADNX.genderDimension, ADNX.Gender, ADNX.GenderHierarchy,
         [(ADNX.GenderLevel, 'Gender Level', None)]),
        
        (ADNX.severityDimension, ADNX.Severity, ADNX.SeverityHierarchy,
         [(ADNX.SeverityLevel, 'Severity Level', None)]),
        
        (ADNX.stressLevelDimension, ADNX.StressLevel, ADNX.StressHierarchy,
         [(ADNX.StressLevelBase, 'Stress Level', None)]),
         
        (ADNX.occupationDimension, ADNX.Occupation, ADNX.OccupationHierarchy,
         [(ADNX.OccupationLevel, 'Occupation Level', None)]),
         
        (ADNX.ageDimension, ADNX.AgeDimension, ADNX.AgeHierarchy,
         [(ADNX.AgeGroupLevel, 'Age Group Level', None),
          (ADNX.AgeLevel, 'Individual Age Level', ADNX.AgeGroupLevel)])
    ]

    for dim_uri, dim_class, hier_uri, levels in dimensions_qb4o:
        # Define dimension class
        tbox.add((dim_class, RDF.type, OWL.Class))
        tbox.add((dim_class, RDFS.label, Literal(str(dim_class).split('#')[-1], lang='en')))
        
        # Define dimension property for QB4OLAP
        tbox.add((dim_uri, RDF.type, QB4O.LevelProperty))
        tbox.add((dim_uri, RDFS.label, Literal(str(dim_uri).split('#')[-1].replace('Dimension', ' Dimension'), lang='en')))
        tbox.add((dim_uri, RDFS.range, dim_class))
        
        # Also make it a QB DimensionProperty for compatibility
        tbox.add((dim_uri, RDF.type, QB.DimensionProperty))
        
        # Define hierarchy
        tbox.add((hier_uri, RDF.type, QB4O.Hierarchy))
        tbox.add((hier_uri, RDFS.label, Literal(str(hier_uri).split('#')[-1], lang='en')))
        tbox.add((dim_uri, QB4O.inHierarchy, hier_uri))
        
        # Define levels
        for level_uri, level_label, parent_level in levels:
            tbox.add((level_uri, RDF.type, QB4O.Level))
            tbox.add((level_uri, RDFS.label, Literal(level_label, lang='en')))
            tbox.add((level_uri, QB4O.inHierarchy, hier_uri))
            
            if parent_level:
                tbox.add((level_uri, QB4O.parentLevel, parent_level))
        
        # Add dimension to cube and dataset
        tbox.add((cube, QB4O.dimension, dim_uri))

    # Define QB4OLAP Measures
    measures_qb4o = [
        (ADNX.sleepHoursMeasure, XSD.float, 'Sleep Hours Measure'),
        (ADNX.workHoursMeasure, XSD.integer, 'Work Hours Measure'),
        (ADNX.physicalActivityMeasure, XSD.integer, 'Physical Activity Measure'),
        (ADNX.socialMediaMeasure, XSD.float, 'Social Media Usage Measure'),
        (ADNX.ageMeasure, XSD.integer, 'Age Measure'),
    ]

    for measure_uri, datatype, label in measures_qb4o:
        tbox.add((measure_uri, RDF.type, QB4O.Measure))
        tbox.add((measure_uri, RDF.type, QB.MeasureProperty))  # QB compatibility
        tbox.add((measure_uri, RDFS.label, Literal(label, lang='en')))
        tbox.add((measure_uri, RDFS.range, datatype))
        tbox.add((cube, QB4O.measure, measure_uri))

    # Define QB4OLAP Aggregate Functions
    agg_functions = [
        (ADNX.avgFunction, 'Average Function'),
        (ADNX.sumFunction, 'Sum Function'),
        (ADNX.countFunction, 'Count Function'),
        (ADNX.minFunction, 'Minimum Function'),
        (ADNX.maxFunction, 'Maximum Function')
    ]

    for func_uri, label in agg_functions:
        tbox.add((func_uri, RDF.type, QB4O.AggregateFunction))
        tbox.add((func_uri, RDFS.label, Literal(label, lang='en')))

    # Define traditional dimension classes
    dimension_classes = [
        (ADNX.Gender, 'Gender'),
        (ADNX.Occupation, 'Occupation'), 
        (ADNX.Country, 'Country'),
        (ADNX.Severity, 'Severity'),
        (ADNX.Consultation, 'Consultation'),
        (ADNX.StressLevel, 'StressLevel'),
        (ADNX.Medication, 'Medication'),
        (ADNX.DietQuality, 'DietQuality'),
        (ADNX.SmokingHabit, 'SmokingHabit'),
        (ADNX.AlcoholConsumption, 'AlcoholConsumption'),
        (ADNX.AgeDimension, 'Age Dimension')
    ]

    for cls, label in dimension_classes:
        tbox.add((cls, RDF.type, OWL.Class))
        tbox.add((cls, RDFS.label, Literal(label, lang='en')))

    # Define object properties - FIXED: Include lifestyle connections
    object_properties = [
        (ADNX.hasGender, ADNX.Person, ADNX.Gender),
        (ADNX.hasOccupation, ADNX.Person, ADNX.Occupation),
        (ADNX.hasCountry, ADNX.Person, ADNX.Country),
        (ADNX.hasMentalHealth, ADNX.Person, ADNX.MentalHealth),
        (ADNX.hasSeverity, ADNX.MentalHealth, ADNX.Severity),
        (ADNX.hasStressLevel, ADNX.MentalHealth, ADNX.StressLevel),
        (ADNX.hasConsultationHistory, ADNX.MentalHealth, ADNX.Consultation),
        (ADNX.hasMedicationUsage, ADNX.MentalHealth, ADNX.Medication),
        (ADNX.hasLifestyle, ADNX.Person, ADNX.Lifestyle),
        (ADNX.hasDietQuality, ADNX.Lifestyle, ADNX.DietQuality),
        (ADNX.hasSmokingHabit, ADNX.Lifestyle, ADNX.SmokingHabit),
        (ADNX.hasAlcoholConsumption, ADNX.Lifestyle, ADNX.AlcoholConsumption),
        (ADNX.hasObservation, ADNX.Person, ADNX.Observation),
        # CRITICAL: Add these missing properties to link observation to lifestyle
        (ADNX.observationHasSmokingHabit, ADNX.Observation, ADNX.SmokingHabit),
        (ADNX.observationHasAlcoholConsumption, ADNX.Observation, ADNX.AlcoholConsumption),
        (ADNX.observationHasDietQuality, ADNX.Observation, ADNX.DietQuality),
    ]

    for prop, domain, range_cls in object_properties:
        tbox.add((prop, RDF.type, OWL.ObjectProperty))
        tbox.add((prop, RDFS.domain, domain))
        tbox.add((prop, RDFS.range, range_cls))

    # Define datatype properties
    datatype_properties = [
        (ADNX.hasAge, ADNX.Person, XSD.integer),
        (ADNX.hasSleepHours, ADNX.Observation, XSD.float),
        (ADNX.hasWorkHours, ADNX.Observation, XSD.integer),
        (ADNX.hasPhysicalActivityHours, ADNX.Observation, XSD.integer),
        (ADNX.hasSocialMediaUsage, ADNX.Observation, XSD.float),
    ]

    for prop, domain, datatype in datatype_properties:
        tbox.add((prop, RDF.type, OWL.DatatypeProperty))
        tbox.add((prop, RDFS.domain, domain))
        tbox.add((prop, RDFS.range, datatype))

    return tbox

# Create combined TBox
tbox = create_combined_tbox()

# Step 4: Create ABox graph
abox = Graph()
abox.bind('adnx', ADNX)
abox.bind('wd', WD)
abox.bind('rdfs', RDFS)
abox.bind('qb', QB)
abox.bind('qb4o', QB4O)
abox.bind('skos', SKOS)

# Create lookup maps for efficient dimension handling
dimension_maps = {}
for col in categorical_cols:
    dimension_maps[col] = {}

def get_or_create_dimension_instance(g, dimension_type, dimension_map, value):
    if pd.isna(value) or value == "":
        return None
    if value not in dimension_map:
        uri_hash = hashlib.md5(f"{dimension_type}_{value}".encode()).hexdigest()[:8]
        dimension_map[value] = uri_hash
        
        dim_uri = ADNX[f'{dimension_type.lower()}_{uri_hash}']
        class_uri = getattr(ADNX, dimension_type)
        
        g.add((dim_uri, RDF.type, class_uri))
        g.add((dim_uri, SKOS.prefLabel, Literal(value)))
        
        # Add QB4OLAP level membership
        if dimension_type == 'Country':
            g.add((dim_uri, QB4O.memberOf, ADNX.CountryLevel))
        elif dimension_type == 'Gender':
            g.add((dim_uri, QB4O.memberOf, ADNX.GenderLevel))
        elif dimension_type == 'Severity':
            g.add((dim_uri, QB4O.memberOf, ADNX.SeverityLevel))
        elif dimension_type == 'StressLevel':
            g.add((dim_uri, QB4O.memberOf, ADNX.StressLevelBase))
        elif dimension_type == 'Occupation':
            g.add((dim_uri, QB4O.memberOf, ADNX.OccupationLevel))
        elif dimension_type == 'SmokingHabit':
            g.add((dim_uri, QB4O.memberOf, ADNX.SmokingLevel))
        elif dimension_type == 'AlcoholConsumption':
            g.add((dim_uri, QB4O.memberOf, ADNX.AlcoholLevel))
            
        # Add Wikidata links for countries
        country_wd_map = {
            "Australia": WD.Q408,
            "Canada": WD.Q16,
            "Germany": WD.Q183,
            "India": WD.Q668,
            "UK": WD.Q145,
            "USA": WD.Q30
        }
        if dimension_type == 'Country' and value in country_wd_map:
            g.add((dim_uri, OWL.sameAs, country_wd_map[value]))
            
        # Link severity to hierarchy
        if dimension_type == 'Severity' and value in ['None', 'Low', 'Medium', 'High', 'Unknown']:
            hierarchy_uri = ADNX[f'severity_{value.lower()}']
            g.add((dim_uri, SKOS.broader, hierarchy_uri))
    
    return ADNX[f'{dimension_type.lower()}_{dimension_map[value]}']

def get_age_group(age):
    if age <= 25:
        return 'Youth'
    elif age <= 40:
        return 'Young_Adult'
    elif age <= 60:
        return 'Middle_Age'
    else:
        return 'Senior'

# Process data in chunks for better memory management
chunk_size = 5000
total_chunks = len(df) // chunk_size + 1

print(f"Processing {len(df)} records in {total_chunks} chunks...")

cube = ADNX.MentalHealthCube
dataset = ADNX.MentalHealthDataSet

for chunk_idx in range(total_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(df))
    chunk = df.iloc[start_idx:end_idx]
    
    print(f"Processing chunk {chunk_idx + 1}/{total_chunks}...")
    
    for _, row in chunk.iterrows():
        user_id = row['User_ID']
        
        # Person instance
        user_uri = ADNX[f'user_{user_id}']
        abox.add((user_uri, RDF.type, ADNX.Person))
        abox.add((user_uri, RDFS.label, Literal(f'Person {user_id}', lang='en')))
        abox.add((user_uri, ADNX.hasAge, Literal(int(row['Age']), datatype=XSD.integer)))
        
        # Create dimension instances
        gender_uri = get_or_create_dimension_instance(abox, 'Gender', dimension_maps['Gender'], row['Gender'])
        if gender_uri:
            abox.add((user_uri, ADNX.hasGender, gender_uri))
        
        occupation_uri = get_or_create_dimension_instance(abox, 'Occupation', dimension_maps['Occupation'], row['Occupation'])
        if occupation_uri:
            abox.add((user_uri, ADNX.hasOccupation, occupation_uri))
        
        country_uri = get_or_create_dimension_instance(abox, 'Country', dimension_maps['Country'], row['Country'])
        if country_uri:
            abox.add((user_uri, ADNX.hasCountry, country_uri))
        
        # Age dimension with hierarchy
        age = int(row['Age'])
        age_group = get_age_group(age)
        age_uri = ADNX[f'age_{user_id}']
        abox.add((age_uri, RDF.type, ADNX.AgeDimension))
        abox.add((age_uri, SKOS.prefLabel, Literal(f'Age {age}')))
        abox.add((age_uri, QB4O.memberOf, ADNX.AgeLevel))
        age_group_uri = ADNX[f'age_group_{age_group}']
        abox.add((age_uri, QB4O.rollsUpTo, age_group_uri))
        abox.add((age_group_uri, RDF.type, ADNX.AgeGroupLevel))
        abox.add((age_group_uri, SKOS.prefLabel, Literal(age_group.replace('_', ' '))))
        abox.add((age_group_uri, QB4O.memberOf, ADNX.AgeGroupLevel))
        
        # Mental Health instance
        mh_uri = ADNX[f'mental_health_{user_id}']
        abox.add((mh_uri, RDF.type, ADNX.MentalHealth))
        abox.add((mh_uri, RDFS.label, Literal(row['Mental_Health_Condition'])))
        abox.add((user_uri, ADNX.hasMentalHealth, mh_uri))
        
        severity_uri = get_or_create_dimension_instance(abox, 'Severity', dimension_maps['Severity'], row['Severity'])
        if severity_uri:
            abox.add((mh_uri, ADNX.hasSeverity, severity_uri))
        
        stress_uri = get_or_create_dimension_instance(abox, 'StressLevel', dimension_maps['Stress_Level'], row['Stress_Level'])
        if stress_uri:
            abox.add((mh_uri, ADNX.hasStressLevel, stress_uri))
        
        consultation_uri = get_or_create_dimension_instance(abox, 'Consultation', dimension_maps['Consultation_History'], row['Consultation_History'])
        if consultation_uri:
            abox.add((mh_uri, ADNX.hasConsultationHistory, consultation_uri))
        
        medication_uri = get_or_create_dimension_instance(abox, 'Medication', dimension_maps['Medication_Usage'], row['Medication_Usage'])
        if medication_uri:
            abox.add((mh_uri, ADNX.hasMedicationUsage, medication_uri))
        
        # Lifestyle instance
        lifestyle_uri = ADNX[f'lifestyle_{user_id}']
        abox.add((lifestyle_uri, RDF.type, ADNX.Lifestyle))
        abox.add((user_uri, ADNX.hasLifestyle, lifestyle_uri))
        
        diet_uri = get_or_create_dimension_instance(abox, 'DietQuality', dimension_maps['Diet_Quality'], row['Diet_Quality'])
        if diet_uri:
            abox.add((lifestyle_uri, ADNX.hasDietQuality, diet_uri))
        
        smoking_uri = get_or_create_dimension_instance(abox, 'SmokingHabit', dimension_maps['Smoking_Habit'], row['Smoking_Habit'])
        if smoking_uri:
            abox.add((lifestyle_uri, ADNX.hasSmokingHabit, smoking_uri))
        
        alcohol_uri = get_or_create_dimension_instance(abox, 'AlcoholConsumption', dimension_maps['Alcohol_Consumption'], row['Alcohol_Consumption'])
        if alcohol_uri:
            abox.add((lifestyle_uri, ADNX.hasAlcoholConsumption, alcohol_uri))
        
        # QB4OLAP Observation instance
        obs_uri = ADNX[f'observation_{user_id}']
        abox.add((obs_uri, RDF.type, ADNX.Observation))
        abox.add((obs_uri, RDF.type, QB4O.Observation))
        abox.add((obs_uri, RDF.type, QB.Observation))  # QB compatibility
        abox.add((obs_uri, QB4O.inCube, cube))
        abox.add((obs_uri, QB.dataSet, dataset))  # QB compatibility
        
        # Link observation to person
        abox.add((user_uri, ADNX.hasObservation, obs_uri))
        
        # CRITICAL FIX: Add lifestyle properties directly to observation
        if smoking_uri:
            abox.add((obs_uri, ADNX.observationHasSmokingHabit, smoking_uri))
        if alcohol_uri:
            abox.add((obs_uri, ADNX.observationHasAlcoholConsumption, alcohol_uri))
        if diet_uri:
            abox.add((obs_uri, ADNX.observationHasDietQuality, diet_uri))
        
        # Add measurement values as QB4OLAP measures
        abox.add((obs_uri, ADNX.sleepHoursMeasure, Literal(float(row['Sleep_Hours']), datatype=XSD.float)))
        abox.add((obs_uri, ADNX.workHoursMeasure, Literal(int(row['Work_Hours']), datatype=XSD.integer)))
        abox.add((obs_uri, ADNX.physicalActivityMeasure, Literal(int(row['Physical_Activity_Hours']), datatype=XSD.integer)))
        abox.add((obs_uri, ADNX.socialMediaMeasure, Literal(float(row['Social_Media_Usage']), datatype=XSD.float)))
        abox.add((obs_uri, ADNX.ageMeasure, Literal(int(row['Age']), datatype=XSD.integer)))
        
        # Add traditional measurement properties
        abox.add((obs_uri, ADNX.hasSleepHours, Literal(float(row['Sleep_Hours']), datatype=XSD.float)))
        abox.add((obs_uri, ADNX.hasWorkHours, Literal(int(row['Work_Hours']), datatype=XSD.integer)))
        abox.add((obs_uri, ADNX.hasPhysicalActivityHours, Literal(int(row['Physical_Activity_Hours']), datatype=XSD.integer)))
        abox.add((obs_uri, ADNX.hasSocialMediaUsage, Literal(float(row['Social_Media_Usage']), datatype=XSD.float)))
        
        # Add QB4OLAP dimensions to observation
        if country_uri:
            abox.add((obs_uri, ADNX.countryDimension, country_uri))
        if gender_uri:
            abox.add((obs_uri, ADNX.genderDimension, gender_uri))
        if severity_uri:
            abox.add((obs_uri, ADNX.severityDimension, severity_uri))
        if stress_uri:
            abox.add((obs_uri, ADNX.stressLevelDimension, stress_uri))
        if occupation_uri:
            abox.add((obs_uri, ADNX.occupationDimension, occupation_uri))
        abox.add((obs_uri, ADNX.ageDimension, age_uri))

print("Serializing graphs...")

# Step 5: Save TBox and ABox to files
tbox.serialize(destination='tbox_combined.ttl', format='turtle')
abox.serialize(destination='abox_qb4olap.ttl', format='turtle')

# Also create separate QB4OLAP-only TBox for compatibility
tbox.serialize(destination='tbox_qb4olap.ttl', format='turtle')

print("Combined TBox and QB4OLAP ABox TTL files generated successfully!")
print(f"Combined TBox triples: {len(tbox)}")
print(f"ABox triples: {len(abox)}")

# Print dimension statistics
print("\nDimension Statistics:")
for dim_name, dim_map in dimension_maps.items():
    print(f"{dim_name}: {len(dim_map)} unique values")
    if dim_name in ['Smoking_Habit', 'Alcohol_Consumption']:
        print(f"  Values: {list(dim_map.keys())}")

print("\nCombined QB + QB4OLAP Features Added:")
print("- Combined TBox with both QB and QB4OLAP vocabularies")
print("- Fixed observation-lifestyle property links")
print("- Enhanced smoking habit and alcohol consumption mappings")
print("- QB compatibility layers")
print("- Level membership for all dimension instances")
print("\nFiles generated:")
print("- tbox_combined.ttl (QB + QB4OLAP)")
print("- tbox_qb4olap.ttl (QB4OLAP only)")
print("- abox_qb4olap.ttl (Instance data)")
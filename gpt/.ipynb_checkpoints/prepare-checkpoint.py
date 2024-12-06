import os
import numpy as np
import random
import json
import tiktoken
from datetime import datetime, timedelta

def generate_synthetic_patient():
    """Generate a single synthetic patient record"""
    
    # Basic patient information
    genders = ['Male', 'Female']
    blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
    conditions = [
        'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Arthritis',
        'Anxiety Disorder', 'Depression', 'Migraine', 'Hypothyroidism'
    ]
    medications = [
        'Lisinopril', 'Metformin', 'Albuterol', 'Ibuprofen',
        'Sertraline', 'Levothyroxine', 'Omeprazole', 'Atorvastatin'
    ]
    
    # Generate random age between 18 and 85
    age = random.randint(18, 85)
    
    # Generate random vital signs within typical ranges
    vitals = {
        'Blood Pressure': f"{random.randint(90, 160)}/{random.randint(60, 100)}",
        'Heart Rate': random.randint(60, 100),
        'Temperature': round(random.uniform(36.1, 37.5), 1),
        'SpO2': random.randint(95, 100)
    }
    
    # Random number of conditions and medications
    patient_conditions = random.sample(conditions, random.randint(0, 3))
    patient_medications = random.sample(medications, random.randint(0, 4))
    
    # Generate random dates for visits within the last year
    current_date = datetime.now()
    visits = []
    for _ in range(random.randint(1, 5)):
        days_ago = random.randint(0, 365)
        visit_date = (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        visits.append({
            'date': visit_date,
            'reason': random.choice(['Regular Checkup', 'Follow-up', 'Acute Illness', 'Prescription Renewal'])
        })
    
    # Construct patient record
    patient = {
        'Patient_ID': f"P{random.randint(10000, 99999)}",
        'Age': age,
        'Gender': random.choice(genders),
        'Blood_Type': random.choice(blood_types),
        'Vital_Signs': vitals,
        'Medical_Conditions': patient_conditions,
        'Medications': patient_medications,
        'Recent_Visits': sorted(visits, key=lambda x: x['date'])
    }
    
    return patient

def generate_medical_dataset(num_patients):
    """Generate a dataset of synthetic patient records"""
    return [generate_synthetic_patient() for _ in range(num_patients)]

def format_patient_record(patient):
    """Format a patient record into a readable text format"""
    record = f"Patient ID: {patient['Patient_ID']}\n"
    record += f"Age: {patient['Age']}\n"
    record += f"Gender: {patient['Gender']}\n"
    record += f"Blood Type: {patient['Blood_Type']}\n"
    record += "\nVital Signs:\n"
    for vital, value in patient['Vital_Signs'].items():
        record += f"- {vital}: {value}\n"
    record += "\nMedical Conditions:\n"
    for condition in patient['Medical_Conditions']:
        record += f"- {condition}\n"
    record += "\nCurrent Medications:\n"
    for medication in patient['Medications']:
        record += f"- {medication}\n"
    record += "\nRecent Visits:\n"
    for visit in patient['Recent_Visits']:
        record += f"- {visit['date']}: {visit['reason']}\n"
    record += "\n---\n\n"
    return record

# Generate synthetic dataset
print("Generating synthetic medical dataset...")
train_patients = generate_medical_dataset(100)  # 5000 training records
val_patients = generate_medical_dataset(20)    # 500 validation records

# Convert to text format
train_data = ''.join([format_patient_record(patient) for patient in train_patients])
val_data = ''.join([format_patient_record(patient) for patient in val_patients])

# Save raw text files (for reference)
os.makedirs('data/medical', exist_ok=True)
with open('data/medical/train_raw.txt', 'w', encoding='utf-8') as f:
    f.write(train_data)
with open('data/medical/val_raw.txt', 'w', encoding='utf-8') as f:
    f.write(val_data)

# Encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"Train dataset has {len(train_ids):,} tokens")
print(f"Validation dataset has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save binary files
train_ids.tofile('data/medical/train.bin')
val_ids.tofile('data/medical/val.bin')

# Save metadata
meta = {
    'vocab_size': enc.n_vocab,
    'data_type': 'medical_records',
    'train_samples': len(train_patients),
    'val_samples': len(val_patients)
}

with open('data/medical/meta.pkl', 'wb') as f:
    import pickle
    pickle.dump(meta, f)

print("Dataset preparation completed!")
print(f"Files saved in data/medical/")
print(f"- train.bin: {len(train_ids):,} tokens")
print(f"- val.bin: {len(val_ids):,} tokens")
print(f"- meta.pkl: contains vocabulary size and dataset metadata")
print(f"- train_raw.txt and val_raw.txt: human-readable versions of the data")
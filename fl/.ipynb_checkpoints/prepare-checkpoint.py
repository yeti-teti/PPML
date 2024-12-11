import os
import numpy as np
import random
import json
import tiktoken
import pickle
from datetime import datetime, timedelta

def generate_synthetic_patient():
    """Generate a single synthetic patient record"""
    
    # Basic patient information
    genders = ['Male', 'Female']
    blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
    conditions = [
        'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Arthritis',
        'Anxiety Disorder', 'Depression', 'Migraine'
    ]
    medications = [
        'Lisinopril', 'Metformin', 'Albuterol', 'Ibuprofen',
        'Sertraline', 'Levothyroxine', 'Omeprazole'
    ]
    
    # Generate random age between 18 and 85
    age = random.randint(18, 85)
    
    # Generate random vital signs within typical ranges
    vitals = {
        'Blood Pressure': f"{random.randint(90, 160)}/{random.randint(60, 100)}",
        'Heart Rate': random.randint(60, 100),
        'Temperature': round(random.uniform(36.1, 37.5), 1)
    }
    
    # Random number of conditions and medications
    patient_conditions = random.sample(conditions, random.randint(0, 2))
    patient_medications = random.sample(medications, random.randint(0, 2))
    
    # Generate random dates for visits within the last year
    current_date = datetime.now()
    visits = []
    for _ in range(random.randint(1, 3)):
        days_ago = random.randint(0, 365)
        visit_date = (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        visits.append({
            'date': visit_date,
            'reason': random.choice(['Regular Checkup', 'Follow-up', 'Acute Illness'])
        })
    
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

# FL Configuration
NUM_CLIENTS = 5
PATIENTS_PER_CLIENT = 50
VAL_SPLIT = 0.2  # 20% validation split

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Create directory structure
base_dir = 'data/medical_fl'
os.makedirs(base_dir, exist_ok=True)

# Generate and save data for each client
for client_id in range(NUM_CLIENTS):
    client_dir = os.path.join(base_dir, f'client_{client_id}')
    os.makedirs(client_dir, exist_ok=True)
    
    print(f"Generating data for client {client_id}")
    
    # Generate all patients for this client
    all_patients = [generate_synthetic_patient() for _ in range(PATIENTS_PER_CLIENT)]
    random.shuffle(all_patients)  # Shuffle before splitting
    
    # Split into train and validation
    num_val = int(PATIENTS_PER_CLIENT * VAL_SPLIT)
    val_patients = all_patients[:num_val]
    train_patients = all_patients[num_val:]
    
    # Process training data
    train_data = ''.join([format_patient_record(p) for p in train_patients])
    train_ids = enc.encode_ordinary(train_data)
    train_ids = np.array(train_ids, dtype=np.uint16)
    
    # Process validation data
    val_data = ''.join([format_patient_record(p) for p in val_patients])
    val_ids = enc.encode_ordinary(val_data)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    # Save binary data
    train_ids.tofile(os.path.join(client_dir, 'train.bin'))
    val_ids.tofile(os.path.join(client_dir, 'val.bin'))
    
    # Save raw text for reference
    with open(os.path.join(client_dir, 'train_raw.txt'), 'w') as f:
        f.write(train_data)
    with open(os.path.join(client_dir, 'val_raw.txt'), 'w') as f:
        f.write(val_data)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'num_train_patients': len(train_patients),
        'num_val_patients': len(val_patients),
        'num_train_tokens': len(train_ids),
        'num_val_tokens': len(val_ids)
    }
    
    with open(os.path.join(client_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Client {client_id}:")
    print(f"- Training: {len(train_ids)} tokens ({len(train_patients)} patients)")
    print(f"- Validation: {len(val_ids)} tokens ({len(val_patients)} patients)")

print("\nDataset preparation completed!")
print(f"Data saved in: {base_dir}")
print(f"Total clients: {NUM_CLIENTS}")
print(f"Patients per client: {PATIENTS_PER_CLIENT}")
print(f"Train/Val split: {1-VAL_SPLIT:.1%}/{VAL_SPLIT:.1%}")
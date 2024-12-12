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
    conditions = {
        'Hypertension': 0.3,
        'Type 2 Diabetes': 0.2,
        'Asthma': 0.15,
        'Arthritis': 0.15,
        'Anxiety Disorder': 0.2,
        'Depression': 0.2,
        'Migraine': 0.15,
        'Hypothyroidism': 0.1,
        'High Cholesterol': 0.25,
        'Obesity': 0.3,
    }
    
    medications = {
        'Hypertension': ['Lisinopril', 'Amlodipine', 'Losartan'],
        'Type 2 Diabetes': ['Metformin', 'Glipizide', 'Januvia'],
        'Asthma': ['Albuterol', 'Flovent', 'Singulair'],
        'Arthritis': ['Ibuprofen', 'Celebrex', 'Meloxicam'],
        'Anxiety Disorder': ['Sertraline', 'Buspirone', 'Alprazolam'],
        'Depression': ['Fluoxetine', 'Bupropion', 'Venlafaxine'],
        'Migraine': ['Sumatriptan', 'Rizatriptan', 'Topiramate'],
        'Hypothyroidism': ['Levothyroxine', 'Synthroid'],
        'High Cholesterol': ['Atorvastatin', 'Simvastatin', 'Rosuvastatin'],
        'Obesity': ['Phentermine', 'Orlistat', 'Contrave']
    }
    
    # Generate age with distribution
    if random.random() < 0.7:  # 70% adults
        age = random.randint(25, 65)
    else:  # 30% elderly
        age = random.randint(66, 85)
    
    # Select conditions based on age and probabilities
    patient_conditions = []
    for condition, prob in conditions.items():
        # Increase probability for older patients
        if age > 65:
            prob *= 1.5
        if random.random() < prob:
            patient_conditions.append(condition)
    
    # Select medications based on conditions
    patient_medications = []
    for condition in patient_conditions:
        if condition in medications:
            # Get available medications for this condition
            available_meds = medications[condition]
            # Sample 1-2 medications, but no more than what's available
            num_meds = min(random.randint(1, 2), len(available_meds))
            patient_medications.extend(random.sample(available_meds, num_meds))
    
    # Generate vital signs
    base_systolic = 110 if age < 65 else 120
    base_diastolic = 70 if age < 65 else 75
    
    if 'Hypertension' in patient_conditions:
        base_systolic += random.randint(10, 30)
        base_diastolic += random.randint(5, 15)
    
    vitals = {
        'Blood Pressure': f"{base_systolic + random.randint(-5, 5)}/{base_diastolic + random.randint(-5, 5)}",
        'Heart Rate': random.randint(60, 100),
        'Temperature': round(random.uniform(36.1, 37.5), 1),
        'SpO2': random.randint(95, 100) if 'Asthma' not in patient_conditions else random.randint(92, 98)
    }
    
    # Generate visits
    current_date = datetime.now()
    visits = []
    num_visits = random.randint(3, 8)
    
    visit_types = [
        "Regular Checkup",
        "Follow-up",
        "Prescription Renewal",
        "Lab Results Review",
        "Acute Illness"
    ]
    
    for i in range(num_visits):
        days_ago = random.randint(i * 45, (i + 1) * 45)  # Space visits out
        visit_date = (current_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        if i == 0:
            reason = "Initial Consultation"
            notes = "Complete health assessment performed."
        else:
            if patient_conditions:
                condition = random.choice(patient_conditions)
                reason = random.choice(visit_types)
                notes = f"Follow-up for {condition}. Medications reviewed."
            else:
                reason = "Regular Checkup"
                notes = "Routine health maintenance."
        
        visits.append({
            'date': visit_date,
            'reason': reason,
            'notes': notes
        })
    
    # Sort visits chronologically
    visits.sort(key=lambda x: x['date'])
    
    # Create patient record
    patient = {
        'Patient_ID': f"P{random.randint(10000, 99999)}",
        'Age': age,
        'Gender': random.choice(genders),
        'Blood_Type': random.choice(blood_types),
        'Vital_Signs': vitals,
        'Medical_Conditions': patient_conditions,
        'Medications': patient_medications,
        'Recent_Visits': visits
    }
    
    return patient

def format_patient_record(patient):
    """Format a patient record into a structured text format"""
    record = f"Patient Record\n{'='*50}\n\n"
    record += f"Patient ID: {patient['Patient_ID']}\n"
    record += f"Age: {patient['Age']}\n"
    record += f"Gender: {patient['Gender']}\n"
    record += f"Blood Type: {patient['Blood_Type']}\n"
    
    record += f"\nVital Signs\n{'-'*20}\n"
    for vital, value in patient['Vital_Signs'].items():
        record += f"{vital}: {value}\n"
    
    record += f"\nMedical Conditions\n{'-'*20}\n"
    if patient['Medical_Conditions']:
        for condition in patient['Medical_Conditions']:
            record += f"- {condition}\n"
    else:
        record += "No chronic conditions\n"
    
    record += f"\nMedications\n{'-'*20}\n"
    if patient['Medications']:
        for medication in patient['Medications']:
            record += f"- {medication}\n"
    else:
        record += "No current medications\n"
    
    record += f"\nVisit History\n{'-'*20}\n"
    for visit in patient['Recent_Visits']:
        record += f"Date: {visit['date']}\n"
        record += f"Reason: {visit['reason']}\n"
        record += f"Notes: {visit['notes']}\n"
        record += "-" * 10 + "\n"
    
    record += "\n" + "="*50 + "\n\n"
    return record

def main():
    print("Generating synthetic medical dataset...")
    
    # Generate synthetic data
    train_patients = [generate_synthetic_patient() for _ in range(500)]
    val_patients = [generate_synthetic_patient() for _ in range(50)]
    
    # Convert to text format
    train_data = ''.join([format_patient_record(patient) for patient in train_patients])
    val_data = ''.join([format_patient_record(patient) for patient in val_patients])
    
    # Create data directory
    os.makedirs('data/medical', exist_ok=True)
    
    # Save raw text files
    with open('data/medical/train_raw.txt', 'w', encoding='utf-8') as f:
        f.write(train_data)
    with open('data/medical/val_raw.txt', 'w', encoding='utf-8') as f:
        f.write(val_data)
    
    # Tokenize data
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"Train dataset has {len(train_ids):,} tokens")
    print(f"Validation dataset has {len(val_ids):,} tokens")
    
    # Save tokenized data
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
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
        pickle.dump(meta, f)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Training records: {len(train_patients)}")
    print(f"Validation records: {len(val_patients)}")
    print(f"Training tokens: {len(train_ids):,}")
    print(f"Validation tokens: {len(val_ids):,}")
    print("\nFiles saved in data/medical/:")
    print("- train.bin")
    print("- val.bin")
    print("- meta.pkl")
    print("- train_raw.txt")
    print("- val_raw.txt")

if __name__ == "__main__":
    main()
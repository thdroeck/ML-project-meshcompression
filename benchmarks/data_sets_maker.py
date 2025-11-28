import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

# --- CONFIGURATIE ---
# Gebruik Pathlib voor robuuste paden
DATA_DIR = Path("../../lib/Manifold40")
MIN_SAMPLES_PER_CLASS = 2  # Vereis minimaal 2 bestanden per categorie
# --------------------

print(f"Bron data: {DATA_DIR}")

# 1. Verzamel alle bestanden en labels
search_path = DATA_DIR / "**" / "*.obj"
all_files = []
labels = []

print("Bestanden verzamelen...")
# Gebruik glob.glob met de Path object (converteer naar string voor glob)
for mesh_path_str in glob.glob(str(search_path), recursive=True):
    mesh_path = Path(mesh_path_str)
    all_files.append(str(mesh_path))  # Sla op als string voor compatibiliteit

    # --- DE FIX: Bepaal het label correct ---
    try:
        # Krijg het pad relatief aan de DATA_DIR
        # bv: "airplane/airplane_001/mesh.obj"
        relative_path = mesh_path.relative_to(DATA_DIR)

        # Het label is het EERSTE deel van dit pad
        # bv: "airplane"
        label = relative_path.parts[0]
        labels.append(label)
    except ValueError:
        print(f"Waarschuwing: Bestand {mesh_path} lijkt buiten de DATA_DIR te vallen. Wordt overgeslagen.")
        continue
    # ----------------------------------------

print(f"Totaal {len(all_files)} bestanden gevonden.")

if not all_files:
    print("Fout: Geen .obj bestanden gevonden. Controleer het DATA_DIR pad.")
    exit()

# 2. Filter de dataset op minimale grootte
df = pd.DataFrame({'file_path': all_files, 'label': labels})

# Toon een voorbeeld van de gevonden categorieën (nuttig voor debuggen)
print("\nGevonden categorieën (top 10):")
print(df['label'].value_counts().head(10))
print("...")

label_counts = df['label'].value_counts()
valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
filtered_df = df[df['label'].isin(valid_labels)]

filtered_files = filtered_df['file_path'].tolist()
filtered_labels = filtered_df['label'].tolist()

num_removed = len(all_files) - len(filtered_files)
print(f"{num_removed} bestanden verwijderd (uit categorieën met < {MIN_SAMPLES_PER_CLASS} samples).")
print(f"Resterende bestanden voor splitsing: {len(filtered_files)}")

if len(filtered_files) == 0:
    print("Geen bestanden over na filteren. Stop script.")
    exit()

# 3. Maak de eerste split: 80% (Train+Val) en 20% (Test)
# Nu zal stratify=filtered_labels werken
train_val_files, test_files, train_val_labels, test_labels = train_test_split(
    filtered_files,
    filtered_labels,
    test_size=0.20,  # 20% voor de testset
    random_state=42,  # Voor reproduceerbaarheid
    stratify=filtered_labels  # Gebruik de gefilterde labels
)

# 4. Maak de tweede split: 70% (Train) en 10% (Validation)
# We willen 10% van het *totaal*, wat 12.5% is van de 80% (0.10 / 0.80 = 0.125)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_val_files,
    train_val_labels,
    test_size=0.125,  # 10% van het totaal (0.10 / 0.80)
    random_state=42,
    stratify=train_val_labels
)

print(f"\nTraining set:   {len(train_files)} bestanden")
print(f"Validation set: {len(val_files)} bestanden")
print(f"Test set:       {len(test_files)} bestanden")

# --- 5. BESTANDEN OPSLAAN ---

output_dir = Path("../../data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "train.txt", "w") as f:
    f.write("\n".join(train_files))
with open(output_dir / "validation.txt", "w") as f:
    f.write("\n".join(val_files))
with open(output_dir / "test.txt", "w") as f:
    f.write("\n".join(test_files))

print(f"\nBestandslijsten (manifesten) opgeslagen in {output_dir}")

print("\nStarten met fysiek kopiëren van bestanden...")
base_output_dir = Path("../../lib/Manifold40_split")
train_dir = base_output_dir / "train"
val_dir = base_output_dir / "validation"
test_dir = base_output_dir / "test"

# # Maak de mappen aan
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# # Helper-functie om te kopiëren met behoud van categoriestructuur
def copy_files(file_list, labels, dest_dir):
    print(f"Kopiëren van {len(file_list)} bestanden naar {dest_dir}...")
    # Gebruik een set om te voorkomen dat mkdir duizenden keren wordt aangeroepen
    created_dirs = set()

    for f, label in zip(file_list, labels):
        category_dest_dir = dest_dir / label
        if label not in created_dirs:
            category_dest_dir.mkdir(exist_ok=True)
            created_dirs.add(label)
        # We moeten de unieke mapnaam (bv. airplane_001) behouden
        instance_name = Path(f).parent.name
        instance_dest_dir = category_dest_dir / instance_name
        instance_dest_dir.mkdir(exist_ok=True)

        dest_path = instance_dest_dir / Path(f).name

        shutil.copy(f, dest_path)

# # Bundel de lijsten voor de kopieerfunctie
sets_to_copy = [
    (train_files, train_labels, train_dir),
    (val_files, val_labels, val_dir),
    (test_files, test_labels, test_dir)
]

for files, lbls, d in sets_to_copy:
    copy_files(files, lbls, d)

print("Bestanden fysiek gekopieerd naar nieuwe mappen.")
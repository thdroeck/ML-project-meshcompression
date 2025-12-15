import csv
import argparse

def average_ratio(csv_path):
    values = []

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            val = row.get("ratio_latent_only", "").strip()
            if val:
                try:
                    values.append(float(val))
                except ValueError:
                    pass  # skip invalid numbers

    if not values:
        return None

    return sum(values) / len(values)

def main():
    parser = argparse.ArgumentParser(description="Compute average of ratio_latent_only field.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()

    avg = average_ratio(args.csv_file)

    if avg is None:
        print("No valid ratio_latent_only values found.")
    else:
        print("Average ratio_latent_only:", f"{avg:.2f}")

if __name__ == "__main__":
    main()

import pandas as pd
import os

FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "LocFacil_rental_demo_database.xlsx")

def load_all_data():
    products = pd.read_excel(FILE_PATH, sheet_name="Products")
    customers = pd.read_excel(FILE_PATH, sheet_name="Customers")
    rentals = pd.read_excel(FILE_PATH, sheet_name="Rentals")
    policies = pd.read_excel(FILE_PATH, sheet_name="Policies")

    return {
        "products": products,
        "customers": customers,
        "rentals": rentals,
        "policies": policies
    }

if __name__ == "__main__":
    data = load_all_data()

    print("Products:")
    print(data["products"].head())

    print("\nCustomers:")
    print(data["customers"].head())

    print("\nRentals:")
    print(data["rentals"].head())

    print("\nPolicies:")
    print(data["policies"].head())
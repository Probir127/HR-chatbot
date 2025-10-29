import json
import os
import re

JSON_PATH = "D:/tutorial/data/employees.json"

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError("❌ Employee JSON not found. Run export_hr_to_json.py first.")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    EMPLOYEES = json.load(f)


# -------------------------------------------------------------
# Helper: normalize emails (ignore spaces, Gmail dots)
# -------------------------------------------------------------
def normalize_email(e: str) -> str:
    e = e.lower().strip().replace(" ", "")
    if e.endswith("@gmail.com"):
        local, domain = e.split("@")
        local = local.replace(".", "")
        return f"{local}@{domain}"
    return e


# -------------------------------------------------------------
# Find employee by partial or fuzzy name
# -------------------------------------------------------------
def find_employee_by_name(name: str):
    name_lower = name.lower().strip()
    best_match = None
    for emp in EMPLOYEES:
        emname = emp.get("name", "").lower()
        # Partial / token fuzzy match
        if all(tok in emname for tok in name_lower.split()):
            best_match = emp
            break
    if not best_match:
        # fallback substring
        for emp in EMPLOYEES:
            if name_lower in emp.get("name", "").lower():
                return emp
    return best_match


# -------------------------------------------------------------
# Verify email belongs to the given employee
# -------------------------------------------------------------
def verify_employee_email(name: str, email: str) -> bool:
    record = find_employee_by_name(name)
    if not record:
        return False
    rec_email = normalize_email(record.get("email", ""))
    inp_email = normalize_email(email)
    return rec_email == inp_email


# -------------------------------------------------------------
# Format employee info for output
# -------------------------------------------------------------
def format_employee_info(emp):
    if not emp:
        return "❌ Employee not found."

    return (
        f"👤 **{emp['name']}**\n"
        f"📧 {emp.get('email','N/A')}\n"
        f"🏢 {emp.get('position','N/A')}\n"
        f"🩸 Blood Group: {emp.get('blood_group','N/A')}\n"
        f"📋 Table: {emp.get('table','N/A')}"
    )


# -------------------------------------------------------------
# Manual test mode
# -------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Employee Lookup Debug Mode\n")
    name = input("Enter employee name: ").strip()
    emp = find_employee_by_name(name)
    print(format_employee_info(emp))

    email = input("Enter email to verify: ").strip()
    verified = verify_employee_email(name, email)
    print("✅ Verified!" if verified else "❌ Invalid email.")

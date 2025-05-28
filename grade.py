
# Second Semester
# First Semester 2022–2023
subjects = [
    ("Math 14", "Calculus II", 4, 0, 1.00, 4),
    ("ITE 13", "Intermediate Programming", 2, 1, 1.25, 3),
    ("CSC 102", "Discrete Structures 1", 2, 1, 2.50, 3),
    ("EEP 2", "ENGLISH ENHANCEMENT PROGRAM 2", 2, 0, 1.75, 2),
    ("RPH", "Readings in the Philippine History", 3, 0, 1.50, 3),
    ("Phys 20", "Physics for Engineers", 3, 1, 1.25, 4),
    ("PE 3", "Individual and Dual Sports", 2, 0, 1.25, 2),
    # ("Phys B", "Physics Enhancement Program for Engineers", 2, 0, None, 0),  # Ignored (non-credit program)
    # ("SE 103", "Intrapersonal and Interpersonal Skills", 0, 0, "PASSED", 0),  # Ignored
]


# # Sample data (subject_code, subject_name, lec_units, lab_units, grade, total_units)
# subjects = [
#     # First Semester 2021-2022
#     ("ITE 10", "INTRODUCTION TO COMPUTING", 2, 1, 1.50, 3),
#     ("ITE 11", "MATHEMATICAL APPLICATIONS FOR ITE", 3, 1, 1.75, 4),
#     ("ITE 15", "SOCIAL ISSUES AND PROFESSIONAL ISSUES", 3, 0, 1.50, 3),
#     ("PC", "PURPOSIVE COMMUNICATION", 3, 0, 1.25, 3),
#     ("PE 1", "PHYSICAL FITNESS", 2, 0, 1.00, 2),
#     ("STS", "SCIENCE, TECHNOLOGY AND SOCIETY", 3, 0, 1.50, 3),

#     # Second Semester 2021-2022
#     ("EEP 1", "ENGLISH ENHANCEMENT PROGRAM 1", 2, 0, 1.75, 2),
#     ("ITE 12", "FUNDAMENTALS OF PROGRAMMING", 2, 1, 2.00, 3),
#     ("LWR", "LIFE AND WORKS OF RIZAL", 3, 0, 1.50, 3),
#     ("MATH 13", "CALCULUS 1", 4, 0, 1.25, 4),
#     ("MMW", "MATHEMATICS IN THE MODERN WORLD (WITH LAB)", 3, 0, 2.00, 3),
#     ("PE 2", "RHYTHMIC ACTIVITIES", 2, 0, 1.25, 2),
#     ("UTS", "UNDERSTANDING THE SELF", 3, 0, 1.00, 3),

#     # First Semester 2022-2023
#     ("CSC 102", "DISCRETE STRUCTURES I", 2, 1, 2.50, 3),
#     ("EEP 2", "ENGLISH ENHANCEMENT PROGRAM 2", 2, 0, 1.75, 2),
#     ("ITE 13", "INTERMEDIATE PROGRAMMING", 2, 1, 1.25, 3),
#     ("MATH 14", "CALCULUS II", 4, 0, 1.00, 4),
#     ("PE 3", "INDIVIDUAL AND DUAL SPORTS", 2, 0, 1.25, 2),
#     ("PHYS 20", "PHYSICS FOR ENGINEERS", 3, 1, 1.25, 4),
#     ("RPH", "READINGS IN THE PHILIPPINE HISTORY", 3, 0, 1.50, 3),

#     # Second Semester 2022-2023
#     ("CSC 103", "DISCRETE STRUCTURES 2", 2, 1, 2.25, 3),
#     ("CSC 104", "OBJECT-ORIENTED PROGRAMMING", 2, 1, 1.75, 3),
#     ("CSC 106", "SOFTWARE ENGINEERING 1", 2, 1, 2.50, 3),
#     ("EEP 3", "ENGLISH ENHANCEMENT PROGRAM 3", 2, 0, 1.50, 2),
#     ("ITE 14", "DATA STRUCTURES AND ALGORITHMS", 2, 1, 1.75, 3),
#     ("ITE 16", "INFORMATION MANAGEMENT", 2, 1, 2.00, 3),
#     ("PE 4", "MAJOR/ TEAM SPORTS", 2, 0, 1.00, 2),
#     ("TCW", "THE CONTEMPORARY WORLD", 3, 0, 2.00, 3),

#     # First Semester 2023-2024
#     ("ARTAPP", "ART APPRECIATION", 3, 0, 1.50, 3),
#     ("CSC 105", "ARCHITECTURE AND ORGANIZATION", 2, 1, 2.00, 3),
#     ("CSC 107", "SOFTWARE ENGINEERING 2", 2, 1, 1.25, 3),
#     ("CSC 108", "ALGORITHMS AND COMPLEXITY", 2, 1, 1.50, 3),
#     ("ETH", "ETHICS", 3, 0, 1.50, 3),
#     ("IT 107", "INFORMATION ASSURANCE AND SECURITY 1", 2, 1, 1.75, 3),
#     ("ITE 18", "APPLICATIONS DEVELOPMENT AND EMERGING TECHNOLOGIES", 2, 1, 1.00, 3),
#     ("LIE", "LIVING IN THE IT ERA", 2, 1, 1.50, 3),

#     # Second Semester 2023-2024
#     ("CSC 109", "AUTOMATA THEORY AND FORMAL LANGUAGES", 3, 0, 1.75, 3),
#     ("CSC 110", "OPERATING SYSTEM", 2, 1, 1.00, 3),
#     ("CSC 111", "COMPUTATIONAL SCIENCE", 2, 1, 1.25, 3),
#     ("CSC 123", "PARALLEL AND DISTRIBUTED COMPUTING", 2, 1, 1.25, 3),
#     ("CSC 198", "THESIS 1", 3, 0, 1.00, 3),
#     ("ITE 17", "ITE TECHNOPRENEURSHIP", 3, 0, 2.00, 3),
#     ("ITE 19", "ITE COMPETENCY APPRAISAL", 0, 1, 1.00, 1),

#     # First Semester 2024-2025
#     ("CSC 112", "PROGRAMMING LANGUAGES", 2, 1, 1.75, 3),
#     ("CSC 113", "NETWORKS AND COMMUNICATIONS", 2, 1, 1.75, 3),
#     ("CSC 120", "INTELLIGENT SYSTEMS", 2, 1, 1.50, 3),
#     ("IT 101", "HUMAN COMPUTER INTERACTION", 2, 1, 1.00, 3),


#     # Second Semester 2024-2025
#     ("CSC 198", "OJT/PRACTICUM", 3, 0, 1.25, 3),
# ]

# Initialize variables
total_grade_units = 0
total_units = 0

# Iterate through subjects
for subject in subjects:
    code, name, lec, lab, grade, units = subject
    
    # Skip NSTP and PASSED subjects
    if "NSTP" in code or grade == "PASSED":
        continue
    
    # Add to totals
    total_grade_units += grade * units
    total_units += units

# Compute GWA
if total_units > 0:
    gwa = total_grade_units / total_units
else:
    gwa = 0.0

# Display results
print(f"Total Grade Units: {total_grade_units}")
print(f"Total Units: {total_units}")
print(f"GWA: {gwa}")  # Rounded to 2 decimal places
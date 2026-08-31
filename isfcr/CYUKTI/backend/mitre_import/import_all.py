from mitre_import.import_techniques import import_techniques
from mitre_import.import_threat_actors import import_threat_actors
from mitre_import.import_courses import import_courses
from mitre_import.import_tools import import_tools
from mitre_import.import_malware import import_malware
from mitre_import.relationships import import_relationships

def main():
    print("\n========== MITRE ATT&CK Import ==========\n")
    print("[1/6] Importing Techniques...")
    import_techniques()
    print("[2/6] Importing Threat Actors...")
    import_threat_actors()
    print("[3/6] Importing Courses of Action...")
    import_courses()
    print("[4/6] Importing Tools...")
    import_tools()
    print("[5/6] Importing Malware...")
    import_malware()
    print("[6/6] Importing Relationships...")
    import_relationships()
    print("\n========== Import Complete ==========\n")
    

if __name__ == "__main__":
    main()
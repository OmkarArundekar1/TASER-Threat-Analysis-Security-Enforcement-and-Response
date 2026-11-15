#!/usr/bin/env python3
"""
apply_integration.py
====================
Automatically applies the transition learner integration to realtime_socgraph.py

This script modifies realtime_socgraph.py to add:
1. Import statement for transition_learner
2. Call to record_attacker_activity after campaign creation

Usage:
    python apply_integration.py [--dry-run] [--backup]

Options:
    --dry-run    Show what would be changed without modifying files
    --backup     Create backup before modifying (recommended)

Safety:
    - Creates backup by default
    - Checks if already integrated (won't duplicate)
    - Validates syntax after changes
"""

import sys
import os
import re
import shutil
from pathlib import Path


def find_realtime_socgraph():
    """Find realtime_socgraph.py in current directory or backend/"""
    candidates = [
        Path("realtime_socgraph.py"),
        Path("backend/realtime_socgraph.py"),
        Path("../backend/realtime_socgraph.py"),
    ]
    
    for path in candidates:
        if path.exists():
            return path.resolve()
    
    print("ERROR: Could not find realtime_socgraph.py")
    print("Looked in:")
    for path in candidates:
        print(f"  - {path.absolute()}")
    sys.exit(1)


def backup_file(filepath):
    """Create backup of file"""
    backup_path = Path(str(filepath) + ".backup")
    shutil.copy2(filepath, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def check_already_integrated(content):
    """Check if integration already exists"""
    has_import = "from transition_learner import" in content
    has_call = "transition_learner.record_attacker_activity" in content
    
    if has_import or has_call:
        print("⚠ Integration already present:")
        if has_import:
            print("  - transition_learner import found")
        if has_call:
            print("  - record_attacker_activity call found")
        return True
    
    return False


def add_import(content):
    """Add transition_learner import after other imports"""
    
    # Find a good place to insert after chain_updater import
    import_pattern = r'(from chain_updater import update_attack_chain\n)'
    
    if not re.search(import_pattern, content):
        print("⚠ Could not find chain_updater import as anchor point")
        print("  Will add after config imports instead")
        # Fallback: add after config imports
        import_pattern = r'(from config import[^\n]+\n(?:\s+[^\n]+\n)*)'
    
    new_import = r'\1from transition_learner import learner as transition_learner\n'
    
    modified = re.sub(import_pattern, new_import, content, count=1)
    
    if modified == content:
        print("✗ Could not find insertion point for import")
        return None
    
    print("✓ Added transition_learner import")
    return modified


def add_learner_call(content):
    """Add learner call after campaign creation"""
    
    # Find the campaign creation pattern
    # Looking for: transition = campaign_manager.get_or_create_campaign_with_transition(...)
    #              campaign_id = transition.campaign_id
    
    pattern = r'''(        transition = campaign_manager\.get_or_create_campaign_with_transition\(
            attacker_ip,
            victim_ip,
            mitre_id,
            event_time=event_time\.value
        \)
        campaign_id = transition\.campaign_id\n)'''
    
    # Simplified pattern that's more flexible
    simple_pattern = r'(campaign_id = transition\.campaign_id\n)'
    
    integration_code = '''
        # Record attacker activity for cross-campaign transition learning
        # This runs regardless of whether campaign continued or split,
        # ensuring the system learns from all attacker behavior.
        if mitre_id != "UNKNOWN":
            try:
                learned = transition_learner.record_attacker_activity(
                    attacker_ip=attacker_ip,
                    technique=mitre_id,
                    event_time=event_time.value
                )
                if learned:
                    print(f"[LEARNER] Learned new transition from {attacker_ip}")
            except Exception as e:
                print(f"[LEARNER ERROR] {e}")
'''
    
    # Try simple pattern first
    modified = re.sub(simple_pattern, r'\1' + integration_code, content, count=1)
    
    if modified == content:
        print("✗ Could not find insertion point for learner call")
        print("  Looking for: campaign_id = transition.campaign_id")
        return None
    
    print("✓ Added transition_learner.record_attacker_activity() call")
    return modified


def validate_syntax(content):
    """Validate Python syntax"""
    try:
        compile(content, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error after changes:")
        print(f"  Line {e.lineno}: {e.msg}")
        return False


def apply_integration(filepath, dry_run=False, backup=True):
    """Apply integration to realtime_socgraph.py"""
    
    print(f"\nProcessing: {filepath}")
    print("="*60)
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if check_already_integrated(content):
        response = input("\nFile appears already integrated. Continue anyway? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    # Create backup
    backup_path = None
    if backup and not dry_run:
        backup_path = backup_file(filepath)
    
    # Apply changes
    print("\nApplying changes...")
    
    # Add import
    modified = add_import(content)
    if modified is None:
        print("\n✗ Failed to add import")
        return False
    
    # Add learner call
    modified = add_learner_call(modified)
    if modified is None:
        print("\n✗ Failed to add learner call")
        return False
    
    # Validate syntax
    print("\nValidating syntax...")
    if not validate_syntax(modified):
        print("\n✗ Integration failed syntax validation")
        if backup_path:
            print(f"Original file preserved at: {backup_path}")
        return False
    
    print("✓ Syntax validation passed")
    
    # Show diff
    print("\nChanges summary:")
    print(f"  - Added 1 import line")
    print(f"  - Added learner call (~10 lines)")
    print(f"  - Total size: {len(content)} → {len(modified)} bytes")
    
    if dry_run:
        print("\n[DRY RUN] Would write changes to disk")
        return True
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(modified)
    
    print(f"\n✓ Successfully integrated into {filepath}")
    if backup_path:
        print(f"  Backup saved: {backup_path}")
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply transition learner integration to realtime_socgraph.py"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Don't create backup (not recommended)"
    )
    parser.add_argument(
        'file',
        nargs='?',
        help="Path to realtime_socgraph.py (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Find file
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
    else:
        filepath = find_realtime_socgraph()
    
    print("CYUKTI Transition Learner Integration")
    print("="*60)
    print(f"File: {filepath}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY CHANGES'}")
    print(f"Backup: {'No' if args.no_backup else 'Yes'}")
    print()
    
    # Apply
    backup = not args.no_backup
    success = apply_integration(filepath, dry_run=args.dry_run, backup=backup)
    
    if success:
        print("\n" + "="*60)
        print("✓ Integration completed successfully!")
        print("\nNext steps:")
        print("1. Review the changes in the file")
        print("2. Test: python realtime_socgraph.py")
        print("3. Monitor logs for [LEARNER] messages")
        print("4. Run: python test_critical_fixes.py")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ Integration failed")
        print("\nYou may need to apply changes manually:")
        print("See integration_patch.py for code snippets")
        sys.exit(1)


if __name__ == "__main__":
    main()

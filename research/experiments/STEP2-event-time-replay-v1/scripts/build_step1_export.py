"""Rebuild the Step-1-accepted backend (HEAD + Step 1 edits only) for the
Step 2 before/after comparison. Usage: build_step1_export.py <head_backend_dir> <current_backend_dir> <out_dir>"""
import re, shutil, sys

head, current, out = sys.argv[1:4]
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(head, out)
# Unchanged by Step 2 -> identical to their Step 1 versions:
for f in ("campaign_context.py", "chain_updater.py"):
    shutil.copy(f"{current}/{f}", f"{out}/{f}")


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        pattern = "\n".join(re.escape(l.rstrip()) + "[ \t]*" if l.strip() else "[ \t]*" for l in old.split("\n"))
        n = len(re.findall(pattern, s)); assert n == 1, (p, n, old[:60])
        s = re.sub(pattern, lambda m: new, s)
    open(p, "w", encoding="utf-8", newline="").write(s)


patch(f"{out}/campaign_manager.py", [
    ("from campaign_context import CampaignContext\n",
     "from campaign_context import CampaignContext, TransitionContext\n"),
    ('''    def activate_campaign(
        self,
        context,
        technique
    ):

        context.status = ACTIVE''', '''    def activate_campaign(
        self,
        context,
        technique
    ):
        return self.activate_campaign_with_transition(
            context,
            technique
        ).campaign_id

    def activate_campaign_with_transition(
        self,
        context,
        technique
    ):
        segment_start = bool(context.skip_next_chain_update)
        transition = TransitionContext(
            campaign_id=context.campaign_id,
            current_technique=technique,
            previous_technique=(
                None if segment_start
                else (context.last_technique or None)
            ),
            segment_start=segment_start,
        )
        context.skip_next_chain_update = False

        context.status = ACTIVE'''),
    ('''        print(f"[CAMPAIGN] {context.campaign_id} ACTIVE")
        return context.campaign_id

    def get_or_create_campaign(
        self,
        attacker_ip,
        victim_ip,
        current_technique
    ):
        context = self.resolve_campaign(
            attacker_ip,
            victim_ip,
            current_technique
        )

        campaign_id = self.activate_campaign(
            context,
            current_technique
        )

        return campaign_id''', '''        print(f"[CAMPAIGN] {context.campaign_id} ACTIVE")
        return transition

    def get_or_create_campaign(
        self,
        attacker_ip,
        victim_ip,
        current_technique
    ):
        return self.get_or_create_campaign_with_transition(
            attacker_ip,
            victim_ip,
            current_technique
        ).campaign_id

    def get_or_create_campaign_with_transition(
        self,
        attacker_ip,
        victim_ip,
        current_technique
    ):
        context = self.resolve_campaign(
            attacker_ip,
            victim_ip,
            current_technique
        )

        return self.activate_campaign_with_transition(
            context,
            current_technique
        )'''),
])
patch(f"{out}/realtime_socgraph.py", [
    ('''        campaign_id = campaign_manager.get_or_create_campaign(
            attacker_ip,
            victim_ip,
            mitre_id
        )''', '''        transition = campaign_manager.get_or_create_campaign_with_transition(
            attacker_ip,
            victim_ip,
            mitre_id
        )
        campaign_id = transition.campaign_id'''),
    ('''        update_attack_chain(
            campaign_id,
            mitre_id
        )''', '''        update_attack_chain(
            campaign_id,
            mitre_id,
            transition
        )'''),
])
print("step1 export built at", out)

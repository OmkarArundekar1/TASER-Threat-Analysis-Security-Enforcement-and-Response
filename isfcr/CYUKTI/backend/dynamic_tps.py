from config import TPS_MAP


class DynamicTPSEngine:
    def __init__(self):
        self.enabled = True

    def get_tps(
        self,
        stage,
        technique_id=None,
        campaign_id=None,
        attacker_ip=None,
    ):

        base_score = TPS_MAP.get(stage, 0)

        return base_score


dynamic_tps = DynamicTPSEngine()
class AttributionSimilarity:

    @staticmethod
    def coverage(
        observed: set,
        historical: set,
    ):

        if not observed:
            return 0.0

        return len(
            observed & historical
        ) / len(observed)

    @staticmethod
    def precision(
        observed: set,
        historical: set,
    ):

        if not historical:
            return 0.0

        return len(
            observed & historical
        ) / len(historical)

    @staticmethod
    def lcs(
        seq1,
        seq2,
    ):

        m = len(seq1)
        n = len(seq2)

        dp = [
            [0] * (n + 1)
            for _ in range(m + 1)
        ]

        for i in range(m):
            for j in range(n):

                if seq1[i] == seq2[j]:

                    dp[i + 1][j + 1] = (
                        dp[i][j] + 1
                    )

                else:

                    dp[i + 1][j + 1] = max(
                        dp[i][j + 1],
                        dp[i + 1][j],
                    )

        return dp[m][n]

    @classmethod
    def chain_similarity(
        cls,
        campaign_chain,
        historical_chain,
    ):

        if not campaign_chain:
            return 0.0

        lcs = cls.lcs(
            campaign_chain,
            historical_chain,
        )

        return lcs / len(campaign_chain)
        
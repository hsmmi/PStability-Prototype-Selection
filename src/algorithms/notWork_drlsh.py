import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from src.algorithms.base import BaseAlgorithm


class DRLSH(BaseAlgorithm):
    def __init__(
        self,
        M=25,
        L=10,
        W=1,
        ST=9,
    ):
        super().__init__()
        self.M = M
        self.L = L
        self.W = W
        self.ST = ST

    def select(self) -> np.ndarray:
        Data = np.hstack((self.X, self.y.reshape(-1, 1)))
        Classes = np.unique(Data[:, -1])

        # Normalizing the data between 0 and 1
        scaler = MinMaxScaler()
        Data[:, :-1] = scaler.fit_transform(Data[:, :-1])

        Dimension = Data.shape[1] - 1  # Number of features
        M = self.M  # Number of hash functions in each table
        L = self.L  # Number of hash tables
        W = self.W  # Bucket size
        Frequency_Neighbors_Threshold = self.ST

        np.random.seed(0)  # Reset Random Number Generator
        a = np.random.normal(0, 1, (M * L, Dimension))  # Generate a in floor((ax+b)/W)
        b = W * np.random.rand(M * L, 1)  # Generate b in floor((ax+b)/W)

        # Calculating the buckets of samples
        Bucket_Index_Decimal_All = np.zeros((self.L, Data.shape[0]), dtype=np.int32)
        for i in range(self.L):
            j = slice((i * self.M), ((i + 1) * self.M))
            Bucket_Index = np.floor((a[j] @ Data[:, :-1].T + b[j]) / self.W).astype(
                np.int16
            )
            BI = Bucket_Index

            Bucket_Index_uniqued = np.unique(BI.T, axis=0).T

            # For splitting BI matrix into PartsNo to make the search faster
            ss = 0
            vectLength = BI.shape[1]
            splitsize = vectLength

            idxs = slice(0, splitsize)
            BI_Part = BI[:, idxs]
            Bucket_Index_Decimal = np.where(
                (BI_Part.T[:, None] == Bucket_Index_uniqued.T[None, :]).all(axis=2)
            )[1]
            Bucket_Index_Decimal = Bucket_Index_Decimal.astype(np.int32)
            Bucket_Index_Decimal_All[i, ss : ss + Bucket_Index_Decimal.shape[0]] = (
                Bucket_Index_Decimal
            )
            ss += Bucket_Index_Decimal.shape[0]

        Removed_Samples_Index_ALL = np.zeros(Data.shape[0], dtype=np.int32)
        RSC = 0
        for classID in Classes:
            All_Indexes = np.where(Data[:, -1] == classID)[0]
            Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All[:, All_Indexes]
            iii = 0
            TRS = Data.shape[0] + 1
            Temporal_Removed_Samples = [TRS]
            while iii < len(All_Indexes):
                Current_Sample_Bucket_Index_Decimal = Bucket_Index_Decimal_All_Class[
                    :, iii
                ]
                Bucket_Index_Decimal_All_Class[:, iii] = -1
                Number_of_Common_Buckets = np.sum(
                    Bucket_Index_Decimal_All_Class
                    == Current_Sample_Bucket_Index_Decimal[:, np.newaxis],
                    axis=0,
                )
                Index_Neighbors = Number_of_Common_Buckets > 0
                Frequency_Neighbors = Number_of_Common_Buckets[Index_Neighbors]
                uniqued_Neighbors = All_Indexes[Index_Neighbors]
                Bucket_Index_Decimal_All_Class[:, iii] = (
                    Current_Sample_Bucket_Index_Decimal
                )
                Removed_Samples_Current = uniqued_Neighbors[
                    Frequency_Neighbors >= Frequency_Neighbors_Threshold
                ]
                Removed_Samples_Index_ALL[RSC : RSC + len(Removed_Samples_Current)] = (
                    Removed_Samples_Current
                )
                RSC += len(Removed_Samples_Current)

                Temporal_Removed_Samples.extend(Removed_Samples_Current)
                if (
                    len(All_Indexes) > iii + 1
                    and min(Temporal_Removed_Samples) <= All_Indexes[iii + 1]
                ) or (iii > 2000):
                    aa = np.isin(All_Indexes, Temporal_Removed_Samples)
                    All_Indexes = All_Indexes[~aa]
                    Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All_Class[
                        :, ~aa
                    ]
                    Temporal_Removed_Samples = [TRS]
                    All_Indexes = All_Indexes[iii:]
                    Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All_Class[
                        :, iii:
                    ]
                    iii = 0
                iii += 1

        Removed_Samples_Index_ALL = np.unique(Removed_Samples_Index_ALL)
        Removed_Samples_Index_ALL = Removed_Samples_Index_ALL[
            Removed_Samples_Index_ALL != 0
        ]
        Selected_Data_Index = np.setdiff1d(
            np.arange(Data.shape[0]), Removed_Samples_Index_ALL
        )

        return Selected_Data_Index


# Usage example:
# M = 25
# L = 10
# w = 1
# ST = 9
# drlsh = DRLSH(M, L, w, ST)
# drlsh.fit(X, y)
# selected_data = drlsh.transform(X, y)

# E-HilbertEPIs
E-HilbertEPIs: a novel method for predicting enhancer-promoter interactions via Hilbert Curve and transfer learning
# 1.Sequence data used in this study：
In our study, we used the same dataset from TargetFinder(https://github.com/shwhalen/targetfinder) as the original EPIs dataset, consisting of six cell lines: GM12878, HUVEC, HeLa-S3, K562, NHEK and IMR90. The dataset of each cell line contains enhancer sequences, promoter sequences, positive samples and negative samples of EPIs. 
# 2. Data prepocess and Hilbert Curve encoding
All sequences are unified into fixed length. The length of enhancer sequence is 3000 base pairs(bp), while that of promoter is 2000bp. In each cell line, the ratio of negative samples to positive samples is 20:1. The class imbalance problem is existed in our dataset exists, so we use two methods, over-sampling and under-sampling to achieve the balanced dataset. 

To address the spatial limitation, Hilbert Curve is a classic space-filling curve, proposed to encode enhancer and promoter sequences which can convert a one-dimensional sequence into a three-dimensional matrix vector, so it can represent enhancer-promoter long range interaction and the spatial structure.

# 3. Usage
To run the tool, open: E-HilbertEPIs.py.

# 4. Output

# 5. Disclaimer
The executable software and the source code of E-HilbertEPIs is distributed free of charge as it is to any non-commercial users. The authors hold no liabilities to the performance of the program.

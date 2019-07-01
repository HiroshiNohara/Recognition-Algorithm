# Recognition-Algorithm(JAVA)

## Algorithms

- Original Local Binary Patterns (Original-LBP)
- Local Binary Patterns (LBP)
- Dual-Cross Patterns (DCP)
- Local Ternary Patterns (LTP)

## algorithm.java

>Encapsulating the Original LBP algorithm, DCP algorithm, etc, the texture features of the specified image can be obtained by calling any one of  `public Mat LBP(Mat src)`, `public Mat DCP1(Mat src, int Rin, int Rex)` or `public Mat DCP2(Mat src, int Rin, int Rex)`, etc.

## histogram.java

>Calculating a one-dimensional histogram of a feature image.

## traintest.java

>Train the training set according to the specified CSV file and test its recognition rate.

## entrance.java

>Program entry.

## Note

- The default grid_x and grid_y are taken as 1. If you want better recognition accurcy, set the value of grid_x and grid_y to 8. But note that this will bring more calculation work and program running time.
- When using the DCP algorithm, it is recommended to set Rin to 1 and Rex to 4.
- The LTP algorithm has two modes, adaptive threshold and non-adaptive threshold, depending on the adaptation value of the adaptation.

## Reference

* T. Ojala, M. Pietikäinen, and D. Harwood (1996), "A Comparative Study of Texture Measures with Classification Based on Feature Distributions", Pattern Recognition, vol. 29, pp. 51-59.
* Changxing Ding, Jonghyun Choi, Dacheng Tao, and Larry S. Davis, `Multi-Directional Multi-Level Dual-Cross Patterns for Robust Face Recognition', Vol.38, No.3, pp.518-531, IEEE TPAMI 2016.
* Tan X , Triggs B . Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions[J]. IEEE Transactions on Image Processing, 2010, 19(6):1635-1650.


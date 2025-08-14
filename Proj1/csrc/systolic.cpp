#include <vector>
#include <iostream>
#include <cassert>

typedef std::vector<std::vector<std::vector<std::vector<int>>>> Tensor4D;
typedef std::vector<std::vector<int>> Matrix;

struct Config {
    int IMG_C; // Number of image channels
    int IMG_W; // Image width
    int IMG_H; // Image height
    int FILTER_SIZE; // Size of the convolution kernel
    int FILTER_NUM; // Number of filters
};


Matrix img2col(const Tensor4D& input, const Config& config) {
    int OUT_H = config.IMG_H - config.FILTER_SIZE + 1;
    int OUT_W = config.IMG_W - config.FILTER_SIZE + 1;
    int FILTER_SIZE = config.FILTER_SIZE;
    int IMG_C = config.IMG_C;

    int M = OUT_H * OUT_W;
    int N = FILTER_SIZE * FILTER_SIZE * IMG_C;

    Matrix X(M, std::vector<int>(N, 0));

    for (int h = 0; h < OUT_H; ++h) {
        for (int w = 0; w < OUT_W; ++w) {
          // For each unit in the output matrix, it results from a convolution, which we need to converts into a dot product of two vector
            int row = h * OUT_W + w;
            // one row of transformed matrix represents C reception fields flatterned and concatenate together 
            for (int c = 0; c < IMG_C; ++c) {
                for (int fh = 0; fh < FILTER_SIZE; ++fh) {
                    for (int fw = 0; fw < FILTER_SIZE; ++fw) {
                        int ih = h + fh;
                        int iw = w + fw;
                        if (ih < config.IMG_H && iw < config.IMG_W) {
                            X[row][c * FILTER_SIZE * FILTER_SIZE + fh * FILTER_SIZE + fw] = input[0][c][ih][iw];
                        }
                    }
                }
            }
        }
    }

    return X;
}

class PE {
public:
    PE() : y_out(0), x_in(0), w_in(0), x_out(0), w_out(0) {}

    void compute() {
        y_out += x_in * w_in;
    }

    void shift_inputs(int new_x_in, int new_w_in) {
        x_out = x_in;
        w_out = w_in;
        x_in = new_x_in;
        w_in = new_w_in;
    }

    int get_output() const {
        return y_out;
    }

    int get_x_out() const {
        return x_out;
    }

    int get_w_out() const {
        return w_out;
    }

    void reset() {
        y_out = 0;
        x_in = 0;
        w_in = 0;
        x_out = 0;
        w_out = 0;
    }

private:
    int y_out;
    int x_in, w_in;
    int x_out, w_out;
};

class SystolicArray {
public:
    SystolicArray(int M, int N, int K) : M(M), N(N), K(K), PEs(M, std::vector<std::vector<PE>>(N, std::vector<PE>(K))) {}

    void compute(const Matrix& X, const Matrix& W) {
        for (int t = 0; t < N; ++t) {
            for (int m = 0; m < M; ++m) {
                for (int k = 0; k < K; ++k) {
                    int x_in = (m > 0) ? PEs[m-1][t][k].get_x_out() : X[m][t];
                    int w_in = (k > 0) ? PEs[m][t][k-1].get_w_out() : W[t][k];
                    PEs[m][t][k].shift_inputs(x_in, w_in);
                    PEs[m][t][k].compute();
                }
            }
        }
    }

    void my_compute(const Matrix& X, const Matrix& W) {
      std::vector<std::vector<PE>> my_pe(N, std::vector<PE>(K)) ; 

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
          my_pe[i][j].reset();
        }
      }

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
          // 没有延时
            int x_in = (i > 0) ? my_pe[i-1][j].get_x_out() : X[i][j];
            int w_in = (j > 0) ? my_pe[i][j-1].get_w_out() : W[j][i];
            my_pe[i][j].shift_inputs(x_in, w_in);
            my_pe[i][j].compute();
        }
      }
    }

    Matrix get_output() const {
        Matrix Y(M, std::vector<int>(K, 0));
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                Y[m][k] = PEs[m][N-1][k].get_output();
            }
        }
        return Y;
    }

    void reset() {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    PEs[m][n][k].reset();
                }
            }
        }
    }

private:
    int M, N, K;
    std::vector<std::vector<std::vector<PE>>> PEs;
};


int main() {
    Config config = {1, 4, 4, 3, 2}; // Example configuration

    // Example input tensor (N, C, H, W)
    Tensor4D input = {
        {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}
            }
        }
    };

    // Example weight tensor (FILTER_NUM, IMG_C, FILTER_SIZE, FILTER_SIZE)
    Tensor4D weights = {
        {
            {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
            }
        },
        {
            {
                {10, 11, 12},
                {13, 14, 15},
                {16, 17, 18}
            }
        }
    };

    // Convert weights to matrix format
    Matrix W(config.FILTER_SIZE * config.FILTER_SIZE * config.IMG_C, std::vector<int>(config.FILTER_NUM, 0));
    for (int f = 0; f < config.FILTER_NUM; ++f) {
        for (int c = 0; c < config.IMG_C; ++c) {
            for (int fh = 0; fh < config.FILTER_SIZE; ++fh) {
                for (int fw = 0; fw < config.FILTER_SIZE; ++fw) {
                    W[c * config.FILTER_SIZE * config.FILTER_SIZE + fh * config.FILTER_SIZE + fw][f] = weights[f][c][fh][fw];
                }
            }
        }
    }
    std::cout << "weight matrix after transforming" << std::endl;
    for (const auto& row : W) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Input before image to col:" << std::endl;
    // Perform img2col
    Matrix X = img2col(input, config);
    // Print output
    std::cout << "Output Matrix X after image to col:" << std::endl;
    for (const auto& row : X) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Initialize systolic array
    SystolicArray systolic_array(config.IMG_H * config.IMG_W, config.FILTER_SIZE * config.FILTER_SIZE * config.IMG_C, config.FILTER_NUM);

    // Compute using systolic array
    systolic_array.compute(X, W);

    // Get output
    Matrix Y = systolic_array.get_output();

    // Print output
    std::cout << "Output Matrix Y:" << std::endl;
    for (const auto& row : Y) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

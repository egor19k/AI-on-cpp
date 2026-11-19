class N {
public:
    double* I;
    double** H;
    double* O;
    double*** W;
    double lr;
    int layers;

    int inSize_, hidSize_, outSize_;
    unsigned long rngState;

    N(int i, int h, int layers_, int o) {
        layers = layers_;
        lr = 0.4;
        inSize_ = i;
        hidSize_ = h;
        outSize_ = o;
        rngState = 1;

        I = new double[i];
        O = new double[o];
        H = new double*[layers];
        for (int L = 0; L < layers; L++) H[L] = new double[h];

        W = new double**[layers + 1];
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? i : h;
            int cols = (L == layers) ? o : h;
            W[L] = new double*[rows];
            for (int a = 0; a < rows; a++) {
                W[L][a] = new double[cols];
                for (int b = 0; b < cols; b++) {
                    rngState = rngState * 1664525UL + 1013904223UL;
                    W[L][a][b] = ((double)(rngState >> 16) / 65535.0) * 2 - 1;
                }
            }
        }
    }

    ~N() {
        delete[] I;
        delete[] O;
        for (int L = 0; L < layers; L++) delete[] H[L];
        delete[] H;
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize_ : hidSize_;
            for (int a = 0; a < rows; a++) delete[] W[L][a];
            delete[] W[L];
        }
        delete[] W;
    }

    double s(double x) {
        if (x > 50.0) x = 50.0;
        if (x < -50.0) x = -50.0;
        double term = 1.0, sum = 1.0;
        for (int n = 1; n < 30; n++) {
            term *= (-x) / n;
            sum += term;
            if (term < 1e-15 && term > -1e-15) break;
        }
        return 1.0 / (1.0 + sum);
    }

    double sd(double x) { return x * (1.0 - x); }

    double tanh(double x) {
        double y = 2.0 * x;
        if (y > 50.0) y = 50.0;
        if (y < -50.0) y = -50.0;
        double term = 1.0, sum = 1.0;
        for (int n = 1; n < 30; n++) {
            term *= y / n;
            sum += term;
            if (term < 1e-15 && term > -1e-15) break;
        }
        return (sum - 1.0) / (sum + 1.0);
    }

    double tanh_d(double x) { return 1.0 - (x * x); }

    double* softmax(double* z, int n) {
        double max = z[0];
        for (int i = 1; i < n; i++) if (z[i] > max) max = z[i];

        double* out = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double y = z[i] - max;
            if (y > 50.0) y = 50.0;
            if (y < -50.0) y = -50.0;
            double term = 1.0, e = 1.0;
            for (int k = 1; k < 30; k++) {
                term *= y / k;
                e += term;
                if (term < 1e-15 && term > -1e-15) break;
            }
            out[i] = e;
            sum += e;
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    double neuron(double* X, double* Wv, int n) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum = sum + X[i] * Wv[i];
        return sum;
    }

    double* forward(double* X, int xLen) {
        for (int i = 0; i < inSize_; i++) {
            I[i] = (i < xLen) ? X[i] : 0;
        }

        double* prev = I;
        int prevLen = inSize_;
        for (int L = 0; L < layers + 1; L++) {
            double* next_layer = (L == layers) ? O : H[L];
            int nextLen = (L == layers) ? outSize_ : hidSize_;
            for (int j = 0; j < nextLen; j++) {
                double sum = 0;
                for (int i = 0; i < prevLen; i++) {
                    sum = sum + prev[i] * W[L][i][j];
                }
                next_layer[j] = s(sum);
            }
            prev = next_layer;
            prevLen = nextLen;
        }

        double* out = new double[outSize_];
        for (int i = 0; i < outSize_; i++) out[i] = O[i];
        return out;
    }

    double backprop(double* T, int tLen) {
        double** gradients = new double*[layers + 1];

        double* OE = new double[outSize_];
        for (int k = 0; k < outSize_; k++) {
            OE[k] = (k < tLen ? T[k] : 0) - O[k];
        }

        double* OD = new double[outSize_];
        for (int k = 0; k < outSize_; k++) {
            OD[k] = OE[k] * sd(O[k]);
        }
        gradients[layers] = OD;

        for (int L = layers - 1; L >= 0; L--) {
            double* current;
            if (L == layers - 1) current = O;
            else current = H[L + 1];
            int curLen = (L == layers - 1) ? outSize_ : hidSize_;

            double* prev;
            if (L == 0) prev = I;
            else prev = H[L - 1];
            int prevLen = (L == 0) ? inSize_ : hidSize_;

            double* grad = new double[prevLen];

            for (int j = 0; j < prevLen; j++) {
                double sum = 0;
                for (int k = 0; k < curLen; k++) {
                    if (j < hidSize_ && k < hidSize_)
                        sum = sum + gradients[L + 1][k] * W[L + 1][j][k];
                }
                grad[j] = sum * sd(prev[j]);
            }
            gradients[L] = grad;
        }

        for (int L = 0; L < layers + 1; L++) {
            double* prev;
            if (L == 0) prev = I;
            else prev = H[L - 1];
            int prevLen = (L == 0) ? inSize_ : hidSize_;

            double* next_layer;
            if (L == layers) next_layer = O;
            else next_layer = H[L];
            int nextLen = (L == layers) ? outSize_ : hidSize_;

            for (int i = 0; i < prevLen; i++) {
                for (int j = 0; j < nextLen; j++) {
                    if (i < prevLen && j < nextLen && j < prevLen) {
                        W[L][i][j] = W[L][i][j] + lr * gradients[L][j] * prev[i];
                    }
                }
            }
        }

        double err = 0;
        for (int k = 0; k < outSize_; k++) {
            err = err + OE[k] * OE[k];
        }

        for (int L = 0; L < layers + 1; L++) delete[] gradients[L];
        delete[] gradients;
        delete[] OE;

        return err / outSize_;
    }

    void train(double*** D, int E) {
        int samples = len(D);
        for (int e = 0; e < E; e++) {
            for (int s = 0; s < samples; s++) {
                forward(D[s][0], len(D[s][0]));
                backprop(D[s][1], len(D[s][1]));
            }
        }
    }

    void train(double*** D) {
        train(D, 4000);
    }

    char* save() {
        int bufSize = 256;
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize_ : hidSize_;
            int cols = (L == layers) ? outSize_ : hidSize_;
            bufSize += rows * cols * 24;
        }
        char* buf = new char[bufSize];
        int pos = 0;

        pos += intToStr(buf + pos, layers); buf[pos++] = '|';
        pos += intToStr(buf + pos, inSize_); buf[pos++] = '|';
        pos += intToStr(buf + pos, hidSize_); buf[pos++] = '|';
        pos += intToStr(buf + pos, outSize_); buf[pos++] = '|';

        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize_ : hidSize_;
            int cols = (L == layers) ? outSize_ : hidSize_;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    pos += doubleToStr(buf + pos, W[L][i][j]);
                    if (!(L == layers && i == rows - 1 && j == cols - 1)) {
                        buf[pos++] = ',';
                    }
                }
            }
        }
        buf[pos] = 0;
        return buf;
    }

    void load(const char* data) {
        int p = 0;
        int vals[4];
        int vi = 0;
        while (data[p] && vi < 4) {
            int sign = 1;
            if (data[p] == '-') { sign = -1; p++; }
            int v = 0;
            while (data[p] >= '0' && data[p] <= '9') { v = v * 10 + (data[p] - '0'); p++; }
            vals[vi++] = v * sign;
            if (data[p] == '|') p++;
        }
        if (vi != 4) return;

        int loadedLayers = vals[0];
        int inputSize = vals[1];
        int hiddenSize = vals[2];
        int outputSize = vals[3];

        layers = loadedLayers;
        inSize_ = inputSize;
        hidSize_ = hiddenSize;
        outSize_ = outputSize;

        I = new double[inputSize];
        O = new double[outputSize];
        H = new double*[loadedLayers];
        for (int L = 0; L < loadedLayers; L++) H[L] = new double[hiddenSize];

        W = new double**[loadedLayers + 1];
        for (int L = 0; L < loadedLayers + 1; L++) {
            int rows = (L == 0) ? inputSize : hiddenSize;
            int cols = (L == loadedLayers) ? outputSize : hiddenSize;
            W[L] = new double*[rows];
            for (int i = 0; i < rows; i++) {
                W[L][i] = new double[cols];
                for (int j = 0; j < cols; j++) {
                    int sign = 1;
                    if (data[p] == '-') { sign = -1; p++; }
                    double v = 0.0;
                    while (data[p] >= '0' && data[p] <= '9') { v = v * 10.0 + (data[p] - '0'); p++; }
                    if (data[p] == '.') {
                        p++;
                        double f = 0.1;
                        while (data[p] >= '0' && data[p] <= '9') { v += (data[p] - '0') * f; f *= 0.1; p++; }
                    }
                    W[L][i][j] = v * sign;
                    if (data[p] == ',') p++;
                }
            }
        }
    }

private:
    static int len(double*** D) {
        int n = 0;
        while (D[n] != nullptr) n++;
        return n;
    }

    static int len(double** D) {
        int n = 0;
        while (D[n] != nullptr) n++;
        return n;
    }

    static int len(double* D) {
        int n = 0;
        while (D[n] != 0.0) n++;
        return n;
    }

    static int intToStr(char* buf, int v) {
        if (v == 0) { buf[0] = '0'; return 1; }
        int n = 0;
        bool neg = v < 0;
        if (neg) v = -v;
        char tmp[12];
        int t = 0;
        while (v > 0) { tmp[t++] = char('0' + v % 10); v /= 10; }
        if (neg) buf[n++] = '-';
        for (int i = t - 1; i >= 0; i--) buf[n++] = tmp[i];
        return n;
    }

    static int doubleToStr(char* buf, double v) {
        int n = 0;
        if (v < 0) { buf[n++] = '-'; v = -v; }
        long long ip = (long long)v;
        double frac = v - (double)ip;
        n += intToStr(buf + n, (int)ip);
        buf[n++] = '.';
        frac *= 1000000.0;
        long long fp = (long long)(frac + 0.5);
        char tmp[8];
        int t = 0;
        if (fp == 0) tmp[t++] = '0';
        else while (fp > 0) { tmp[t++] = char('0' + fp % 10); fp /= 10; }
        while (t < 6) tmp[t++] = '0';
        for (int i = t - 1; i >= 0; i--) buf[n++] = tmp[i];
        return n;
    }
};
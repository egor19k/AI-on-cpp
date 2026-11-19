class N {
public:
    float* I;
    float** H;
    float* O;
    float*** W;

    float lr;
    int layers;
    int inSize, hidSize, outSize;

    float** gradBuf;
    float* OE;
    float* OD;

    unsigned long rngState;

    N(int i, int h, int l, int o) {
        layers = l;
        lr = 0.4f;
        inSize = i;
        hidSize = h;
        outSize = o;
        rngState = 1;

        I = new float[i];
        O = new float[o];

        H = new float*[l];
        for (int L = 0; L < l; L++) H[L] = new float[h];

        W = new float**[l + 1];
        for (int L = 0; L < l + 1; L++) {
            int rows = (L == 0) ? i : h;
            int cols = (L == l) ? o : h;
            W[L] = new float*[rows];
            for (int a = 0; a < rows; a++) {
                W[L][a] = new float[cols];
                for (int b = 0; b < cols; b++) {
                    rngState = rngState * 1664525UL + 1013904223UL;
                    W[L][a][b] = ((float)(rngState >> 16) / 65535.0f) * 2.0f - 1.0f;
                }
            }
        }

        gradBuf = new float*[l + 1];
        for (int L = 0; L < l + 1; L++) {
            int sz = (L == 0) ? i : h;
            if (L == l) sz = o;
            gradBuf[L] = new float[sz];
        }
        OE = new float[o];
        OD = new float[o];
    }

    ~N() {
        delete[] I;
        delete[] O;
        for (int L = 0; L < layers; L++) delete[] H[L];
        delete[] H;
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            for (int a = 0; a < rows; a++) delete[] W[L][a];
            delete[] W[L];
        }
        delete[] W;
        for (int L = 0; L < layers + 1; L++) delete[] gradBuf[L];
        delete[] gradBuf;
        delete[] OE;
        delete[] OD;
    }

    float s(float x) {
        if (x > 50.0f) x = 50.0f;
        if (x < -50.0f) x = -50.0f;
        float term = 1.0f, sum = 1.0f;
        for (int n = 1; n < 25; n++) {
            term *= (-x) / n;
            sum += term;
            if (term < 1e-7f && term > -1e-7f) break;
        }
        return 1.0f / (1.0f + sum);
    }

    float sd(float x) { return x * (1.0f - x); }

    float tanh_(float x) {
        float y = 2.0f * x;
        if (y > 50.0f) y = 50.0f;
        if (y < -50.0f) y = -50.0f;
        float term = 1.0f, sum = 1.0f;
        for (int n = 1; n < 25; n++) {
            term *= y / n;
            sum += term;
            if (term < 1e-7f && term > -1e-7f) break;
        }
        return (sum - 1.0f) / (sum + 1.0f);
    }

    float tanh_d(float x) { return 1.0f - (x * x); }

    void softmax(float* z, int n, float* out) {
        float max = z[0];
        for (int i = 1; i < n; i++) if (z[i] > max) max = z[i];
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float y = z[i] - max;
            if (y > 50.0f) y = 50.0f;
            if (y < -50.0f) y = -50.0f;
            float term = 1.0f, e = 1.0f;
            for (int k = 1; k < 25; k++) {
                term *= y / k;
                e += term;
                if (term < 1e-7f && term > -1e-7f) break;
            }
            out[i] = e;
            sum += e;
        }
        for (int i = 0; i < n; i++) out[i] /= sum;
    }

    float neuron(float* X, float* Wv, int n) {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += X[i] * Wv[i];
        return sum;
    }

    float* forward(float* X, int xLen) {
        for (int i = 0; i < inSize; i++) I[i] = (i < xLen) ? X[i] : 0;

        float* prev = I;
        int prevLen = inSize;

        for (int L = 0; L < layers + 1; L++) {
            int nextLen = (L == layers) ? outSize : hidSize;
            float* next_layer = (L == layers) ? O : H[L];
            for (int j = 0; j < nextLen; j++) {
                float sum = 0;
                for (int i = 0; i < prevLen; i++) sum += prev[i] * W[L][i][j];
                next_layer[j] = s(sum);
            }
            prev = next_layer;
            prevLen = nextLen;
        }
        return O;
    }

    float backprop(float* T, int tLen) {
        for (int k = 0; k < outSize; k++) {
            OE[k] = (k < tLen ? T[k] : 0) - O[k];
            OD[k] = OE[k] * sd(O[k]);
        }
        for (int k = 0; k < outSize; k++) gradBuf[layers][k] = OD[k];

        for (int L = layers - 1; L >= 0; L--) {
            float* current = (L == layers - 1) ? O : H[L + 1];
            int curLen = (L == layers - 1) ? outSize : hidSize;
            float* prev = (L == 0) ? I : H[L - 1];
            int prevLen = (L == 0) ? inSize : hidSize;

            for (int j = 0; j < prevLen; j++) {
                float sum = 0;
                for (int k = 0; k < curLen; k++)
                    if (j < hidSize && k < hidSize)
                        sum += gradBuf[L + 1][k] * W[L + 1][j][k];
                gradBuf[L][j] = sum * sd(prev[j]);
            }
        }

        for (int L = 0; L < layers + 1; L++) {
            float* prev = (L == 0) ? I : H[L - 1];
            int prevLen = (L == 0) ? inSize : hidSize;
            float* next_layer = (L == layers) ? O : H[L];
            int nextLen = (L == layers) ? outSize : hidSize;
            for (int i = 0; i < prevLen; i++)
                for (int j = 0; j < nextLen; j++)
                    W[L][i][j] += lr * gradBuf[L][j] * prev[i];
        }

        float err = 0;
        for (int k = 0; k < outSize; k++) err += OE[k] * OE[k];
        return err / outSize;
    }

    void train(float*** X, int* xLens, float*** T, int* tLens, int samples, int E) {
        for (int e = 0; e < E; e++)
            for (int s = 0; s < samples; s++) {
                forward(X[s], xLens[s]);
                backprop(T[s], tLens[s]);
            }
    }

    void train(float*** X, int* xLens, float*** T, int* tLens, int samples) {
        train(X, xLens, T, tLens, samples, 4000);
    }

    char* save() {
        int bufSize = 64;
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            int cols = (L == layers) ? outSize : hidSize;
            bufSize += rows * cols * 16;
        }
        char* buf = new char[bufSize];
        int pos = 0;

        pos += intToStr(buf + pos, layers); buf[pos++] = '|';
        pos += intToStr(buf + pos, inSize); buf[pos++] = '|';
        pos += intToStr(buf + pos, hidSize); buf[pos++] = '|';
        pos += intToStr(buf + pos, outSize); buf[pos++] = '|';

        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            int cols = (L == layers) ? outSize : hidSize;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    pos += floatToStr(buf + pos, W[L][i][j]);
                    if (!(L == layers && i == rows - 1 && j == cols - 1))
                        buf[pos++] = ',';
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

        delete[] I; delete[] O;
        for (int L = 0; L < layers; L++) delete[] H[L];
        delete[] H;
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            for (int a = 0; a < rows; a++) delete[] W[L][a];
            delete[] W[L];
        }
        delete[] W;
        for (int L = 0; L < layers + 1; L++) delete[] gradBuf[L];
        delete[] gradBuf;
        delete[] OE;
        delete[] OD;

        layers = vals[0]; inSize = vals[1]; hidSize = vals[2]; outSize = vals[3];

        I = new float[inSize];
        O = new float[outSize];
        H = new float*[layers];
        for (int L = 0; L < layers; L++) H[L] = new float[hidSize];
        W = new float**[layers + 1];
        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            int cols = (L == layers) ? outSize : hidSize;
            W[L] = new float*[rows];
            for (int a = 0; a < rows; a++) W[L][a] = new float[cols];
        }
        gradBuf = new float*[layers + 1];
        for (int L = 0; L < layers + 1; L++) {
            int sz = (L == 0) ? inSize : hidSize;
            if (L == layers) sz = outSize;
            gradBuf[L] = new float[sz];
        }
        OE = new float[outSize];
        OD = new float[outSize];

        for (int L = 0; L < layers + 1; L++) {
            int rows = (L == 0) ? inSize : hidSize;
            int cols = (L == layers) ? outSize : hidSize;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) {
                    int sign = 1;
                    if (data[p] == '-') { sign = -1; p++; }
                    float v = 0.0f;
                    while (data[p] >= '0' && data[p] <= '9') { v = v * 10.0f + (data[p] - '0'); p++; }
                    if (data[p] == '.') {
                        p++;
                        float f = 0.1f;
                        while (data[p] >= '0' && data[p] <= '9') { v += (data[p] - '0') * f; f *= 0.1f; p++; }
                    }
                    W[L][i][j] = v * sign;
                    if (data[p] == ',') p++;
                }
        }
    }

private:
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

    static int floatToStr(char* buf, float v) {
        int n = 0;
        if (v < 0) { buf[n++] = '-'; v = -v; }
        long long ip = (long long)v;
        float frac = v - (float)ip;
        n += intToStr(buf + n, (int)ip);
        buf[n++] = '.';
        frac *= 1000000.0f;
        long long fp = (long long)(frac + 0.5f);
        char tmp[8];
        int t = 0;
        if (fp == 0) tmp[t++] = '0';
        else while (fp > 0) { tmp[t++] = char('0' + fp % 10); fp /= 10; }
        while (t < 6) tmp[t++] = '0';
        for (int i = t - 1; i >= 0; i--) buf[n++] = tmp[i];
        return n;
    }
};